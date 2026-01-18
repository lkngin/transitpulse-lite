import json
import time
import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk
from streamlit_autorefresh import st_autorefresh

from datetime import datetime
from zoneinfo import ZoneInfo

from src.gtfs_static import load_gtfs_static
from src.gtfs_rt import fetch_vehicle_positions
from src.features import (
    attach_routes,
    build_shape_cache,
    compute_progress_along_shape,
    compute_headway_proxy,
)
from src.anomaly import label_anomalies_isoforest
from src.llm import groq_chat
from src.prompts import SYSTEM_PROMPT, USER_TEMPLATE

# -----------------------
# URLs
# -----------------------
GTFS_STATIC_URL = "https://api.data.gov.my/gtfs-static/prasarana/?category=rapid-bus-kl"
GTFS_RT_URL = "https://api.data.gov.my/gtfs-realtime/vehicle-position/prasarana/?category=rapid-bus-kl"
AREA_NAME = "Kuala Lumpur / Selangor (Rapid Bus KL)"

# -----------------------
# Timezone (KL)
# -----------------------
KL_TZ = ZoneInfo("Asia/Kuala_Lumpur")
TZ_LABEL = "Kuala Lumpur (GMT+8)"

def now_kl() -> datetime:
    return datetime.now(KL_TZ)

def fmt_kl(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S") + " GMT+8"

def iso_kl(dt: datetime) -> str:
    return dt.isoformat(timespec="seconds")

def calculate_initial_compass_bearing(point1, point2):
    """
    Calculates the bearing between two points.
    The formulae used is the following:
        θ = atan2(sin(Δlong).cos(lat2),
                  cos(lat1).sin(lat2) − sin(lat1).cos(lat2).cos(Δlong))
    :Parameters:
      - point1: The tuple representing the (lat, lon) of the first point.
      - point2: The tuple representing the (lat, lon) of the second point.
    :Returns:
      The bearing in degrees
    """
    import math
    if (type(point1) != tuple) or (type(point2) != tuple):
        raise TypeError("Only tuples are supported as arguments")

    lat1 = math.radians(point1[0])
    lat2 = math.radians(point2[0])
    diffLong = math.radians(point2[1] - point1[1])

    x = math.sin(diffLong) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1) * math.cos(lat2) * math.cos(diffLong))

    initial_bearing = math.atan2(x, y)
    initial_bearing = math.degrees(initial_bearing)
    compass_bearing = (initial_bearing + 360) % 360
    return compass_bearing

# -----------------------
# Streamlit UI Setup
# -----------------------
st.set_page_config(page_title="TransitPulse Lite", layout="wide")
st.title("🚌 TransitPulse Lite — Live Vehicles + V2 Headway Bunching (ML) + Rider Updates (LLM)")
st.caption(f"🕒 All times shown are {TZ_LABEL} (Asia/Kuala_Lumpur).")

# -----------------------
# Data Loading & Caching
# -----------------------
# -----------------------
# Data Loading & Caching
# -----------------------
@st.cache_resource(show_spinner=False)
def cached_static_v4(url):
    return load_gtfs_static(url)

# Note: We remove the cache decorator here to control it manually via session state/time
def fetch_rt_manual(url):
    return fetch_vehicle_positions(url)

routes, trips, shapes, stops = cached_static_v4(GTFS_STATIC_URL)

# RT Data Logic
if "rt_data" not in st.session_state:
    st.session_state["rt_data"] = fetch_rt_manual(GTFS_RT_URL)
    st.session_state["last_fetch"] = time.time()

# If auto-refresh is enabled OR user manually refreshes (rerun), we check if we need new data.
# However, Streamlit reruns on any interaction. We only want to fetch NEW data if enough time passed 
# AND auto-refresh is on, OR if it's a manual reload (which writes to session state).

# Reworked: Validating the user request
# 1. Static is loaded once.
# 2. RT is loaded from session state.
# 3. If auto_refresh is ON, we re-fetch RT data.

rt = st.session_state["rt_data"]
df = attach_routes(rt, trips, routes)

# -----------------------
# Streamlit UI Setup
# -----------------------
if "picked_route" not in st.session_state:
    st.session_state["picked_route"] = "(All Top Routes)"

with st.sidebar:
    st.subheader("Controls")
    auto_refresh = st.toggle("Auto-refresh (rerun every 30s)", value=False)
    show_only_flagged = st.toggle("Show only flagged (BUNCHED/GAP)", value=False)
    circular = st.toggle("Treat routes as circular (wrap-around headway)", value=False)

    lang = st.selectbox("Output language", ["BM + EN", "EN only", "BM only"], index=0)
    
    if "picked_route" in st.session_state and st.session_state["picked_route"] != "(All Top Routes)":
        rid = st.session_state["picked_route"]
        relevant_trips = trips[trips["route_id"].astype(str) == str(rid)]
        s_ids = relevant_trips["shape_id"].unique()
        s_count = len(s_ids)
        
        # Check cache presence
        in_cache = sum(1 for sid in s_ids if str(sid) in st.session_state.get("shape_cache_v2", {}))
        st.caption(f"🔍 Route {rid}: {s_count} shape(s) found. ({in_cache} in cache)")

# Build shape cache once per session
if "shape_cache_v3" not in st.session_state:
    st.session_state["shape_cache_v3"] = build_shape_cache(shapes, downsample_step=8)
shape_cache = st.session_state["shape_cache_v3"]

# -----------------------
# Route Availability
# -----------------------
route_counts = (
    df[df["route_id"] != "UNKNOWN"]
    .groupby("route_id")["entity_id"]
    .count()
    .sort_values(ascending=False)
)
top_routes = route_counts.head(15).index.tolist()
route_options = ["(All Top Routes)"] + top_routes
if (df["route_id"] == "UNKNOWN").any():
    route_options += ["UNKNOWN"]

# -----------------------
# Top Bar (Metrics & Refresh info)
# -----------------------
c1, c2, c3 = st.columns([1.1, 1.1, 1.8])
with c2:
    st.metric("Vehicles (current)", len(df))
with c3:
    ts_dt = now_kl()
    ts_display = fmt_kl(ts_dt)
    ts_iso = iso_kl(ts_dt)
    st.info(f"Area: {AREA_NAME} | Last refresh: {ts_display}")

with c1:
    curr_route = st.session_state["picked_route"]
    try:
        r_idx = route_options.index(curr_route)
    except ValueError:
        r_idx = 0
    picked_route = st.selectbox("Route filter", route_options, index=r_idx, key="route_select")
    st.session_state["picked_route"] = picked_route

# -----------------------
# Core Processing Pipeline
# -----------------------
work = df[df["route_id"].isin(top_routes)] if picked_route == "(All Top Routes)" else df[df["route_id"] == picked_route]
work = compute_progress_along_shape(work, shape_cache)
work = compute_headway_proxy(work, circular=circular)

# Fallback for missing direction_id
missing = work["headway_m"].isna()
if missing.any() and "direction_id" in work.columns:
    tmp_all = work.copy().drop(columns=["direction_id"])
    tmp_all = compute_headway_proxy(tmp_all, circular=circular)
    work.loc[missing, "headway_m"] = tmp_all.loc[missing, "headway_m"]

# Tooltip Metadata
work["timestamp_utc"] = pd.to_datetime(work["timestamp"], unit="s", utc=True, errors="coerce")
work["timestamp_kl"] = work["timestamp_utc"].dt.tz_convert("Asia/Kuala_Lumpur")
work["timestamp_kl_str"] = work["timestamp_kl"].dt.strftime("%Y-%m-%d %H:%M:%S GMT+8")

# Anomaly Detection
if "route_history" not in st.session_state:
    st.session_state["route_history"] = {}

out_frames = []
for rid, g in work.groupby("route_id"):
    hist = st.session_state["route_history"].get(rid)
    g2, new_hist = label_anomalies_isoforest(g, hist)
    st.session_state["route_history"][rid] = new_hist
    out_frames.append(g2)

work2 = pd.concat(out_frames, ignore_index=True) if out_frames else work
if show_only_flagged:
    work2 = work2[work2["anomaly_label"].isin(["BUNCHED", "GAP"])]

work2["headway_str"] = work2["headway_m"].apply(lambda x: "" if pd.isna(x) else f"{x:.0f}")
def make_color(label: str):
    if label == "BUNCHED": return [255, 0, 0, 190]
    if label == "GAP": return [255, 165, 0, 190]
    if label == "NO_DATA": return [120, 120, 120, 160]
    return [0, 128, 255, 170]
work2["color"] = work2["anomaly_label"].apply(make_color)
work2["val_emoji"] = "🚌"

# -----------------------
# Health Table Aggregation
# -----------------------
health = (
    work2.groupby(["route_id", "route_name"])
    .agg(
        active=("entity_id", "count"),
        bunched=("anomaly_label", lambda s: (s == "BUNCHED").sum()),
        gap=("anomaly_label", lambda s: (s == "GAP").sum()),
        flagged=("flagged", lambda s: int(np.nansum(pd.Series(s).fillna(False)))),
    )
    .reset_index()
    .sort_values(["bunched", "gap", "active"], ascending=False)
)

if "health_history" not in st.session_state:
    st.session_state["health_history"] = []
hh = health.copy()
hh["ts"] = pd.Timestamp.now(tz="Asia/Kuala_Lumpur")
st.session_state["health_history"].append(hh[["ts", "route_id", "route_name", "active", "bunched", "gap"]])
st.session_state["health_history"] = st.session_state["health_history"][-40:]

# -----------------------
# Layout
# -----------------------
left, right = st.columns([1.35, 1.0])

with left:
    st.markdown("**Legend:** 🔴 BUNCHED  |  🟠 GAP  |  🔵 NORMAL  |  ⚪ NO_DATA")
    if len(work2) > 0:
        # Define icon data
        icon_url = "https://img.icons8.com/ios-filled/50/000000/bus.png"
        work2["icon_data"] = [{
            "url": icon_url,
            "width": 128,
            "height": 128,
            "anchorY": 128,
            "mask": True  # Enable coloring
        } for _ in range(len(work2))]

        # Precompute tooltip text for vehicles
        work2["tooltip_text"] = work2.apply(
            lambda x: f"{x['route_name']}\nvehicle={x['vehicle_id']}\n{x['anomaly_label']}\nheadway={x.get('headway_str','')}m", 
            axis=1
        )

        v_layer = pdk.Layer(
            "IconLayer",
            work2,
            get_position=["lon", "lat"],
            get_icon="icon_data",
            get_size=30,
            get_color="color",
            pickable=True,
        )
        # We start with empty layers and add background elements first
        layers = []

        # Detailed Overlays for Selected Route
        if picked_route != "(All Top Routes)" and picked_route != "UNKNOWN":
            target_ids = trips[trips["route_id"].astype(str) == str(picked_route)]
            
            # Paths
            for sid in target_ids["shape_id"].unique():
                sid_str = str(sid)
                if sid_str in st.session_state["shape_cache_v3"]:
                    sc = st.session_state["shape_cache_v3"][sid_str]
                    # Convert numpy arrays to python lists for JSON serialization
                    path_coords = list(zip(sc["lon"].tolist(), sc["lat"].tolist()))
                    layers.append(pdk.Layer(
                        "PathLayer",
                        [{"path": path_coords}],
                        get_path="path", get_color=[0, 100, 255, 120], width_min_pixels=4,
                    ))

                    # 1a. Direction Arrows (Sampled)
                    arrow_data = []
                    # Sample every 20th point to avoid clutter
                    lats = sc["lat"]
                    lons = sc["lon"]
                    step = 20
                    for i in range(0, len(lats) - step, step):
                        p1 = (lats[i], lons[i])
                        p2 = (lats[i+5], lons[i+5]) # Look ahead 5 points for smoother bearing
                        angle = calculate_initial_compass_bearing(p1, p2)
                        # -angle because pydeck rotation is counter-clockwise? No, usually standard bearing works if icon is pointing up.
                        # Icon: Simple chevron pointing UP. 0 deg = Up/North.
                        arrow_data.append({
                            "lat": float(lats[i]), 
                            "lon": float(lons[i]), 
                            "angle": -angle # Adjust sign for Pydeck if needed (often requires negative for standard compass)
                        })

                    if arrow_data:
                        arrow_url = "https://img.icons8.com/ios-filled/50/000000/sort-up.png"
                        layers.append(pdk.Layer(
                            "IconLayer",
                            arrow_data,
                            get_position=["lon", "lat"],
                            get_icon={"url": arrow_url, "width": 128, "height": 128, "anchorY": 128, "mask": True},
                            get_size=15,
                            get_color=[0, 0, 100], # Dark blue arrows
                            get_angle="angle",
                            pickable=False
                        ))
            
            # Stops
            r_stops = stops[stops["route_id"].astype(str) == str(picked_route)].copy()
            if not r_stops.empty:
                # Precompute tooltip text for stops
                r_stops["tooltip_text"] = r_stops.apply(
                    lambda x: f"Stop: {x['stop_name']}\nID: {x['stop_id']}", axis=1
                )
                layers.append(pdk.Layer(
                    "ScatterplotLayer", r_stops,
                    get_position=["stop_lon", "stop_lat"],
                    get_fill_color=[0, 0, 0, 200], get_radius=30, pickable=True,
                ))

            # Headway Lines
            hw_lines = []
            for _, g in work2.groupby("direction_id" if "direction_id" in work2 else "route_id"):
                if len(g) < 2: continue
                gg = g.sort_values("progress_m")
                coords = gg[["lon", "lat"]].to_numpy()
                for i in range(len(coords)-1):
                    hw_lines.append({"start": coords[i].tolist(), "end": coords[i+1].tolist(), "color": [255, 0, 255, 120]})
                if circular:
                    hw_lines.append({"start": coords[-1].tolist(), "end": coords[0].tolist(), "color": [255, 0, 255, 60]})
            if hw_lines:
                layers.append(pdk.Layer(
                    "LineLayer", hw_lines,
                    get_source_position="start", get_target_position="end",
                    get_color="color", get_width=3
                ))
        
        # Add the vehicle (icon) layer LAST so it appears on top
        layers.append(v_layer)

        view = pdk.ViewState(latitude=float(work2["lat"].mean()), longitude=float(work2["lon"].mean()), zoom=11)
        st.pydeck_chart(pdk.Deck(
            layers=layers, initial_view_state=view, map_provider="carto", map_style="light",
            tooltip={"text": "{tooltip_text}"}
        ), width="stretch")
    else:
        st.warning("No vehicles to show.")

with right:
    st.subheader("Route Health (Selection Enabled)")
    event = st.dataframe(
        health, width="stretch", height=300, 
        on_select="rerun", selection_mode="single-row", key="h_table"
    )
    
    sel_rows = event.selection.get("rows", [])
    if sel_rows:
        row_route = str(health.iloc[sel_rows[0]]["route_id"])
        if row_route != st.session_state["picked_route"]:
            st.session_state["picked_route"] = row_route
            st.rerun()

    st.download_button("Download current health (CSV)", data=health.to_csv(index=False).encode("utf-8"), file_name="health.csv")

    st.subheader("Health Trend")
    hist = pd.concat(st.session_state["health_history"], ignore_index=True) if st.session_state["health_history"] else None
    if hist is not None and not hist.empty:
        r_list = hist["route_id"].unique().tolist()
        if r_list:
            r_sel_trend = st.selectbox("Trend route", r_list, key="trend_sel")
            h2 = hist[hist["route_id"] == r_sel_trend].sort_values("ts")
            st.line_chart(h2.set_index("ts")[["bunched", "gap", "active"]])

    st.subheader("Generate Rider Update (LLM)")
    if not health.empty:
        route_pick_llm = st.selectbox("Route for update", health["route_id"].tolist(), index=0, key="llm_sel")
        row = health[health["route_id"] == route_pick_llm].iloc[0]
        if st.button("Generate update"):
            if not enable_llm: st.warning("Enable LLM in the sidebar first.")
            else:
                user_msg = USER_TEMPLATE.format(
                    area=AREA_NAME, ts=ts_display, route_name=str(row["route_name"]),
                    active=int(row["active"]), bunched=int(row["bunched"]), gap=int(row["gap"]),
                )
                try:
                    content = groq_chat([{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_msg}])
                    j = json.loads(content.strip())
                    if lang == "BM + EN":
                        st.markdown(f"**EN:** {j.get('en_short','')}\n\n**BM:** {j.get('bm_short','')}")
                    elif lang == "EN only": st.write(j.get('en_detail',''))
                    else: st.write(j.get('bm_detail',''))
                except Exception as e: st.error(f"Error: {e}")

if auto_refresh:
    # Fetch new data on the next run
    st.session_state["rt_data"] = fetch_rt_manual(GTFS_RT_URL)
    st_autorefresh(interval=30_000, key="autorefresh")
