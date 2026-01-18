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
# GTFS_RT_URL = "https://api.data.gov.my/gtfs-realtime/vehicle-position/prasarana/?category=rapid-bus-mrtfeeder"

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
st.title("🚌 TransitPulse Lite — Live Vehicles (from data.gov.my) + Headway Bunching (ML) + Rider Updates (LLM)")
st.caption("Published by: Loo Keen Ngin, 18/01/2026")
st.caption(f"🕒 All times shown are {TZ_LABEL} (Asia/Kuala_Lumpur).")

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

# Show warning if API failed
if rt.empty:
    st.warning("⚠️ **GTFS Realtime API is currently unavailable.** The data.gov.my service may be down or experiencing issues. Please try again later.")

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
    show_vehicle_ids = st.toggle("Show vehicle IDs on map", value=True)
    show_stops = st.toggle("Show stops", value=True)
    show_headway = st.toggle("Show headway lines", value=True)
    circular = st.toggle("Treat routes as circular (wrap-around headway)", value=False)
    enable_llm = st.toggle("Enable LLM rider updates", value=False)

    lang = st.selectbox("Output language", ["BM + EN", "EN only", "BM only"], index=0)
    
    st.divider()
    matched_count = len(df[df["route_id"] != "UNKNOWN"])
    total_count = len(df)
    st.metric("Total Vehicles", f"{total_count}")
    st.caption(f"Matched to Route: {matched_count} | Unmatched: {total_count - matched_count}")

    with st.expander("ℹ️ How bunching is considered?"):
        st.markdown("""
        **Anomaly Detection (Hybrid ML):**
        1. **Median Headway** (distance along route shape) is calculated for the active route.
        2. **🔴 BUNCHED**: Headway is **< 45%** of the median.
        3. **🟠 GAP**: Headway is **> 250%** of the median.
        4. **⚠️ Flagged**: Anomaly detected by the **ML model (Isolation Forest)**, independent of the rules above.
        """)

    if "picked_route" in st.session_state and st.session_state["picked_route"] != "(All Top Routes)":
        rid = st.session_state["picked_route"]
        relevant_trips = trips[trips["route_id"].astype(str) == str(rid)]
        s_ids = relevant_trips["shape_id"].unique()
        s_count = len(s_ids)
        
        # Check cache presence
        # Use shape_cache_v3 as per session state initialization logic
        in_cache = sum(1 for sid in s_ids if str(sid) in st.session_state.get("shape_cache_v3", {}))
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
    if label == "GAP": return [255, 140, 0, 200]
    if label == "NO_DATA": return [120, 120, 120, 160]
    return [0, 128, 255, 170]
work2["color"] = work2["anomaly_label"].apply(make_color)
work2["val_emoji"] = "🚌"

# Compute Neighbors (Ahead/Behind)
work2 = work2.sort_values(["route_id", "progress_m"])
grp_cols = ["route_id", "direction_id"] if "direction_id" in work2.columns else ["route_id"]
g = work2.groupby(grp_cols)
if circular:
    # Wrap-around logic for IDs
    ahead = g["vehicle_id"].shift(-1)
    # Fill the last item (NaN) with the First item of the group
    first_ids = g["vehicle_id"].transform("first")
    work2["ahead_id"] = ahead.fillna(first_ids)

    behind = g["vehicle_id"].shift(1)
    # Fill the first item (NaN) with the Last item of the group
    last_ids = g["vehicle_id"].transform("last")
    work2["behind_id"] = behind.fillna(last_ids)
    
    # For behind distance, we confusingly need the 'headway_m' of the Last Bus?
    # Usually dist_behind for First Bus = Headway of Last Bus (gap from Last -> First).
    # Shift(1) gives headway of previous bus.
    # For First bus, shift(1) is NaN. We need Last Bus's headway (which is the Wrap headway).
    d_behind = g["headway_m"].shift(1)
    last_headways = g["headway_m"].transform("last") # The 'wrap' headway is stored in the Last Bus row
    work2["dist_behind"] = d_behind.fillna(last_headways)

else:
    # Standard Linear Logic
    work2["ahead_id"] = g["vehicle_id"].shift(-1).fillna("-")
    work2["behind_id"] = g["vehicle_id"].shift(1).fillna("-")
    work2["dist_behind"] = g["headway_m"].shift(1).fillna(0)

# Precompute tooltip text for vehicles
work2["tooltip_text"] = work2.apply(
    lambda x: (
        f"{x['route_name']}\n"
        f"Vehicle: {x['vehicle_id']}\n"
        f"Status (to Front): {x['anomaly_label']}\n"
        f"Ahead: {x['ahead_id']} ({x['headway_m']:.0f}m)\n"
        f"Behind: {x['behind_id']} ({x['dist_behind']:.0f}m)\n"
        f"Updated: {x.get('timestamp_kl_str', '')}"
    ), 
    axis=1
)

# -----------------------
# Health Table Aggregation
# -----------------------
health = (
    work2.groupby(["route_id", "route_name"])
    .agg(
        active=("entity_id", "count"),
        bunched=("anomaly_label", lambda s: (s == "BUNCHED").sum()),
        gap=("anomaly_label", lambda s: (s == "GAP").sum()),
        ML_Flag=("flagged", lambda s: int(np.nansum(pd.Series(s).fillna(False)))),
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
# -----------------------
# Layout
# -----------------------
c_map, c_llm = st.columns([2, 1])

with c_map:
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

        # Precompute tooltip text for vehicles (already done above)
        # Note: tooltip_text column exists in work2

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
                        arrow_data.append({
                            "lat": float(lats[i]), 
                            "lon": float(lons[i]), 
                            "angle": -angle 
                        })

                    if arrow_data:
                        arrow_url = "https://img.icons8.com/ios-filled/50/000000/sort-up.png"
                        # Add icon_data key to each item
                        for ad in arrow_data:
                            ad["icon_data"] = {"url": arrow_url, "width": 128, "height": 128, "anchorY": 128, "mask": True}
                        
                        layers.append(pdk.Layer(
                            "IconLayer",
                            arrow_data,
                            get_position=["lon", "lat"],
                            get_icon="icon_data",
                            get_size=15,
                            get_color=[0, 100, 255, 200], # Blue arrows
                            get_angle="angle",
                            pickable=False
                        ))
            
            # Stops
            if show_stops:
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

            # Headway Lines + Arrows
            if show_headway:
                hw_lines = []
                hw_arrows = []
                for _, g in work2.groupby("direction_id" if "direction_id" in work2 else "route_id"):
                    if len(g) < 2: continue
                    gg = g.sort_values("progress_m")
                    coords = gg[["lon", "lat"]].to_numpy() # array of [lon, lat]
                    
                    # Iterate segments
                    for i in range(len(coords)-1):
                        p_start = coords[i] # [lon, lat]
                        p_end = coords[i+1] # [lon, lat]
                        
                        # Line
                        hw_lines.append({
                            "start": p_start.tolist(), 
                            "end": p_end.tolist(), 
                            "color": [255, 0, 255, 120]
                        })
                        
                        # Arrow at Midpoint
                        mid_lon = (p_start[0] + p_end[0]) / 2
                        mid_lat = (p_start[1] + p_end[1]) / 2
                        
                        # Bearing from start(lat,lon) to end(lat,lon) - note coords are [lon, lat]
                        angle = calculate_initial_compass_bearing((p_start[1], p_start[0]), (p_end[1], p_end[0]))
                        hw_arrows.append({
                            "lat": float(mid_lat), "lon": float(mid_lon), "angle": -(angle + 180)
                        })

                    if circular:
                        # Closing loop: Last -> First
                        p_start = coords[-1]
                        p_end = coords[0]
                        
                        # Safety: Only draw circular link if endpoints are close (< 2km approx)
                        # 0.02 degrees is roughly 2.2km
                        if (abs(p_start[0] - p_end[0]) < 0.02) and (abs(p_start[1] - p_end[1]) < 0.02):
                            hw_lines.append({
                                "start": p_start.tolist(), 
                                "end": p_end.tolist(), 
                                "color": [255, 0, 255, 60]
                            })
                            
                            mid_lon = (p_start[0] + p_end[0]) / 2
                            mid_lat = (p_start[1] + p_end[1]) / 2
                            angle = calculate_initial_compass_bearing((p_start[1], p_start[0]), (p_end[1], p_end[0]))
                            hw_arrows.append({
                                "lat": float(mid_lat), "lon": float(mid_lon), "angle": -(angle + 180)
                            })
                        # Else: Route implies linear; drawing a closing link is visually misleading.

                if hw_lines:
                    layers.append(pdk.Layer(
                        "LineLayer", hw_lines,
                        get_source_position="start", get_target_position="end",
                        get_color="color", get_width=3
                    ))
                
                if hw_arrows:
                    st.sidebar.caption(f"DEBUG: {len(hw_arrows)} headway arrows generated.")
                    arrow_url = "https://cdn-icons-png.flaticon.com/512/60/60995.png"
                    # Add icon_data key to each item
                    for ha in hw_arrows:
                        ha["icon_data"] = {"url": arrow_url, "width": 512, "height": 512, "anchorY": 256, "mask": True}

                    layers.append(pdk.Layer(
                        "IconLayer",
                        hw_arrows,
                        get_position=["lon", "lat"],
                        get_icon="icon_data",
                        get_size=50,
                        get_color=[255, 0, 255, 200], # Pink arrows
                        get_angle="angle",
                        pickable=False,
                        billboard=False # Ensure arrows rotate with the map, not the camera
                    ))
        
        # Add the vehicle (icon) layer LAST so it appears on top
        layers.append(v_layer)

        # Labels for vehicles
        if show_vehicle_ids:
            text_layer = pdk.Layer(
                "TextLayer",
                work2,
                get_position=["lon", "lat"],
                get_text="vehicle_id",
                get_color=[0, 0, 0, 255], # Black text
                get_size=15,
                get_alignment_baseline="'bottom'",
                get_pixel_offset=[0, -15], # Shift up slightly
                background=True,
                get_background_color=[255, 255, 255, 200], # White background block
                pickable=False
            )
            layers.append(text_layer)

        # Manage Map View State (Preserve view on refresh unless route changes)
        if "map_view_state" not in st.session_state or st.session_state.get("last_picked_route") != picked_route:
            # Recalculate center only on route switch
            lat_mean = float(work2["lat"].mean()) if not work2.empty else 3.14
            lon_mean = float(work2["lon"].mean()) if not work2.empty else 101.69
            st.session_state["map_view_state"] = pdk.ViewState(latitude=lat_mean, longitude=lon_mean, zoom=11)
            st.session_state["last_picked_route"] = picked_route

        st.pydeck_chart(pdk.Deck(
            layers=layers, initial_view_state=st.session_state["map_view_state"], map_provider="carto", map_style="light",
            tooltip={"text": "{tooltip_text}"}
        ), width="stretch")
    else:
        st.warning("No vehicles to show.")

with c_llm:
    st.subheader("Generate Rider Update (LLM)")
    if not health.empty:
        route_pick_llm = st.selectbox("Route for update", health["route_id"].tolist(), index=0, key="llm_sel")
        row = health[health["route_id"] == route_pick_llm].iloc[0]
        if st.button("Generate update"):
            if not enable_llm: st.warning("Enable LLM in the sidebar first.")
            else:
                # Identify affected buses first
                r_mask = work2["route_id"] == route_pick_llm
                affected_df = work2[r_mask & work2["anomaly_label"].isin(["BUNCHED", "GAP"])]
                affected_txt = "None"
                
                if not affected_df.empty:
                    bus_list = ", ".join([f"{x.vehicle_id} ({x.anomaly_label})" for x in affected_df.itertuples()])
                    affected_txt = bus_list
                    # UI Fallback: Show explicitly
                    st.info(f"**Identified Issues:** {bus_list}")

                # Construct prompt strictly
                user_msg = USER_TEMPLATE.format(
                    area=AREA_NAME, ts=ts_display, route_name=str(row["route_name"]),
                    active=int(row["active"]), bunched=int(row["bunched"]), gap=int(row["gap"]),
                )
                
                # Inject into context clearly
                user_msg = user_msg.replace("- ML indicators:", f"- Affected Buses: {affected_txt}\n- ML indicators:")
                if affected_txt != "None":
                    user_msg += "\n\nREQUIREMENT: You MUST mention the 'Affected Buses' IDs in the 'en_short' and 'bm_short' output."
                try:
                    content = groq_chat([{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_msg}])
                    j = json.loads(content.strip())
                    if lang == "BM + EN":
                        st.markdown(f"**EN:** {j.get('en_short','')}\n\n**BM:** {j.get('bm_short','')}")
                    elif lang == "EN only": st.write(j.get('en_detail',''))
                    else: st.write(j.get('bm_detail',''))
                except Exception as e: st.error(f"Error: {e}")

st.divider()

# Bottom Row
c_health, c_chart = st.columns([1, 1])

with c_health:
    st.subheader("Route Health")
    st.dataframe(
        health, width="stretch", height=300, 
        key="h_table"
    )
    st.download_button("Download current health (CSV)", data=health.to_csv(index=False).encode("utf-8"), file_name="health.csv")

with c_chart:
    st.subheader("Health Trend")
    hist = pd.concat(st.session_state["health_history"], ignore_index=True) if st.session_state["health_history"] else None
    if hist is not None and not hist.empty:
        r_list = hist["route_id"].unique().tolist()
        if r_list:
            # Sync with main filter if possible
            default_ix = 0
            if picked_route in r_list:
                default_ix = r_list.index(picked_route)
            
            # Use dynamic key to force reset when main filter changes
            r_sel_trend = st.selectbox("Trend route", r_list, index=default_ix, key=f"trend_sel_{picked_route}")
            
            h2 = hist[hist["route_id"] == r_sel_trend].sort_values("ts")
            st.line_chart(
                h2.set_index("ts")[["bunched", "gap", "active"]],
                color=["#FF0000", "#FF8C00", "#1E90FF"]
            )

if auto_refresh:
    # Fetch new data on the next run
    st.session_state['rt_data'] = fetch_rt_manual(GTFS_RT_URL)
    st_autorefresh(interval=30_000, key='autorefresh')
