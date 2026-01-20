"""
Configuration module for TransitPulse Lite
"""
from zoneinfo import ZoneInfo
from datetime import datetime
import os

class APIConfig:
    """API-related configurations"""
    GTFS_STATIC_URL = "https://api.data.gov.my/gtfs-static/prasarana/?category=rapid-bus-kl"
    GTFS_RT_URL = "https://api.data.gov.my/gtfs-realtime/vehicle-position/prasarana/?category=rapid-bus-kl"
    # GTFS_RT_URL = "https://api.data.gov.my/gtfs-realtime/vehicle-position/prasarana/?category=rapid-bus-mrtfeeder"

class RegionConfig:
    """Regional configurations"""
    AREA_NAME = "Kuala Lumpur / Selangor (Rapid Bus KL)"
    TIMEZONE = ZoneInfo("Asia/Kuala_Lumpur")
    TZ_LABEL = "Kuala Lumpur (GMT+8)"

class UIConfig:
    """UI-related configurations"""
    COLORS = {
        "BUNCHED": [255, 0, 0, 190],
        "GAP": [255, 140, 0, 200],
        "NO_DATA": [120, 120, 120, 160],
        "NORMAL": [0, 128, 255, 170],
        "STOP_FILL": [0, 0, 0, 200],
        "HEADWAY_LINE": [255, 0, 255, 120],
        "CIRCULAR_HEADWAY_LINE": [255, 0, 255, 60],
        "ARROW_PINK": [255, 0, 255, 200],
        "TEXT_COLOR": [0, 0, 0, 255],
        "BACKGROUND_COLOR": [255, 255, 255, 200]
    }
    
    CHART_COLORS = {
        "BUNCHED": "#FF0000",
        "GAP": "#FF8C00",
        "ACTIVE": "#1E90FF"
    }
    
    VEHICLE_ICON_URL = "https://img.icons8.com/ios-filled/50/000000/bus.png"
    HEADWAY_ARROW_URL = "https://cdn-icons-png.flaticon.com/512/60/60995.png"
    VEHICLE_ICON_SIZE = 30
    TEXT_LABEL_SIZE = 15
    LINE_LAYER_WIDTH = 3

class AppConfig:
    """Application-wide configurations"""
    AUTO_REFRESH_INTERVAL_MS = 30_000
    VERSION = "1.0.0"
    AUTHOR = "Loo Keen Ngin"
    PUBLISH_DATE = datetime(2026, 1, 18)
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
from src.utils import now_kl, fmt_kl, iso_kl, calculate_initial_compass_bearing, make_color, create_text_layer
from src.services import (
    process_core_pipeline, 
    detect_anomalies, 
    compute_neighbors, 
    prepare_visualization_data, 
    aggregate_health_metrics,
    create_map_layers
)
from config import APIConfig, RegionConfig, AppConfig, UIConfig

# -----------------------
# URLs
# -----------------------
GTFS_STATIC_URL = APIConfig.GTFS_STATIC_URL
GTFS_RT_URL = APIConfig.GTFS_RT_URL
# GTFS_RT_URL = "https://api.data.gov.my/gtfs-realtime/vehicle-position/prasarana/?category=rapid-bus-mrtfeeder"

AREA_NAME = RegionConfig.AREA_NAME

# -----------------------
# Timezone (KL)
# -----------------------
KL_TZ = RegionConfig.TIMEZONE
TZ_LABEL = RegionConfig.TZ_LABEL

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
    st.session_state["shape_cache_v3"] = build_shape_cache(shapes, downsample_step=AppConfig.SHAPE_CACHE_DOWNSAMPLE_STEP)
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
top_routes = route_counts.head(AppConfig.TOP_ROUTES_COUNT).index.tolist()
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
work = process_core_pipeline(work, shape_cache, circular)

# Anomaly Detection
if "route_history" not in st.session_state:
    st.session_state["route_history"] = {}

work2, st.session_state["route_history"] = detect_anomalies(work, st.session_state["route_history"])

if show_only_flagged:
    work2 = work2[work2["anomaly_label"].isin(["BUNCHED", "GAP"])]

work2 = prepare_visualization_data(work2, show_only_flagged)

work2 = compute_neighbors(work2, circular)

# -----------------------
# Health Table Aggregation
# -----------------------
health = aggregate_health_metrics(work2)

if "health_history" not in st.session_state:
    st.session_state["health_history"] = []
hh = health.copy()
hh["ts"] = pd.Timestamp.now(tz="Asia/Kuala_Lumpur")
st.session_state["health_history"].append(hh[["ts", "route_id", "route_name", "active", "bunched", "gap"]])
st.session_state["health_history"] = st.session_state["health_history"][-AppConfig.HEALTH_HISTORY_LIMIT:]

# -----------------------
# Layout
# -----------------------
c_map, c_llm = st.columns([2, 1])

with c_map:
    st.markdown("**Legend:** 🔴 BUNCHED  |  🟠 GAP  |  🔵 NORMAL  |  ⚪ NO_DATA")
    if len(work2) > 0:
        # Define icon data
        icon_url = UIConfig.VEHICLE_ICON_URL
        work2["icon_data"] = [{
            "url": icon_url,
            "width": 128,
            "height": 128,
            "anchorY": 128,
            "mask": True  # Enable coloring
        } for _ in range(len(work2))]

        layers = create_map_layers(
            work2, 
            trips, 
            stops, 
            picked_route, 
            show_stops, 
            show_headway, 
            circular,
            shape_cache
        )

        # Labels for vehicles
        if show_vehicle_ids:
            text_layer = create_text_layer(work2)
            layers.append(text_layer)

        # Manage Map View State (Preserve view on refresh unless route changes)
        if "map_view_state" not in st.session_state or st.session_state.get("last_picked_route") != picked_route:
            # Recalculate center only on route switch
            lat_mean = float(work2["lat"].mean()) if not work2.empty else RegionConfig.DEFAULT_LATITUDE
            lon_mean = float(work2["lon"].mean()) if not work2.empty else RegionConfig.DEFAULT_LONGITUDE
            st.session_state["map_view_state"] = pdk.ViewState(
                latitude=lat_mean, 
                longitude=lon_mean, 
                zoom=RegionConfig.DEFAULT_ZOOM_LEVEL
            )
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
                color=[UIConfig.CHART_COLORS["BUNCHED"], UIConfig.CHART_COLORS["GAP"], UIConfig.CHART_COLORS["ACTIVE"]]
            )

if auto_refresh:
    # Fetch new data on the next run
    st.session_state['rt_data'] = fetch_rt_manual(GTFS_RT_URL)
    st_autorefresh(interval=AppConfig.AUTO_REFRESH_INTERVAL_MS, key='autorefresh')
