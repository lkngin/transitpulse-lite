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

# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="TransitPulse Lite", layout="wide")
st.title("🚌 TransitPulse Lite — Live Vehicles + V2 Headway Bunching (ML) + Rider Updates (LLM)")
st.caption(f"🕒 All times shown are {TZ_LABEL} (Asia/Kuala_Lumpur).")

with st.sidebar:
    st.subheader("Controls")
    auto_refresh = st.toggle("Auto-refresh (rerun every 30s)", value=False)
    show_only_flagged = st.toggle("Show only flagged (BUNCHED/GAP)", value=False)
    circular = st.toggle("Treat routes as circular (wrap-around headway)", value=False)

    st.divider()
    st.subheader("Optional LLM")
    enable_llm = st.toggle("Enable LLM rider updates", value=False)
    lang = st.selectbox("Output language", ["BM + EN", "EN only", "BM only"], index=0)

# -----------------------
# Caching
# -----------------------
@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def cached_static(url):
    return load_gtfs_static(url)

@st.cache_data(ttl=25, show_spinner=False)
def cached_rt(url):
    return fetch_vehicle_positions(url)

routes, trips, shapes = cached_static(GTFS_STATIC_URL)
rt = cached_rt(GTFS_RT_URL)

df = attach_routes(rt, trips, routes)

# Build shape cache once per session (avoid hashing huge df each rerun)
if "shape_cache" not in st.session_state:
    st.session_state["shape_cache"] = build_shape_cache(shapes, downsample_step=8)

shape_cache = st.session_state["shape_cache"]

# -----------------------
# Route selector
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

c1, c2, c3 = st.columns([1.1, 1.1, 1.8])
with c1:
    picked_route = st.selectbox("Route filter", route_options, index=0)
with c2:
    st.metric("Vehicles (current)", len(df))
with c3:
    ts_dt = now_kl()
    ts_display = fmt_kl(ts_dt)
    ts_iso = iso_kl(ts_dt)
    st.info(f"Area: {AREA_NAME} | Last refresh: {ts_display} | Timezone: {TZ_LABEL}")

work = df[df["route_id"].isin(top_routes)] if picked_route == "(All Top Routes)" else df[df["route_id"] == picked_route]

# -----------------------
# V2: progress along shape -> headway along shape
# -----------------------
work = compute_progress_along_shape(work, shape_cache)
work = compute_headway_proxy(work, circular=circular)

# XXXXXXXXXX

# Fallback: if direction split causes singletons, compute headway ignoring direction for remaining NaNs
# Fallback: if direction split causes singletons, compute headway ignoring direction for remaining NaNs
missing = work["headway_m"].isna()
if missing.any() and "direction_id" in work.columns:
    # Fix: Compute on the FULL set (ignoring direction) to get gaps, then backfill
    # Previous logic only computed on subset, which fails to find neighbors
    tmp_all = work.copy()
    tmp_all = tmp_all.drop(columns=["direction_id"])
    tmp_all = compute_headway_proxy(tmp_all, circular=circular)
    
    # Only fill where it was missing
    work.loc[missing, "headway_m"] = tmp_all.loc[missing, "headway_m"]


st.subheader("Debug V2 Headway")
st.write({
    "rows_work": len(work),
    "missing_trip_id": int(work["trip_id"].isna().sum()) if "trip_id" in work else None,
    "missing_shape_id": int(work["shape_id"].isna().sum()) if "shape_id" in work else None,
    "missing_progress_m": int(work["progress_m"].isna().sum()) if "progress_m" in work else None,
    "missing_headway_m": int(work["headway_m"].isna().sum()) if "headway_m" in work else None,
})
# XXXXXXXXXX

# Convert feed timestamp to KL for tooltip
work["timestamp_utc"] = pd.to_datetime(work["timestamp"], unit="s", utc=True, errors="coerce")
work["timestamp_kl"] = work["timestamp_utc"].dt.tz_convert("Asia/Kuala_Lumpur")
work["timestamp_kl_str"] = work["timestamp_kl"].dt.strftime("%Y-%m-%d %H:%M:%S GMT+8")

# -----------------------
# ML anomaly detection (per route) with rolling history
# -----------------------
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

# strings for tooltip
work2["headway_str"] = work2["headway_m"].apply(lambda x: "" if pd.isna(x) else f"{x:.0f}")

def make_color(label: str):
    if label == "BUNCHED":
        return [255, 0, 0, 190]
    if label == "GAP":
        return [255, 165, 0, 190]
    if label == "NO_DATA":
        return [120, 120, 120, 160]
    return [0, 128, 255, 170]

work2 = work2.copy()
work2["color"] = work2["anomaly_label"].apply(make_color)

# -----------------------
# Route health table
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

# Trend history (KL time)
if "health_history" not in st.session_state:
    st.session_state["health_history"] = []
hh = health.copy()
hh["ts"] = pd.Timestamp.now(tz="Asia/Kuala_Lumpur")
st.session_state["health_history"].append(hh[["ts", "route_id", "route_name", "active", "bunched", "gap"]])
st.session_state["health_history"] = st.session_state["health_history"][-40:]

# -----------------------
# Layout: Map + Table + LLM
# -----------------------
left, right = st.columns([1.35, 1.0])

with left:
    st.markdown("**Legend:** 🔴 BUNCHED  |  🟠 GAP  |  🔵 NORMAL  |  ⚪ NO_DATA")
    if len(work2) > 0:
        layer = pdk.Layer(
            "ScatterplotLayer",
            work2,
            get_position=["lon", "lat"],
            get_fill_color="color",
            get_radius=120,
            pickable=True,
        )
        view = pdk.ViewState(latitude=float(work2["lat"].mean()),
                             longitude=float(work2["lon"].mean()),
                             zoom=10)

        deck = pdk.Deck(
            layers=[layer],
            initial_view_state=view,
            map_provider="carto",
            map_style="light",
            tooltip={"text": "{route_name}\nvehicle={vehicle_id}\n{anomaly_label}\nupdated={timestamp_kl_str}\nheadway={headway_str}m\nscore={anomaly_score}"},
        )
        st.pydeck_chart(deck, width="stretch")
    else:
        st.warning("No vehicles to show for this filter.")

with right:
    st.subheader("Route Health (V2 ML)")
    st.dataframe(health, width="stretch", height=300)

    # Exports with KL time
    health_export = health.copy()
    health_export.insert(0, "export_time_kl", ts_display)
    health_export.insert(1, "export_time_kl_iso", ts_iso)
    health_export.insert(2, "timezone", TZ_LABEL)

    st.download_button(
        "Download current health (CSV)",
        data=health_export.to_csv(index=False).encode("utf-8"),
        file_name="route_health_kl_v2.csv",
        mime="text/csv",
    )

    snapshot = {
        "export_time_kl": ts_display,
        "export_time_kl_iso": ts_iso,
        "timezone": TZ_LABEL,
        "area": AREA_NAME,
        "summary": health.to_dict(orient="records"),
    }

    st.download_button(
        "Download current health (JSON)",
        data=json.dumps(snapshot, indent=2).encode("utf-8"),
        file_name="route_health_kl_v2.json",
        mime="application/json",
    )

    st.subheader("Health Trend (last ~40 refreshes)")
    hist = pd.concat(st.session_state["health_history"], ignore_index=True) if st.session_state["health_history"] else None
    if hist is not None and len(hist):
        route_ids = hist["route_id"].dropna().unique().tolist()
        if route_ids:
            rsel = st.selectbox("Trend route", route_ids)
            h2 = hist[hist["route_id"] == rsel].sort_values("ts")
            st.line_chart(h2.set_index("ts")[["bunched", "gap", "active"]])

    st.subheader("Generate Rider Update (LLM)")
    if len(health) > 0:
        route_pick = st.selectbox("Route for update", health["route_id"].tolist(), index=0)
        row = health[health["route_id"] == route_pick].iloc[0]

        if st.button("Generate update"):
            if not enable_llm:
                st.warning("Enable LLM in the sidebar first.")
            else:
                route_name = str(row["route_name"])
                user_msg = USER_TEMPLATE.format(
                    area=AREA_NAME,
                    ts=ts_display,
                    route_name=route_name,
                    active=int(row["active"]),
                    bunched=int(row["bunched"]),
                    gap=int(row["gap"]),
                )

                try:
                    content = groq_chat(
                        [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": user_msg},
                        ]
                    )
                except RuntimeError as e:
                    st.error(f"Configuration Error: {e}")
                    st.stop()
                except Exception as e:
                    st.error(f"LLM Error: {e}")
                    st.stop()

                try:
                    # j = json.loads(content)
                    j = json.loads(content.strip())

                except Exception:
                    st.error("LLM did not return valid JSON. Raw output:")
                    st.code(content)
                else:
                    if lang == "BM + EN":
                        st.markdown(f"**EN (Short):** {j.get('en_short','')}")
                        st.markdown(f"**EN (Detail):** {j.get('en_detail','')}")
                        st.markdown(f"**BM (Pendek):** {j.get('bm_short','')}")
                        st.markdown(f"**BM (Penuh):** {j.get('bm_detail','')}")
                    elif lang == "EN only":
                        st.markdown(f"**EN (Short):** {j.get('en_short','')}")
                        st.markdown(f"**EN (Detail):** {j.get('en_detail','')}")
                    else:
                        st.markdown(f"**BM (Pendek):** {j.get('bm_short','')}")
                        st.markdown(f"**BM (Penuh):** {j.get('bm_detail','')}")

                    st.markdown("**Actions:**")
                    for a in j.get("actions", []):
                        st.write(f"- {a}")
                    st.caption(j.get("disclaimer", ""))



# -----------------------
# Auto refresh (non-blocking)
# -----------------------
if auto_refresh:
    st_autorefresh(interval=30_000, key="autorefresh")  # 30 seconds

