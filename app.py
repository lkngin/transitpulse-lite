import os
import json
import time
import numpy as np
import pandas as pd
import requests
import streamlit as st
import pydeck as pdk

from google.transit import gtfs_realtime_pb2
from sklearn.ensemble import IsolationForest

from gtfs_static import load_gtfs_static

# -----------------------
# URLs (Malaysia Open API)
# -----------------------
GTFS_STATIC_URL = "https://api.data.gov.my/gtfs-static/prasarana?category=rapid-bus-kl"
GTFS_RT_URL     = "https://api.data.gov.my/gtfs-realtime/vehicle-position/prasarana?category=rapid-bus-kl"

AREA_NAME = "Kuala Lumpur / Selangor (Rapid Bus KL)"

# -----------------------
# Helpers
# -----------------------
EARTH_R = 6371000.0

def haversine_m(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * EARTH_R * np.arcsin(np.sqrt(a))

@st.cache_data(ttl=6*60*60, show_spinner=False)
def cached_static(url):
    return load_gtfs_static(url)

@st.cache_data(ttl=25, show_spinner=False)
def fetch_rt(url: str) -> pd.DataFrame:
    r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()

    feed = gtfs_realtime_pb2.FeedMessage()
    feed.ParseFromString(r.content)

    rows = []
    for ent in feed.entity:
        if not ent.vehicle:
            continue
        v = ent.vehicle
        pos = v.position
        rows.append({
            "vehicle_id": getattr(v.vehicle, "id", None),
            "trip_id": getattr(v.trip, "trip_id", None),
            "lat": getattr(pos, "latitude", None),
            "lon": getattr(pos, "longitude", None),
            "speed": getattr(pos, "speed", None),
            "timestamp": getattr(v, "timestamp", None),
        })

    df = pd.DataFrame(rows)
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df = df.dropna(subset=["lat", "lon"])
    return df

def attach_routes(rt_df: pd.DataFrame, trips: pd.DataFrame, routes: pd.DataFrame) -> pd.DataFrame:
    df = rt_df.merge(trips, on="trip_id", how="left")
    df = df.merge(routes, on="route_id", how="left")

    df["route_short_name"] = df["route_short_name"].fillna("")
    df["route_long_name"]  = df["route_long_name"].fillna("")
    df["route_name"] = (df["route_short_name"] + " " + df["route_long_name"]).str.strip()
    df.loc[df["route_name"] == "", "route_name"] = "UNKNOWN ROUTE"

    df["route_id"] = df["route_id"].fillna("UNKNOWN")
    df["vehicle_id"] = df["vehicle_id"].fillna("unknown_vehicle")
    return df

def nearest_neighbor_distance_per_route(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute nearest-neighbor distance (meters) for each vehicle within its route group.
    O(n^2) per route, but route group sizes are usually small enough for a demo.
    """
    out = df.copy()
    out["nn_dist_m"] = np.nan

    for rid, g in out.groupby("route_id"):
        if g.shape[0] < 2:
            continue

        lat = g["lat"].to_numpy()
        lon = g["lon"].to_numpy()

        # distance matrix
        dmat = np.full((len(g), len(g)), np.inf)
        for i in range(len(g)):
            d = haversine_m(lat[i], lon[i], lat, lon)
            d[i] = np.inf
            dmat[i, :] = d

        nn = np.min(dmat, axis=1)
        out.loc[g.index, "nn_dist_m"] = nn

    return out

def label_anomalies_isoforest(df: pd.DataFrame, history_feat: pd.DataFrame | None):
    """
    Traditional ML (unsupervised): IsolationForest on nearest-neighbor spacing features.
    Combines ML score + rule thresholds for BUNCHED / GAP.
    """
    d = df.copy()
    d = d.dropna(subset=["nn_dist_m"]).copy()
    if d.empty:
        df["anomaly_label"] = "NO_DATA"
        df["anomaly_score"] = np.nan
        df["flagged"] = False
        return df, history_feat

    d["log_nn"] = np.log(d["nn_dist_m"].clip(lower=1.0))

    # training set
    if history_feat is not None and len(history_feat) >= 40:
        train = history_feat.copy()
    else:
        train = d[["log_nn", "nn_dist_m"]].copy()

    X_train = train.to_numpy()
    X_now = d[["log_nn", "nn_dist_m"]].to_numpy()

    model = IsolationForest(
        n_estimators=150,
        contamination=0.08,   # a bit sensitive for demos
        random_state=42
    )
    model.fit(X_train)
    score = model.decision_function(X_now)  # higher = more normal

    d["anomaly_score"] = score

    # rule thresholds relative to median spacing
    med = float(np.nanmedian(d["nn_dist_m"])) if np.isfinite(np.nanmedian(d["nn_dist_m"])) else 0.0
    d["anomaly_label"] = "NORMAL"
    if med > 0:
        d.loc[d["nn_dist_m"] < 0.45 * med, "anomaly_label"] = "BUNCHED"
        d.loc[d["nn_dist_m"] > 2.50 * med, "anomaly_label"] = "GAP"

    # flagged = bottom 10% by anomaly_score
    thresh = np.quantile(d["anomaly_score"], 0.10) if len(d) >= 10 else np.min(d["anomaly_score"])
    d["flagged"] = d["anomaly_score"] <= thresh

    # merge back into original df
    merged = df.merge(
        d[["vehicle_id", "nn_dist_m", "anomaly_score", "anomaly_label", "flagged"]],
        on="vehicle_id",
        how="left",
        suffixes=("", "_y")
    )

    # update history features (keep rolling window)
    new_hist = d[["log_nn", "nn_dist_m"]].copy()
    if history_feat is None:
        history_feat = new_hist.tail(600)
    else:
        history_feat = pd.concat([history_feat, new_hist], ignore_index=True).tail(1200)

    return merged, history_feat

def make_color(label: str):
    # RGBA
    if label == "BUNCHED":
        return [255, 0, 0, 190]
    if label == "GAP":
        return [255, 165, 0, 190]
    if label == "NO_DATA":
        return [120, 120, 120, 160]
    return [0, 128, 255, 170]

def groq_chat(messages, model="llama-3.1-8b-instant", temperature=0.2, max_tokens=350):
    """
    Optional LLM: Groq OpenAI-compatible endpoint.
    Set GROQ_API_KEY in your environment to enable.
    """
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        raise RuntimeError("Missing GROQ_API_KEY")

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": os.getenv("GROQ_MODEL", model),
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    r = requests.post(url, headers=headers, json=payload, timeout=30)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

SYSTEM_PROMPT = """You are TransitPulse Lite, a public transit service-status assistant.
You MUST base your response ONLY on the provided anomaly summary.
Be calm, practical, and non-alarmist. If data is insufficient, say so.

Return STRICT JSON with keys:
en_short, en_detail, bm_short, bm_detail, actions (array), disclaimer
"""

USER_TEMPLATE = """Context:
- Area: {area}
- Timestamp (local): {ts}
- Route: {route_name}
- Active vehicles: {active}
- ML indicators:
  - Bunched vehicles: {bunched}
  - Gap vehicles: {gap}

Task:
1) Explain what riders may experience (uneven waiting time, bunching, service gaps).
2) Provide 3-6 practical actions riders can take.
3) Provide a short disclaimer about realtime GPS & refresh interval.

Output STRICT JSON only.
"""

# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="TransitPulse Lite", layout="wide")
st.title("🚌 TransitPulse Lite — Live Vehicles + Irregularity Detection (ML) + Rider Updates (LLM)")

with st.sidebar:
    st.subheader("Controls")
    auto_refresh = st.toggle("Auto-refresh (~30s)", value=True)
    st.caption("Tip: turn OFF while debugging.")

    st.divider()
    st.subheader("Optional LLM")
    enable_llm = st.toggle("Enable LLM rider updates", value=False)
    lang = st.selectbox("Output language", ["BM + EN", "EN only", "BM only"], index=0)

    st.caption("To enable: set env var GROQ_API_KEY in Codespaces terminal.")

routes, trips = cached_static(GTFS_STATIC_URL)
rt = fetch_rt(GTFS_RT_URL)

df = attach_routes(rt, trips, routes)

# Route list (top routes by active vehicles)
route_counts = df[df["route_id"] != "UNKNOWN"].groupby("route_id")["vehicle_id"].count().sort_values(ascending=False)
top_routes = route_counts.head(15).index.tolist()
route_options = ["(All Top Routes)"] + top_routes + (["UNKNOWN"] if (df["route_id"] == "UNKNOWN").any() else [])

col1, col2, col3 = st.columns([1.1, 1.3, 1.6])
with col1:
    picked_route = st.selectbox("Route filter", route_options, index=0)
with col2:
    st.metric("Vehicles (current)", len(df))
with col3:
    ts = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    st.info(f"Area: {AREA_NAME} | Last refresh: {ts}")

work = df[df["route_id"].isin(top_routes)] if picked_route == "(All Top Routes)" else df[df["route_id"] == picked_route]

# Compute nearest-neighbor distances (spacing proxy)
work = nearest_neighbor_distance_per_route(work)

# Keep rolling history per route in session for better IsolationForest
if "route_history" not in st.session_state:
    st.session_state["route_history"] = {}

out_frames = []
for rid, g in work.groupby("route_id"):
    hist = st.session_state["route_history"].get(rid)
    g2, new_hist = label_anomalies_isoforest(g, hist)
    st.session_state["route_history"][rid] = new_hist
    out_frames.append(g2)

work2 = pd.concat(out_frames, ignore_index=True) if out_frames else work

# Build map colors
work2["color"] = work2["anomaly_label"].apply(make_color)

# Route health table
health = (work2.groupby(["route_id", "route_name"])
          .agg(active=("vehicle_id", "count"),
               bunched=("anomaly_label", lambda s: (s == "BUNCHED").sum()),
               gap=("anomaly_label", lambda s: (s == "GAP").sum()),
               flagged=("flagged", lambda s: int(np.nansum(s.fillna(False)))))
          .reset_index()
          .sort_values(["bunched", "gap", "active"], ascending=False))

left, right = st.columns([1.35, 1.0])

with left:
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
            tooltip={"text": "{route_name}\nvehicle={vehicle_id}\n{anomaly_label}\nnn={nn_dist_m}m\nscore={anomaly_score}"}
        )
        st.pydeck_chart(deck, use_container_width=True)
    else:
        st.warning("No vehicles to show for this filter.")

with right:
    st.subheader("Route Health (ML)")
    st.dataframe(health, use_container_width=True, height=360)

    st.subheader("Generate Rider Update (LLM)")
    if len(health) == 0:
        st.caption("Pick a route with active vehicles first.")
    else:
        route_pick = st.selectbox("Route", health["route_id"].tolist(), index=0)
        row = health[health["route_id"] == route_pick].iloc[0]
        if st.button("Generate update"):
            if not enable_llm:
                st.warning("Enable LLM in the sidebar first.")
            else:
                try:
                    route_name = str(row["route_name"])
                    user_msg = USER_TEMPLATE.format(
                        area=AREA_NAME,
                        ts=ts,
                        route_name=route_name,
                        active=int(row["active"]),
                        bunched=int(row["bunched"]),
                        gap=int(row["gap"]),
                    )

                    content = groq_chat([
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg},
                    ])

                    try:
                        j = json.loads(content)
                    except Exception:
                        st.error("LLM did not return valid JSON. Showing raw output:")
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
                except Exception as e:
                    st.error(str(e))

# Auto refresh
if auto_refresh:
    time.sleep(1)
