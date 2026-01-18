import streamlit as st
import pandas as pd
import requests
import pydeck as pdk
from google.transit import gtfs_realtime_pb2

GTFS_RT_URL = "https://api.data.gov.my/gtfs-realtime/vehicle-position/prasarana?category=rapid-bus-kl"

st.set_page_config(page_title="TransitPulse Lite", layout="wide")
st.title("🚌 TransitPulse Lite — Live Vehicle Positions (MVP)")

@st.cache_data(ttl=25)
def fetch_rt(url: str) -> pd.DataFrame:
    r = requests.get(url, timeout=30)
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

    df = pd.DataFrame(rows).dropna(subset=["lat", "lon"])
    return df

df = fetch_rt(GTFS_RT_URL)
st.caption(f"Vehicles: {len(df)} | Feed updates about every ~30 seconds")
st.dataframe(df.head(30), use_container_width=True)

if len(df):
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df = df.dropna(subset=["lat","lon"])

    layer = pdk.Layer(
        "ScatterplotLayer",
        df,
        get_position=["lon", "lat"],
        get_radius=120,
        pickable=True,
    )

    view = pdk.ViewState(
        latitude=float(df["lat"].mean()),
        longitude=float(df["lon"].mean()),
        zoom=10,
    )

    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=view,
        map_provider="carto",
        map_style="light",
        tooltip={"text": "{vehicle_id}\ntrip={trip_id}\nspeed={speed}"},
    )

    st.pydeck_chart(deck, use_container_width=True)
else:
    st.warning("No vehicle data returned right now. Try refresh.")
