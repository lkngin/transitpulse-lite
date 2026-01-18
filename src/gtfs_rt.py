import requests
import pandas as pd
from google.transit import gtfs_realtime_pb2

def fetch_vehicle_positions(gtfs_rt_url: str) -> pd.DataFrame:
    r = requests.get(gtfs_rt_url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
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
            "entity_id": getattr(ent, "id", None),  # ✅ unique key
            "vehicle_id": getattr(v.vehicle, "id", None),
            "trip_id": getattr(v.trip, "trip_id", None),
            "rt_route_id": getattr(v.trip, "route_id", None), # Extract route_id directly from RT
            "lat": getattr(pos, "latitude", None),
            "lon": getattr(pos, "longitude", None),
            "speed": getattr(pos, "speed", None),
            "timestamp": getattr(v, "timestamp", None),
        })

    df = pd.DataFrame(rows)

    df["trip_id"] = df["trip_id"].astype("string")
    df["entity_id"] = df["entity_id"].astype("string")
    df["vehicle_id"] = df["vehicle_id"].astype("string")

    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df = df.dropna(subset=["lat", "lon"])
    return df

