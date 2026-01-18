import requests
import pandas as pd
from google.transit import gtfs_realtime_pb2

def fetch_vehicle_positions(gtfs_rt_url: str) -> pd.DataFrame:
    """
    Fetch vehicle positions from GTFS Realtime feed.
    Returns empty DataFrame with proper structure if API fails.
    """
    try:
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
                "entity_id": getattr(ent, "id", None),
                "vehicle_id": getattr(v.vehicle, "id", None),
                "trip_id": getattr(v.trip, "trip_id", None),
                "rt_route_id": getattr(v.trip, "route_id", None),
                "lat": getattr(pos, "latitude", None),
                "lon": getattr(pos, "longitude", None),
                "speed": getattr(pos, "speed", None),
                "timestamp": getattr(v, "timestamp", None),
            })

        if not rows:
            # No vehicles in feed
            return _empty_vehicle_df()

        df = pd.DataFrame(rows)
        df["trip_id"] = df["trip_id"].astype("string")
        df["entity_id"] = df["entity_id"].astype("string")
        df["vehicle_id"] = df["vehicle_id"].astype("string")

        df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
        df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
        df = df.dropna(subset=["lat", "lon"])
        return df

    except requests.exceptions.HTTPError as e:
        print(f"⚠️ GTFS RT API HTTP Error: {e.response.status_code} - {e}")
        return _empty_vehicle_df()
    except requests.exceptions.RequestException as e:
        print(f"⚠️ GTFS RT API Network Error: {e}")
        return _empty_vehicle_df()
    except Exception as e:
        print(f"⚠️ GTFS RT Parsing Error: {e}")
        return _empty_vehicle_df()


def _empty_vehicle_df() -> pd.DataFrame:
    """Return empty DataFrame with expected schema."""
    return pd.DataFrame(columns=[
        "entity_id", "vehicle_id", "trip_id", "rt_route_id",
        "lat", "lon", "speed", "timestamp"
    ])

