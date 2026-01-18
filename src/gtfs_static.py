import io
import zipfile
import requests
import pandas as pd

def load_gtfs_static(gtfs_static_url: str):
    r = requests.get(gtfs_static_url, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()

    z = zipfile.ZipFile(io.BytesIO(r.content))

    routes = pd.read_csv(z.open("routes.txt"), dtype={"route_id": str})
    trips  = pd.read_csv(z.open("trips.txt"), dtype={"trip_id": str, "route_id": str, "shape_id": str})
    shapes = pd.read_csv(z.open("shapes.txt"), dtype={"shape_id": str})

    routes = routes[["route_id", "route_short_name", "route_long_name"]].copy()

    keep_trip_cols = ["trip_id", "route_id", "shape_id"]
    if "direction_id" in trips.columns:
        keep_trip_cols.append("direction_id")
    trips = trips[keep_trip_cols].copy()

    shapes = shapes[["shape_id", "shape_pt_lat", "shape_pt_lon", "shape_pt_sequence"]].copy()

    # New: Load stops and stop_times
    stops = pd.read_csv(z.open("stops.txt"), dtype={"stop_id": str})
    stop_times = pd.read_csv(z.open("stop_times.txt"), dtype={"trip_id": str, "stop_id": str})

    # To keep it light, we only need the association: route_id -> stop_id -> lat/lon
    # We'll merge stops with stop_times and then with trips to get route_id
    stops = stops[["stop_id", "stop_name", "stop_lat", "stop_lon"]].copy()
    stop_times = stop_times[["trip_id", "stop_id", "stop_sequence"]].copy()
    
    # Merge stop_times with trips to get route_ids for each stop
    route_stops = stop_times.merge(trips[["trip_id", "route_id"]], on="trip_id")
    # Drop duplicates so we have unique stops per route (ignoring sequence for now)
    route_stops = route_stops[["route_id", "stop_id"]].drop_duplicates()
    # Join with stop details
    route_stops = route_stops.merge(stops, on="stop_id")

    trips["trip_id"] = trips["trip_id"].astype("string")
    trips["route_id"] = trips["route_id"].astype("string")
    trips["shape_id"] = trips["shape_id"].astype("string")
    shapes["shape_id"] = shapes["shape_id"].astype("string")
    route_stops["route_id"] = route_stops["route_id"].astype("string")

    return routes, trips, shapes, route_stops

