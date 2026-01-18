import io
import zipfile
import requests
import pandas as pd

def load_gtfs_static(gtfs_static_url: str):
    r = requests.get(gtfs_static_url, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()

    z = zipfile.ZipFile(io.BytesIO(r.content))

    routes = pd.read_csv(z.open("routes.txt"))
    trips  = pd.read_csv(z.open("trips.txt"))
    shapes = pd.read_csv(z.open("shapes.txt"))

    routes = routes[["route_id", "route_short_name", "route_long_name"]].copy()

    keep_trip_cols = ["trip_id", "route_id", "shape_id"]
    if "direction_id" in trips.columns:
        keep_trip_cols.append("direction_id")
    trips = trips[keep_trip_cols].copy()

    shapes = shapes[["shape_id", "shape_pt_lat", "shape_pt_lon", "shape_pt_sequence"]].copy()

    trips["trip_id"] = trips["trip_id"].astype("string")
    trips["shape_id"] = trips["shape_id"].astype("string")
    shapes["shape_id"] = shapes["shape_id"].astype("string")

    return routes, trips, shapes

