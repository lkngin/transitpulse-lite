import io
import zipfile
import requests
import pandas as pd

def load_gtfs_static(gtfs_static_url: str):
    """
    Download GTFS static ZIP and load routes.txt and trips.txt.
    """
    r = requests.get(gtfs_static_url, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()

    z = zipfile.ZipFile(io.BytesIO(r.content))
    routes = pd.read_csv(z.open("routes.txt"))
    trips  = pd.read_csv(z.open("trips.txt"))

    # Keep only what we need
    routes = routes[["route_id", "route_short_name", "route_long_name"]].copy()
    trips  = trips[["trip_id", "route_id"]].copy()

    return routes, trips
