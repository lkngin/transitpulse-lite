import pandas as pd
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.gtfs_static import load_gtfs_static
from src.gtfs_rt import fetch_vehicle_positions
from src.features import attach_routes, compute_progress_along_shape, build_shape_cache, compute_headway_proxy

GTFS_STATIC_URL = "https://api.data.gov.my/gtfs-static/prasarana/?category=rapid-bus-kl"
GTFS_RT_URL = "https://api.data.gov.my/gtfs-realtime/vehicle-position/prasarana/?category=rapid-bus-kl"

def diagnose_data_flow():
    print("--- Loading Static Data ---")
    routes, trips, shapes = load_gtfs_static(GTFS_STATIC_URL)
    print(f"Loaded {len(routes)} routes, {len(trips)} trips, {len(shapes)} shape points.")

    print("\n--- Fetching Realtime Data ---")
    rt = fetch_vehicle_positions(GTFS_RT_URL)
    print(f"Fetched {len(rt)} vehicle positions.")

    print("\n--- Attaching Routes ---")
    df = attach_routes(rt, trips, routes)
    unknown_routes = (df["route_id"] == "UNKNOWN").sum()
    print(f"Buses with unknown route_id: {unknown_routes}/{len(df)}")
    
    missing_shape = df["shape_id"].isna().sum()
    print(f"Buses missing shape_id: {missing_shape}/{len(df)}")

    print("\n--- Computing Progress ---")
    shape_cache = build_shape_cache(shapes)
    df = compute_progress_along_shape(df, shape_cache)
    missing_progress = df["progress_m"].isna().sum()
    print(f"Buses missing progress_m: {missing_progress}/{len(df)}")

    print("\n--- Computing Headway ---")
    df = compute_headway_proxy(df, circular=False)
    missing_headway = df["headway_m"].isna().sum()
    print(f"Buses missing headway_m: {missing_headway}/{len(df)}")

    if not df.empty:
        print("\n--- Sample Results ---")
        print(df[["vehicle_id", "route_id", "shape_id", "progress_m", "headway_m"]].dropna(subset=["progress_m"]).head(10))

if __name__ == "__main__":
    diagnose_data_flow()
