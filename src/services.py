"""
Service module for TransitPulse Lite application.

This module encapsulates the main business logic and processing pipeline
to improve modularity and maintainability.
"""
import pandas as pd
import numpy as np
import pydeck as pdk
from typing import Tuple, Dict, List, Optional
import streamlit as st

from config import AppConfig, UIConfig
from src.utils import make_color, create_icon_layer, create_text_layer, create_scatterplot_layer, create_path_layer, create_line_layer, create_direction_arrows, calculate_initial_compass_bearing


def process_core_pipeline(df: pd.DataFrame, shape_cache: Dict, circular: bool = False):
    """
    Execute the core processing pipeline for transit data.
    
    Args:
        df: Input dataframe with transit data
        shape_cache: Precomputed shape cache
        circular: Whether to treat routes as circular
        
    Returns:
        Processed dataframe with computed metrics and labels
    """
    from src.features import compute_progress_along_shape, compute_headway_proxy
    from src.anomaly import label_anomalies_isoforest

    # Filter and compute progress along shape
    work = df.copy()
    work = compute_progress_along_shape(work, shape_cache)
    work = compute_headway_proxy(work, circular=circular)

    # Handle missing direction_id
    missing = work["headway_m"].isna()
    if missing.any() and "direction_id" in work.columns:
        tmp_all = work.copy().drop(columns=["direction_id"])
        tmp_all = compute_headway_proxy(tmp_all, circular=circular)
        work.loc[missing, "headway_m"] = tmp_all.loc[missing, "headway_m"]

    # Add timestamp metadata
    work["timestamp_utc"] = pd.to_datetime(work["timestamp"], unit="s", utc=True, errors="coerce")
    work["timestamp_kl"] = work["timestamp_utc"].dt.tz_convert("Asia/Kuala_Lumpur")
    work["timestamp_kl_str"] = work["timestamp_kl"].dt.strftime("%Y-%m-%d %H:%M:%S GMT+8")

    return work


def detect_anomalies(work: pd.DataFrame, route_history: Dict) -> Tuple[pd.DataFrame, Dict]:
    """
    Detect anomalies in the transit data using ML models.
    
    Args:
        work: Input dataframe with transit data
        route_history: History of previous anomaly detection results
        
    Returns:
        Tuple of (processed dataframe, updated route history)
    """
    from src.anomaly import label_anomalies_isoforest
    
    out_frames = []
    for rid, g in work.groupby("route_id"):
        hist = route_history.get(rid)
        g2, new_hist = label_anomalies_isoforest(g, hist)
        route_history[rid] = new_hist
        out_frames.append(g2)

    work2 = pd.concat(out_frames, ignore_index=True) if out_frames else work
    return work2, route_history


def compute_neighbors(work2: pd.DataFrame, circular: bool = False) -> pd.DataFrame:
    """
    Compute neighbor relationships (ahead/behind vehicles) for the given data.
    
    Args:
        work2: Input dataframe with transit data
        circular: Whether to treat routes as circular
        
    Returns:
        Dataframe with neighbor information added
    """
    work2 = work2.sort_values(["route_id", "progress_m"])
    grp_cols = ["route_id", "direction_id"] if "direction_id" in work2.columns else ["route_id"]
    g = work2.groupby(grp_cols)
    
    if circular:
        # Wrap-around logic for IDs
        ahead = g["vehicle_id"].shift(-1)
        first_ids = g["vehicle_id"].transform("first")
        work2["ahead_id"] = ahead.fillna(first_ids)

        behind = g["vehicle_id"].shift(1)
        last_ids = g["vehicle_id"].transform("last")
        work2["behind_id"] = behind.fillna(last_ids)
        
        d_behind = g["headway_m"].shift(1)
        last_headways = g["headway_m"].transform("last")
        work2["dist_behind"] = d_behind.fillna(last_headways)
    else:
        # Standard Linear Logic
        work2["ahead_id"] = g["vehicle_id"].shift(-1).fillna("-")
        work2["behind_id"] = g["vehicle_id"].shift(1).fillna("-")
        work2["dist_behind"] = g["headway_m"].shift(1).fillna(0)
    
    return work2


def prepare_visualization_data(work2: pd.DataFrame, show_only_flagged: bool = False) -> pd.DataFrame:
    """
    Prepare data for visualization including colors, labels, and tooltips.
    
    Args:
        work2: Input dataframe with processed transit data
        show_only_flagged: Whether to show only flagged anomalies
        
    Returns:
        Dataframe prepared for visualization
    """
    if show_only_flagged:
        work2 = work2[work2["anomaly_label"].isin(["BUNCHED", "GAP"])]

    work2["headway_str"] = work2["headway_m"].apply(lambda x: "" if pd.isna(x) else f"{x:.0f}")
    work2["color"] = work2["anomaly_label"].apply(make_color)
    work2["val_emoji"] = "🚌"

    # Precompute tooltip text for vehicles
    work2["tooltip_text"] = work2.apply(
        lambda x: (
            f"{x['route_name']}\n"
            f"Vehicle: {x['vehicle_id']}\n"
            f"Status (to Front): {x['anomaly_label']}\n"
            f"Ahead: {x['ahead_id']} ({x['headway_m']:.0f}m)\n"
            f"Behind: {x['behind_id']} ({x['dist_behind']:.0f}m)\n"
            f"Updated: {x.get('timestamp_kl_str', '')}"
        ), 
        axis=1
    )
    
    return work2


def aggregate_health_metrics(work2: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate health metrics for routes.
    
    Args:
        work2: Input dataframe with processed transit data
        
    Returns:
        Dataframe with aggregated health metrics
    """
    health = (
        work2.groupby(["route_id", "route_name"])
        .agg(
            active=("entity_id", "count"),
            bunched=("anomaly_label", lambda s: (s == "BUNCHED").sum()),
            gap=("anomaly_label", lambda s: (s == "GAP").sum()),
            ML_Flag=("flagged", lambda s: int(np.nansum(pd.Series(s).fillna(False)))),
        )
        .reset_index()
        .sort_values(["bunched", "gap", "active"], ascending=False)
    )
    
    return health


def create_map_layers(
    work2: pd.DataFrame, 
    trips: pd.DataFrame, 
    stops: pd.DataFrame, 
    picked_route: str, 
    show_stops: bool, 
    show_headway: bool, 
    circular: bool,
    shape_cache: Dict
) -> List[pdk.Layer]:
    """
    Create all map layers for visualization.
    
    Args:
        work2: Processed transit data
        trips: Trip information
        stops: Stop information
        picked_route: Currently selected route
        show_stops: Whether to show stops
        show_headway: Whether to show headway lines
        circular: Whether to treat routes as circular
        shape_cache: Shape cache dictionary
        
    Returns:
        List of pydeck layers for the map
    """
    layers = []

    # Detailed Overlays for Selected Route
    if picked_route != "(All Top Routes)" and picked_route != "UNKNOWN":
        target_ids = trips[trips["route_id"].astype(str) == str(picked_route)]
        
        # Paths
        for sid in target_ids["shape_id"].unique():
            sid_str = str(sid)
            if sid_str in shape_cache:
                sc = shape_cache[sid_str]
                # Convert numpy arrays to python lists for JSON serialization
                path_coords = list(zip(sc["lon"].tolist(), sc["lat"].tolist()))
                layers.append(create_path_layer(path_coords))

                # Direction Arrows
                arrow_data = create_direction_arrows(sc, step=20)
                if arrow_data:
                    arrow_layer = create_icon_layer(
                        pd.DataFrame(arrow_data),
                        layer_type="arrow"
                    )
                    layers.append(arrow_layer)
        
        # Stops
        if show_stops:
            r_stops = stops[stops["route_id"].astype(str) == str(picked_route)].copy()
            if not r_stops.empty:
                # Precompute tooltip text for stops
                r_stops["tooltip_text"] = r_stops.apply(
                    lambda x: f"Stop: {x['stop_name']}\nID: {x['stop_id']}", axis=1
                )
                layers.append(create_scatterplot_layer(r_stops))

        # Headway Lines + Arrows
        if show_headway:
            hw_lines = []
            hw_arrows = []
            for _, g in work2.groupby("direction_id" if "direction_id" in work2 else "route_id"):
                if len(g) < 2: continue
                gg = g.sort_values("progress_m")
                coords = gg[["lon", "lat"]].to_numpy() # array of [lon, lat]
                
                # Iterate segments
                for i in range(len(coords)-1):
                    p_start = coords[i] # [lon, lat]
                    p_end = coords[i+1] # [lon, lat]
                    
                    # Line
                    hw_lines.append({
                        "start": p_start.tolist(), 
                        "end": p_end.tolist(), 
                        "color": UIConfig.COLORS["HEADWAY_LINE"]
                    })
                    
                    # Arrow at Midpoint
                    mid_lon = (p_start[0] + p_end[0]) / 2
                    mid_lat = (p_start[1] + p_end[1]) / 2
                    
                    # Bearing from start(lat,lon) to end(lat,lon) - note coords are [lon, lat]
                    angle = calculate_initial_compass_bearing((p_start[1], p_start[0]), (p_end[1], p_end[0]))
                    hw_arrows.append({
                        "lat": float(mid_lat), "lon": float(mid_lon), "angle": -(angle + 180)
                    })

                if circular:
                    # Closing loop: Last -> First
                    p_start = coords[-1]
                    p_end = coords[0]
                    
                    # Safety: Only draw circular link if endpoints are close (< 2km approx)
                    # 0.02 degrees is roughly 2.2km
                    if (abs(p_start[0] - p_end[0]) < 0.02) and (abs(p_start[1] - p_end[1]) < 0.02):
                        hw_lines.append({
                            "start": p_start.tolist(), 
                            "end": p_end.tolist(), 
                            "color": UIConfig.COLORS["CIRCULAR_HEADWAY_LINE"]
                        })
                        
                        mid_lon = (p_start[0] + p_end[0]) / 2
                        mid_lat = (p_start[1] + p_end[1]) / 2
                        angle = calculate_initial_compass_bearing((p_start[1], p_start[0]), (p_end[1], p_end[0]))
                        hw_arrows.append({
                            "lat": float(mid_lat), "lon": float(mid_lon), "angle": -(angle + 180)
                        })

            if hw_lines:
                layers.append(create_line_layer(hw_lines))
            
            if hw_arrows:
                arrow_data_df = pd.DataFrame(hw_arrows)
                # Add icon data to each arrow
                for idx, _ in arrow_data_df.iterrows():
                    arrow_data_df.at[idx, 'icon_data'] = {
                        "url": UIConfig.HEADWAY_ARROW_URL, 
                        "width": 512, 
                        "height": 512, 
                        "anchorY": 256, 
                        "mask": True
                    }
                
                headway_arrow_layer = create_icon_layer(
                    arrow_data_df,
                    layer_type="headway_arrow"
                )
                layers.append(headway_arrow_layer)
    
    # Add the vehicle (icon) layer LAST so it appears on top
    vehicle_layer = create_icon_layer(work2, layer_type="vehicle")
    layers.append(vehicle_layer)

    return layers