"""
Utility functions for TransitPulse Lite application.

This module contains utility functions extracted from the main app file
to improve modularity and maintainability.
"""
import math
import numpy as np
import pandas as pd
import pydeck as pdk
from datetime import datetime
from zoneinfo import ZoneInfo

from config import RegionConfig, UIConfig


def now_kl() -> datetime:
    """Get current time in KL timezone."""
    return datetime.now(RegionConfig.TIMEZONE)


def fmt_kl(dt: datetime) -> str:
    """Format datetime for display in KL timezone."""
    return dt.strftime("%Y-%m-%d %H:%M:%S") + " GMT+8"


def iso_kl(dt: datetime) -> str:
    """Convert datetime to ISO format in KL timezone."""
    return dt.isoformat(timespec="seconds")


def calculate_initial_compass_bearing(point1, point2):
    """
    Calculates the bearing between two points.
    The formulae used is the following:
        θ = atan2(sin(Δlong).cos(lat2),
                  cos(lat1).sin(lat2) − sin(lat1).cos(lat2).cos(Δlong))
    :Parameters:
      - point1: The tuple representing the (lat, lon) of the first point.
      - point2: The tuple representing the (lat, lon) of the second point.
    :Returns:
      The bearing in degrees
    """
    if (type(point1) != tuple) or (type(point2) != tuple):
        raise TypeError("Only tuples are supported as arguments")

    lat1 = math.radians(point1[0])
    lat2 = math.radians(point2[0])
    diffLong = math.radians(point2[1] - point1[1])

    x = math.sin(diffLong) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1) * math.cos(lat2) * math.cos(diffLong))

    initial_bearing = math.atan2(x, y)
    initial_bearing = math.degrees(initial_bearing)
    compass_bearing = (initial_bearing + 360) % 360
    return compass_bearing


def make_color(label: str):
    """Map anomaly label to color values."""
    return UIConfig.COLORS.get(label, UIConfig.COLORS["NORMAL"])


def create_icon_layer(data, layer_type="vehicle"):
    """Create a standardized icon layer for vehicles or arrows."""
    if layer_type == "vehicle":
        icon_url = UIConfig.VEHICLE_ICON_URL
        size = UIConfig.VEHICLE_ICON_SIZE
    elif layer_type == "arrow":
        icon_url = UIConfig.ARROW_ICON_URL
        size = 15
    elif layer_type == "headway_arrow":
        icon_url = UIConfig.HEADWAY_ARROW_URL
        size = 50
    else:
        raise ValueError(f"Unknown layer_type: {layer_type}")

    # Prepare icon data for all entries
    icon_data = [{
        "url": icon_url,
        "width": 128 if layer_type == "vehicle" else (128 if layer_type == "arrow" else 512),
        "height": 128 if layer_type == "vehicle" else (128 if layer_type == "arrow" else 512),
        "anchorY": 128 if layer_type == "vehicle" else (128 if layer_type == "arrow" else 256),
        "mask": True  # Enable coloring
    } for _ in range(len(data))]

    data["icon_data"] = icon_data

    return pdk.Layer(
        "IconLayer",
        data,
        get_position=["lon", "lat"],
        get_icon="icon_data",
        get_size=size,
        get_color="color" if "color" in data.columns else [0, 128, 255, 170],
        get_angle="angle" if "angle" in data.columns else 0,
        pickable=layer_type == "vehicle",
        billboard=layer_type != "headway_arrow"  # Disable billboard for headway arrows
    )


def create_text_layer(data):
    """Create a standardized text layer for displaying vehicle IDs."""
    return pdk.Layer(
        "TextLayer",
        data,
        get_position=["lon", "lat"],
        get_text="vehicle_id",
        get_color=UIConfig.COLORS["TEXT_COLOR"],  # Black text
        get_size=UIConfig.TEXT_LABEL_SIZE,
        get_alignment_baseline="'bottom'",
        get_pixel_offset=[0, -15],  # Shift up slightly
        background=True,
        get_background_color=UIConfig.COLORS["BACKGROUND_COLOR"],  # White background block
        pickable=False
    )


def create_scatterplot_layer(data, get_color=None):
    """Create a standardized scatterplot layer for stops."""
    if get_color is None:
        get_color = UIConfig.COLORS["STOP_FILL"]
    
    return pdk.Layer(
        "ScatterplotLayer", 
        data,
        get_position=["stop_lon", "stop_lat"],
        get_fill_color=get_color, 
        get_radius=30, 
        pickable=True,
    )


def create_path_layer(path_coords):
    """Create a standardized path layer for route shapes."""
    return pdk.Layer(
        "PathLayer",
        [{"path": path_coords}],
        get_path="path", 
        get_color=UIConfig.COLORS["PATH"], 
        width_min_pixels=UIConfig.PATH_WIDTH_MIN_PIXELS,
    )


def create_line_layer(lines_data):
    """Create a standardized line layer for headway lines."""
    return pdk.Layer(
        "LineLayer", 
        lines_data,
        get_source_position="start", 
        get_target_position="end",
        get_color="color", 
        get_width=UIConfig.LINE_LAYER_WIDTH
    )


def create_direction_arrows(shape_cache_data, step=20):
    """Create direction arrows for route visualization."""
    arrow_data = []
    # Sample every 20th point to avoid clutter
    lats = shape_cache_data["lat"]
    lons = shape_cache_data["lon"]
    
    for i in range(0, len(lats) - step, step):
        p1 = (lats[i], lons[i])
        p2 = (lats[i + 5], lons[i + 5])  # Look ahead 5 points for smoother bearing
        angle = calculate_initial_compass_bearing(p1, p2)
        arrow_data.append({
            "lat": float(lats[i]),
            "lon": float(lons[i]),
            "angle": -angle
        })

    if arrow_data:
        for ad in arrow_data:
            ad["icon_data"] = {
                "url": UIConfig.ARROW_ICON_URL, 
                "width": 128, 
                "height": 128, 
                "anchorY": 128, 
                "mask": True
            }

    return arrow_data