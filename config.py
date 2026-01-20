"""
Configuration file for TransitPulse Lite application.

This module centralizes all configuration values to improve maintainability
and allow easy customization without changing code.
"""

from zoneinfo import ZoneInfo


# URLs and API Configuration
class APIConfig:
    GTFS_STATIC_URL = "https://api.data.gov.my/gtfs-static/prasarana/?category=rapid-bus-kl"
    GTFS_RT_URL = "https://api.data.gov.my/gtfs-realtime/vehicle-position/prasarana/?category=rapid-bus-kl"
    # GTFS_RT_URL_MRT_FEEDER = "https://api.data.gov.my/gtfs-realtime/vehicle-position/prasarana/?category=rapid-bus-mrtfeeder"

    # Timeout settings
    STATIC_API_TIMEOUT = 60
    REALTIME_API_TIMEOUT = 30
    LLM_API_TIMEOUT = 30

    # Retry settings
    MAX_RETRIES = 3
    RETRY_DELAY = 2  # seconds


# Geographic and Regional Configuration
class RegionConfig:
    AREA_NAME = "Kuala Lumpur / Selangor (Rapid Bus KL)"
    TIMEZONE = ZoneInfo("Asia/Kuala_Lumpur")
    TZ_LABEL = "Kuala Lumpur (GMT+8)"
    
    # Default map view coordinates
    DEFAULT_LATITUDE = 3.14
    DEFAULT_LONGITUDE = 101.69
    DEFAULT_ZOOM_LEVEL = 11
    
    # Distance thresholds
    MAX_CIRCULAR_DISTANCE_KM = 2.2  # Maximum distance in km to consider endpoints for circular route


# Application Behavior Configuration
class AppConfig:
    # Auto-refresh settings
    AUTO_REFRESH_INTERVAL_MS = 30_000  # 30 seconds
    
    # Cache settings
    SHAPE_CACHE_DOWNSAMPLE_STEP = 8
    
    # Thresholds for anomaly detection
    BUNCHING_THRESHOLD_RATIO = 0.45  # Headway < 45% of median = BUNCHED
    GAP_THRESHOLD_RATIO = 2.50       # Headway > 250% of median = GAP
    
    # Display settings
    TOP_ROUTES_COUNT = 15
    HEALTH_HISTORY_LIMIT = 40  # Number of historical records to keep
    
    # ML settings
    ISOLATION_FOREST_ESTIMATORS = 100
    ISOLATION_FOREST_CONTAMINATION = 0.1
    ISOLATION_FOREST_RANDOM_STATE = 42
    MIN_HISTORY_POINTS_FOR_TRAINING = 20
    HISTORY_FEATURES_LIMIT = 2000  # Max records in feature history
    ANOMALY_PERCENTILE_THRESHOLD = 0.1  # Bottom 10% marked as anomalous


# UI/UX Configuration
class UIConfig:
    # Map layer settings
    VEHICLE_ICON_SIZE = 30
    TEXT_LABEL_SIZE = 15
    PATH_WIDTH_MIN_PIXELS = 4
    LINE_LAYER_WIDTH = 3
    
    # Color schemes
    COLORS = {
        "BUNCHED": [255, 0, 0, 190],      # Red
        "GAP": [255, 140, 0, 200],        # Orange
        "NO_DATA": [120, 120, 120, 160],  # Gray
        "NORMAL": [0, 128, 255, 170],     # Blue
        "PATH": [0, 100, 255, 120],       # Light blue for route paths
        "HEADWAY_LINE": [255, 0, 255, 120],  # Magenta for headway lines
        "CIRCULAR_HEADWAY_LINE": [255, 0, 255, 60],  # Lighter magenta for circular connections
        "ARROW_BLUE": [0, 100, 255, 200], # Blue for direction arrows
        "ARROW_PINK": [255, 0, 255, 200], # Pink for headway arrows
        "STOP_FILL": [0, 0, 0, 200],      # Black for stops
        "TEXT_COLOR": [0, 0, 0, 255],     # Black for text
        "BACKGROUND_COLOR": [255, 255, 255, 200]  # White for backgrounds
    }
    
    # Icon URLs
    VEHICLE_ICON_URL = "https://img.icons8.com/ios-filled/50/000000/bus.png"
    ARROW_ICON_URL = "https://img.icons8.com/ios-filled/50/000000/sort-up.png"
    HEADWAY_ARROW_URL = "https://cdn-icons-png.flaticon.com/512/60/60995.png"
    
    # Chart colors
    CHART_COLORS = {
        "BUNCHED": "#FF0000",
        "GAP": "#FF8C00", 
        "ACTIVE": "#1E90FF"
    }


# Feature computation settings
class FeatureConfig:
    DOWNSAMPLE_STEP = 8
    BEARING_SAMPLE_STEP = 20
    BEARING_LOOKAHEAD_POINTS = 5