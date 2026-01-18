import numpy as np
import pandas as pd

EARTH_R = 6371000.0

def haversine_m(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * EARTH_R * np.arcsin(np.sqrt(a))

def attach_routes(rt_df: pd.DataFrame, trips: pd.DataFrame, routes: pd.DataFrame) -> pd.DataFrame:
    """
    Join realtime df -> trips -> routes to add route_name AND shape_id/direction_id if present.
    """
    df = rt_df.merge(trips, on="trip_id", how="left")
    
    if "shape_id" in df.columns:
        df["shape_id"] = df["shape_id"].astype("string")

    df = df.merge(routes, on="route_id", how="left")

    df["route_id"] = df["route_id"].fillna("UNKNOWN")
    if "direction_id" in df.columns:
        df["direction_id"] = df["direction_id"].fillna(-1)

    df["route_short_name"] = df.get("route_short_name", pd.Series([""] * len(df))).fillna("")
    df["route_long_name"]  = df.get("route_long_name", pd.Series([""] * len(df))).fillna("")
    df["route_name"] = (df["route_short_name"] + " " + df["route_long_name"]).str.strip()
    df.loc[df["route_name"] == "", "route_name"] = "UNKNOWN ROUTE"

    # vehicle_id is for display; entity_id is better as a unique key
    df["vehicle_id"] = df["vehicle_id"].fillna("unknown_vehicle")
    return df

def build_shape_cache(shapes: pd.DataFrame, downsample_step: int = 8) -> dict:
    """
    Precompute per-shape polyline and cumulative distance.
    Returns dict: shape_id -> {lat, lon, cum, ds_idx}
    """
    cache = {}

    for sid, g in shapes.groupby("shape_id"):
        g = g.sort_values("shape_pt_sequence")
        lat = g["shape_pt_lat"].to_numpy(dtype=float)
        lon = g["shape_pt_lon"].to_numpy(dtype=float)

        if len(lat) < 2:
            continue

        seg = haversine_m(lat[:-1], lon[:-1], lat[1:], lon[1:])
        cum = np.concatenate([[0.0], np.cumsum(seg)])

        ds_idx = np.arange(0, len(lat), downsample_step, dtype=int)
        if ds_idx[-1] != len(lat) - 1:
            ds_idx = np.append(ds_idx, len(lat) - 1)

        cache[sid] = {"lat": lat, "lon": lon, "cum": cum, "ds_idx": ds_idx}

    return cache

def compute_progress_along_shape(df: pd.DataFrame, shape_cache: dict) -> pd.DataFrame:
    """
    Snap each vehicle to nearest DOWN-SAMPLED shape point, output progress_m = cum[idx].
    """
    out = df.copy()
    out["progress_m"] = np.nan

    if "shape_id" not in out.columns:
        return out

    for sid, g in out.dropna(subset=["shape_id"]).groupby("shape_id"):
        if sid not in shape_cache:
            continue
        sc = shape_cache[sid]
        lat_s = sc["lat"]; lon_s = sc["lon"]; cum = sc["cum"]; ds_idx = sc["ds_idx"]

        lat_ds = lat_s[ds_idx]
        lon_ds = lon_s[ds_idx]

        vlat = g["lat"].to_numpy(dtype=float)
        vlon = g["lon"].to_numpy(dtype=float)

        prog = []
        for i in range(len(g)):
            d = haversine_m(vlat[i], vlon[i], lat_ds, lon_ds)
            j = int(np.argmin(d))
            idx = int(ds_idx[j])
            prog.append(float(cum[idx]))

        out.loc[g.index, "progress_m"] = prog

    return out

def compute_headway_proxy(df: pd.DataFrame, circular: bool = False) -> pd.DataFrame:
    """
    Using progress_m within each (route_id, direction_id), compute headway proxy:
    headway_m = min(gap_ahead, gap_behind) along the shape.
    """
    out = df.copy()
    out["headway_m"] = np.nan

    group_cols = ["route_id"]
    if "direction_id" in out.columns:
        group_cols.append("direction_id")

    for _, g in out.dropna(subset=["progress_m"]).groupby(group_cols):
        if len(g) < 2:
            continue

        gg = g.sort_values("progress_m")
        prog = gg["progress_m"].to_numpy(dtype=float)

        gaps = np.diff(prog)
        gap_ahead = np.concatenate([gaps, [np.nan]])
        gap_behind = np.concatenate([[np.nan], gaps])

        if circular:
            route_len = float(np.nanmax(prog) - np.nanmin(prog))
            wrap = (route_len - prog[-1]) + prog[0] if route_len > 0 else np.nan
            gap_ahead[-1] = wrap
            gap_behind[0] = wrap

        headway = np.nanmin(np.vstack([gap_ahead, gap_behind]), axis=0)
        out.loc[gg.index, "headway_m"] = headway

    return out
