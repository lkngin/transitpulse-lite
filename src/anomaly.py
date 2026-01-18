import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

def label_anomalies_isoforest(route_df: pd.DataFrame, history_feat: pd.DataFrame | None):
    """
    V2: Uses headway_m (spacing along GTFS route shape progress) rather than geographic NN distance.

    Inputs:
      route_df: must include columns: entity_id (recommended), headway_m
      history_feat: dataframe of prior features (rolling) for stable isolation forest training

    Outputs:
      merged_df: route_df with anomaly_label/anomaly_score/flagged
      updated_history_feat
    """
    df = route_df.copy()

    # Remove potentially stale columns to avoid merge/suffix issues
    for col in ["anomaly_label", "anomaly_score", "flagged"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Defaults
    df["anomaly_label"] = "NO_DATA"
    df["anomaly_score"] = np.nan
    df["flagged"] = False

    # Headway is required for V2
    if "headway_m" not in df.columns:
        return df, history_feat

    # 1. Select candidates for labeling (must have a calculated headway)
    d = df.dropna(subset=["headway_m"]).copy()
    if d.empty:
        # If no headways are found, they all stay as NO_DATA (default)
        return df, history_feat

    # 2. Basic Rule-Based Labeling (Guarantees every bus with headway gets a label)
    d["headway_m"] = d["headway_m"].astype(float)
    med = float(np.nanmedian(d["headway_m"])) if np.isfinite(np.nanmedian(d["headway_m"])) else 0.0
    
    d["anomaly_label"] = "NORMAL"
    if med > 0:
        d.loc[d["headway_m"] < 0.45 * med, "anomaly_label"] = "BUNCHED"
        d.loc[d["headway_m"] > 2.50 * med, "anomaly_label"] = "GAP"

    # 3. ML Anomaly Score (Isolation Forest)
    try:
        d["log_hw"] = np.log(d["headway_m"].clip(lower=1.0))
        
        # Stability: Use history if we have it, else current batch
        if history_feat is not None and len(history_feat) >= 20:
            train = history_feat.copy()
        else:
            train = d[["log_hw", "headway_m"]].copy()

        model = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
        model.fit(train[["log_hw", "headway_m"]])
        d["anomaly_score"] = model.decision_function(d[["log_hw", "headway_m"]])
        
        # Flag bottom 10%
        thresh = np.quantile(d["anomaly_score"], 0.1) if len(d) >= 5 else -1.0
        d["flagged"] = d["anomaly_score"] <= thresh
    except Exception:
        # Fallback if ML fails to initialize or run
        d["anomaly_score"] = 0.0
        d["flagged"] = False

    # 4. Final Merge (Force string keys for reliability)
    key = "entity_id" if "entity_id" in df.columns and "entity_id" in d.columns else "vehicle_id"
    df[key] = df[key].astype(str)
    d[key] = d[key].astype(str)

    merged = df.merge(
        d[[key, "anomaly_score", "anomaly_label", "flagged"]],
        on=key,
        how="left",
        suffixes=("", "_p")
    )

    # Clean up merge artifacts
    if "anomaly_label_p" in merged.columns:
        merged["anomaly_label"] = merged["anomaly_label_p"].fillna("NO_DATA")
        merged["anomaly_score"] = merged["anomaly_score_p"].fillna(np.nan)
        merged["flagged"] = merged["flagged_p"].fillna(False)
        merged = merged.drop(columns=["anomaly_label_p", "anomaly_score_p", "flagged_p"])

    # Update rolling history (limit to last 2000 points)
    new_feat = d[["log_hw", "headway_m"]].copy()
    if history_feat is None:
        history_feat = new_feat.tail(500)
    else:
        history_feat = pd.concat([history_feat, new_feat], ignore_index=True).tail(2000)

    return merged, history_feat
