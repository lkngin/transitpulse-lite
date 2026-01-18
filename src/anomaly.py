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

    d = df.dropna(subset=["headway_m"]).copy()
    if d.empty:
        return df, history_feat

    # Feature engineering
    d["headway_m"] = d["headway_m"].astype(float)
    d["log_hw"] = np.log(d["headway_m"].clip(lower=1.0))

    # Train set (rolling history helps stability)
    if history_feat is not None and len(history_feat) >= 60:
        train = history_feat.copy()
    else:
        train = d[["log_hw", "headway_m"]].copy()

    X_train = train.to_numpy()
    X_now = d[["log_hw", "headway_m"]].to_numpy()

    model = IsolationForest(
        n_estimators=200,
        contamination=0.08,
        random_state=42
    )
    model.fit(X_train)

    # Higher = more normal
    d["anomaly_score"] = model.decision_function(X_now)

    # Rule-based labels relative to route median headway (very explainable)
    med = float(np.nanmedian(d["headway_m"])) if np.isfinite(np.nanmedian(d["headway_m"])) else 0.0
    d["anomaly_label"] = "NORMAL"
    if med > 0:
        d.loc[d["headway_m"] < 0.45 * med, "anomaly_label"] = "BUNCHED"
        d.loc[d["headway_m"] > 2.50 * med, "anomaly_label"] = "GAP"

    # Flag strongest anomalies (bottom 10% score)
    thresh = np.quantile(d["anomaly_score"], 0.10) if len(d) >= 10 else float(np.min(d["anomaly_score"]))
    d["flagged"] = d["anomaly_score"] <= thresh

    # Prefer entity_id merge (unique). Fallback to vehicle_id if needed.
    key = "entity_id" if "entity_id" in df.columns and "entity_id" in d.columns else "vehicle_id"

    merged = df.merge(
        d[[key, "anomaly_score", "anomaly_label", "flagged"]],
        on=key,
        how="left",
        suffixes=("", "_p")
    )

    # Fill missing preds
    merged["anomaly_label"] = merged["anomaly_label"].fillna("NO_DATA")
    merged["flagged"] = merged["flagged"].fillna(False)

    # Update rolling history
    new_hist = d[["log_hw", "headway_m"]].copy()
    if history_feat is None:
        history_feat = new_hist.tail(800)
    else:
        history_feat = pd.concat([history_feat, new_hist], ignore_index=True).tail(2000)

    return merged, history_feat
