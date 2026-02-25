"""
Select which rows are most valuable to verify (oracle queries) to maximize
prediction accuracy on the rest of the dataset.

Strategy (no labels used):
1. Anomaly score: frauds are often anomalies → Isolation Forest scores.
2. Diversity: avoid querying similar rows → greedy farthest-point (core-set) in
   standardized feature space so the selected set covers the distribution.
3. Combined value: at each step pick the row that maximizes
   value = w_anomaly * anomaly_score + w_diversity * min_distance_to_selected.

Uses only: numpy, pandas, sklearn, scipy.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
DATASET_PATH = "dataset.csv"
BUDGET = 100
# Weight for anomaly (higher = prefer likely-fraud candidates). Diversity weight = 1 - W_ANOMALY.
W_ANOMALY = 0.6
RANDOM_STATE = 42
OUTPUT_INDICES_PATH = "valuable_row_indices.txt"

# -----------------------------------------------------------------------------
# Load and prepare data
# -----------------------------------------------------------------------------
def load_and_scale(path: str):
    df = pd.read_csv(path)
    X = df.values.astype(np.float64)
    # Handle any NaN (e.g. fill with column median)
    nan_mask = np.isnan(X)
    if nan_mask.any():
        col_med = np.nanmedian(X, axis=0)
        X = np.where(nan_mask, np.broadcast_to(col_med, X.shape), X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return df, X_scaled, scaler


# -----------------------------------------------------------------------------
# Anomaly scores (Isolation Forest: more negative = more anomalous)
# -----------------------------------------------------------------------------
def compute_anomaly_scores(X: np.ndarray, random_state: int) -> np.ndarray:
    clf = IsolationForest(random_state=random_state, contamination="auto")
    # Raw score: negative = anomaly. We want "value" to be higher for more anomalous.
    raw = clf.fit_predict(X)  # -1 or 1
    scores = clf.decision_function(X)  # lower = more anomalous
    # Normalize to [0, 1] with 1 = most anomalous (most negative decision function)
    min_s, max_s = scores.min(), scores.max()
    if max_s <= min_s:
        return np.ones(len(scores)) * 0.5
    anomaly_value = (max_s - scores) / (max_s - min_s)
    return anomaly_value


# -----------------------------------------------------------------------------
# Greedy selection: maximize w_anomaly * anomaly + w_diversity * min_dist_to_selected
# -----------------------------------------------------------------------------
def select_valuable_indices(
    X: np.ndarray,
    anomaly_value: np.ndarray,
    budget: int,
    w_anomaly: float,
    random_state: int,
) -> np.ndarray:
    n = X.shape[0]
    rng = np.random.default_rng(random_state)
    w_diversity = 1.0 - w_anomaly

    # Normalize anomaly to [0,1] if not already (already is from compute_anomaly_scores)
    a = np.asarray(anomaly_value, dtype=np.float64)
    a_min, a_max = a.min(), a.max()
    if a_max > a_min:
        a = (a - a_min) / (a_max - a_min)

    selected = []
    # Start with the most anomalous point to seed the set
    first = int(np.argmax(a))
    selected.append(first)
    remaining = np.array([i for i in range(n) if i != first], dtype=np.intp)

    while len(selected) < budget and len(remaining) > 0:
        S = np.array(selected, dtype=np.intp)
        X_sel = X[S]
        X_rem = X[remaining]

        # Min distance from each remaining point to the selected set
        d = cdist(X_rem, X_sel, metric="euclidean")
        min_dist = np.min(d, axis=1)

        # Normalize min_dist to [0,1] (1 = farthest)
        d_min, d_max = min_dist.min(), min_dist.max()
        if d_max > d_min:
            diversity = (min_dist - d_min) / (d_max - d_min)
        else:
            diversity = np.ones_like(min_dist)

        a_rem = a[remaining]
        value = w_anomaly * a_rem + w_diversity * diversity

        # Pick argmax value; tie-break by index for reproducibility
        best_local = np.argmax(value)
        best_idx = remaining[best_local]
        selected.append(int(best_idx))
        remaining = np.delete(remaining, best_local)

    return np.array(selected, dtype=np.intp)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    print("Loading dataset...")
    df, X_scaled, _ = load_and_scale(DATASET_PATH)
    n, p = X_scaled.shape
    print(f"  Shape: {n} x {p}")

    print("Computing anomaly scores (Isolation Forest)...")
    anomaly_value = compute_anomaly_scores(X_scaled, RANDOM_STATE)

    print(f"Selecting {BUDGET} valuable rows (w_anomaly={W_ANOMALY})...")
    indices = select_valuable_indices(
        X_scaled,
        anomaly_value,
        budget=min(BUDGET, n),
        w_anomaly=W_ANOMALY,
        random_state=RANDOM_STATE,
    )

    print(f"  Selected indices (first 20): {indices[:20].tolist()}")
    np.savetxt(OUTPUT_INDICES_PATH, indices, fmt="%d")
    print(f"  Saved all {len(indices)} indices to {OUTPUT_INDICES_PATH}")

    # Optional: summary stats of anomaly scores for selected vs rest
    sel_anomaly = anomaly_value[indices]
    rest_mask = np.ones(n, dtype=bool)
    rest_mask[indices] = False
    rest_anomaly = anomaly_value[rest_mask]
    print(f"  Mean anomaly (selected): {sel_anomaly.mean():.4f}")
    print(f"  Mean anomaly (rest):    {rest_anomaly.mean():.4f}")
    print("Done.")


if __name__ == "__main__":
    main()
