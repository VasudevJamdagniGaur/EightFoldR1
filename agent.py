"""
Innov8 4.0 — Talent Fraud Detection Challenge
Simple agent: query a subset of rows via oracle, train RF, predict on full dataset.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def run_agent(df: pd.DataFrame, oracle_fn, budget: int) -> np.ndarray:
    """
    Parameters
    ----------
    df        : pd.DataFrame, shape (10000, 25)
                The full feature matrix. Row indices 0–9999.
                No label column is present.

    oracle_fn : callable
                oracle_fn(indices: list[int]) -> list[int]
                    indices  — list of row indices to query (0-based)
                    returns  — list of int labels, same length, values in {0, 1}
                               0 = legitimate candidate, 1 = fraudulent

    budget    : int — maximum oracle queries available (100 in final evaluation)

    Returns
    -------
    predictions : np.ndarray, dtype int, shape (10000,)
                  Binary predictions for every row in df. Values in {0, 1}.
    """
    n = len(df)

    # Use full budget to get labels for a simple subset of indices (no active learning yet)
    indices = list(range(min(budget, n)))
    labels = oracle_fn(indices)

    # Training data: df.iloc[indices]
    X_train = df.iloc[indices]
    y_train = np.array(labels, dtype=np.intp)

    # Train RandomForestClassifier with class_weight="balanced"
    clf = RandomForestClassifier(class_weight="balanced", random_state=42)
    clf.fit(X_train, y_train)

    # Predict probabilities on the full dataset (fraud = positive class, index 1)
    proba = clf.predict_proba(df)
    fraud_proba = proba[:, 1]

    # Binary predictions: >= 0.5 → 1, else 0
    predictions = (fraud_proba >= 0.5).astype(np.intp)

    return predictions
