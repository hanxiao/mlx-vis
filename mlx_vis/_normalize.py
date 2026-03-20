"""Shared input normalization for all DR methods."""

import numpy as np


def normalize_input(X: np.ndarray, method: str | bool) -> np.ndarray:
    """Normalize input data before dimensionality reduction.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Input data (float32).
    method : str or bool
        - False or None: no normalization (passthrough).
        - True or "standard": z-score standardization per feature.
        - "minmax": min-max scaling to [0, 1] per feature.

    Returns
    -------
    np.ndarray of shape (n_samples, n_features)
    """
    if not method or method is None:
        return X
    if method is True or method == "standard":
        mean = X.mean(axis=0)
        std = X.std(axis=0) + 1e-8
        return (X - mean) / std
    elif method == "minmax":
        xmin = X.min(axis=0)
        xmax = X.max(axis=0)
        return (X - xmin) / (xmax - xmin + 1e-8)
    else:
        raise ValueError(f"Unknown normalize method: {method}")
