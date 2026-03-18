"""Lightweight PCA via MLX SVD."""

import mlx.core as mx
import numpy as np


class PCA:
    """Principal Component Analysis using MLX.

    Parameters
    ----------
    n_components : int
        Number of components to keep.
    """

    def __init__(self, n_components: int = 2):
        self.n_components = n_components
        self.components_ = None
        self.mean_ = None
        self.embedding_ = None

    def fit_transform(self, X) -> np.ndarray:
        """Fit PCA and return the projection.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data. Accepts numpy or mlx arrays.

        Returns
        -------
        Y : np.ndarray, shape (n_samples, n_components)
        """
        if isinstance(X, np.ndarray):
            X = mx.array(X, dtype=mx.float32)
        else:
            X = X.astype(mx.float32)

        self.mean_ = mx.mean(X, axis=0)
        X_centered = X - self.mean_

        n, d = X_centered.shape
        if n > d:
            # n >> d: SVD on d x d covariance matrix (much faster)
            cov = (X_centered.T @ X_centered) / (n - 1)
            mx.eval(cov)
            _, _, Vt = mx.linalg.svd(cov, stream=mx.cpu)
        else:
            _, _, Vt = mx.linalg.svd(X_centered, stream=mx.cpu)

        self.components_ = Vt[: self.n_components]
        Y = X_centered @ self.components_.T
        mx.eval(Y)

        self.embedding_ = np.array(Y, copy=False)
        return self.embedding_
