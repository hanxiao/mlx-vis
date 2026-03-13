"""Shared KNN computation: brute-force or NNDescent."""

import numpy as np
import mlx.core as mx


def compute_knn(X, k, method="auto", verbose=False, random_state=None, return_euclidean=True):
    """Compute k-nearest neighbors using brute-force or NNDescent.

    Args:
        X: (n, d) numpy array or MLX array.
        k: number of neighbors.
        method: "auto", "brute", or "nndescent".
            - "auto": brute for n <= 20000, nndescent for larger.
            - "brute": exact brute-force on GPU.
            - "nndescent": approximate via NNDescent.
        verbose: print progress.
        random_state: random seed for NNDescent.
        return_euclidean: if True, return Euclidean distances (sqrt).
            If False, return squared distances.

    Returns:
        (indices, distances): both (n, k) numpy arrays.
    """
    if isinstance(X, mx.array):
        X_np = np.array(X)
    else:
        X_np = X if isinstance(X, np.ndarray) else np.array(X)

    n = X_np.shape[0]

    if method == "auto":
        method = "brute" if n <= 20000 else "nndescent"

    if method == "nndescent":
        from mlx_vis._nndescent.nndescent import NNDescent
        nn = NNDescent(
            k=k,
            verbose=verbose,
            random_state=random_state if random_state is not None else 42,
        )
        indices, dists = nn.build(X_np)
        if not return_euclidean:
            dists = dists ** 2
        return indices, dists

    # Brute-force on GPU
    X_mx = mx.array(X_np.astype(np.float32))
    sq_norms = mx.sum(X_mx * X_mx, axis=1)
    mx.eval(sq_norms)

    chunk_size = min(n, max(1000, 500_000_000 // (n * 4)))
    knn_indices = np.zeros((n, k), dtype=np.int32)
    knn_sq_dists = np.zeros((n, k), dtype=np.float32)

    prev_idx = prev_dists = prev_start = prev_end = None

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        D_chunk = mx.maximum(
            sq_norms[start:end, None] + sq_norms[None, :] - 2.0 * (X_mx[start:end] @ X_mx.T),
            0.0,
        )
        if end - start == n:
            D_chunk = D_chunk + mx.eye(n) * 1e30
        else:
            arange_chunk = mx.arange(start, end)[:, None]
            arange_all = mx.arange(n)[None, :]
            D_chunk = D_chunk + (arange_chunk == arange_all).astype(mx.float32) * 1e30

        idx = mx.argsort(D_chunk, axis=1)[:, :k]
        dists = mx.take_along_axis(D_chunk, idx, axis=1)

        if prev_idx is not None:
            mx.eval(prev_idx, prev_dists)
            knn_indices[prev_start:prev_end] = np.array(prev_idx).astype(np.int32)
            knn_sq_dists[prev_start:prev_end] = np.array(prev_dists).astype(np.float32)

        prev_idx, prev_dists, prev_start, prev_end = idx, dists, start, end

    if prev_idx is not None:
        mx.eval(prev_idx, prev_dists)
        knn_indices[prev_start:prev_end] = np.array(prev_idx).astype(np.int32)
        knn_sq_dists[prev_start:prev_end] = np.array(prev_dists).astype(np.float32)

    if return_euclidean:
        return knn_indices, np.sqrt(np.maximum(knn_sq_dists, 0)).astype(np.float32)
    return knn_indices, knn_sq_dists
