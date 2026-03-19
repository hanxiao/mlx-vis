"""PaCMAP and LocalMAP in pure MLX."""

import math
import time

import mlx.core as mx
import numpy as np


def _brute_knn(X_mx, k, chunk_size=None):
    """Brute-force KNN using chunked pairwise distances on GPU."""
    n = X_mx.shape[0]
    # Auto chunk size: fit chunk_size * n floats in ~500MB
    if chunk_size is None:
        chunk_size = min(n, max(1000, 500_000_000 // (n * 4)))

    # Precompute squared norms once
    sq_norms = mx.sum(X_mx * X_mx, axis=1)  # (n,)
    mx.eval(sq_norms)

    all_indices = []
    all_distances = []
    prev = None

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        sq_chunk = sq_norms[start:end, None]  # reuse, don't recompute

        # dist^2 = ||a||^2 + ||b||^2 - 2*a.b
        dists = mx.maximum(
            sq_chunk + sq_norms[None, :] - 2.0 * (X_mx[start:end] @ X_mx.T),
            0.0,
        )

        # Set self-distance to inf (simple offset approach)
        if end - start == n:
            dists = dists + mx.eye(n) * 1e30
        else:
            arange_chunk = mx.arange(start, end)[:, None]
            arange_all = mx.arange(n)[None, :]
            dists = dists + (arange_chunk == arange_all).astype(mx.float32) * 1e30

        # Full sort, take top k (argsort and argpartition same speed on MLX)
        indices = mx.argsort(dists, axis=1)[:, :k]
        gathered = mx.sqrt(mx.take_along_axis(dists, indices, axis=1))

        # Pipeline: collect previous while current computes
        if prev is not None:
            p_idx, p_dist, _, _ = prev
            mx.eval(p_idx, p_dist)
            all_indices.append(p_idx)
            all_distances.append(p_dist)

        prev = (indices, gathered, start, end)

    if prev is not None:
        p_idx, p_dist, _, _ = prev
        mx.eval(p_idx, p_dist)
        all_indices.append(p_idx)
        all_distances.append(p_dist)

    return mx.concatenate(all_indices, axis=0), mx.concatenate(all_distances, axis=0)


def _sample_neighbors(knn_distances_np, knn_indices_np, n_neighbors):
    """Sample neighbour pairs using scaled distance (vectorised numpy)."""
    n, _ = knn_distances_np.shape
    # sigma = mean of distances to 4th-6th neighbours
    sig = np.maximum(np.mean(knn_distances_np[:, 3:6], axis=1), 1e-10)

    # Scale distances: d^2 / (sig[i] * sig[neighbour])
    sig_nb = sig[knn_indices_np]  # (n, k_extra)
    scaled = knn_distances_np**2 / (sig[:, None] * sig_nb)  # (n, k_extra)

    # Pick top n_neighbors by scaled distance
    sorted_idx = np.argsort(scaled, axis=1)[:, :n_neighbors]  # (n, n_neighbors)
    picked = np.take_along_axis(knn_indices_np, sorted_idx, axis=1)  # (n, n_neighbors)

    src = np.repeat(np.arange(n, dtype=np.int32), n_neighbors)
    dst = picked.ravel().astype(np.int32)
    return np.stack([src, dst], axis=1)


def _sample_MN_pairs(X_mx, n_MN, rng):
    """Sample mid-near pairs on GPU. For each point, sample 6 random, pick 2nd closest."""
    n = X_mx.shape[0]
    if n_MN == 0:
        return np.empty((0, 2), dtype=np.int32)

    results = []

    for j in range(n_MN):
        # Sample random candidates (avoid self via offset)
        offsets = mx.array(rng.integers(1, n, size=(n, 6)).astype(np.int32))
        idx = mx.arange(n)[:, None]
        candidates = (idx + offsets) % n  # (n, 6)

        # Gather candidate vectors: (n, 6, dim)
        cand_flat = candidates.reshape(-1)  # (n*6,)
        X_cand = X_mx[cand_flat].reshape(n, 6, -1)  # (n, 6, dim)

        # Squared distances
        diffs = X_cand - X_mx[:, None, :]  # (n, 6, dim)
        dists = mx.sum(diffs * diffs, axis=2)  # (n, 6)

        # Pick 2nd closest (argsort, take index 1)
        sorted_idx = mx.argsort(dists, axis=1)  # (n, 6)
        picked = mx.take_along_axis(candidates, sorted_idx[:, 1:2], axis=1).squeeze(1)  # (n,)

        results.append(picked)

    mx.eval(*results)

    # Build pair array in numpy
    pair_MN = np.empty((n * n_MN, 2), dtype=np.int32)
    src = np.arange(n, dtype=np.int32)
    for j, picked in enumerate(results):
        start = j * n
        pair_MN[start:start + n, 0] = src
        pair_MN[start:start + n, 1] = np.array(picked, dtype=np.int32)

    return pair_MN


def _sample_FP_pairs(n, pair_neighbors_np, n_neighbors, n_FP, rng):
    """Sample further pairs, excluding self and nearest neighbours."""
    if n_FP == 0:
        return np.empty((0, 2), dtype=np.int32)

    pair_FP = np.empty((n * n_FP, 2), dtype=np.int32)
    neighbour_rows = pair_neighbors_np[:, 1].reshape(n, n_neighbors) if n_neighbors else None
    sample_factor = max(16, 4 * n_FP)
    max_rounds = 24

    for i in range(n):
        row_start = i * n_FP
        row_end = row_start + n_FP
        chosen = []
        chosen_set = set()

        reject = {i}
        if neighbour_rows is not None:
            reject.update(int(x) for x in neighbour_rows[i])

        rounds = 0
        while len(chosen) < n_FP and rounds < max_rounds:
            remaining = n_FP - len(chosen)
            batch_size = max(sample_factor, remaining * sample_factor)
            candidates = rng.integers(0, n, size=batch_size, dtype=np.int32)
            for candidate in candidates.tolist():
                if candidate in reject or candidate in chosen_set:
                    continue
                chosen.append(candidate)
                chosen_set.add(candidate)
                if len(chosen) == n_FP:
                    break
            rounds += 1

        if len(chosen) < n_FP:
            # Extremely dense neighbourhoods can exhaust valid random draws.
            # Fall back to the explicit remaining candidate pool to keep invariants.
            unavailable = np.fromiter(reject.union(chosen_set), dtype=np.int32)
            pool = np.setdiff1d(np.arange(n, dtype=np.int32), unavailable, assume_unique=False)
            need = n_FP - len(chosen)
            if pool.size:
                extra = rng.choice(pool, size=need, replace=pool.size < need).astype(np.int32)
                chosen.extend(extra.tolist())
            else:
                # Degenerate fallback: at least keep self out of FP pairs.
                fallback_pool = np.setdiff1d(np.arange(n, dtype=np.int32), np.array([i], dtype=np.int32), assume_unique=False)
                extra = rng.choice(fallback_pool, size=need, replace=fallback_pool.size < need).astype(np.int32)
                chosen.extend(extra.tolist())

        pair_FP[row_start:row_end, 0] = i
        pair_FP[row_start:row_end, 1] = np.asarray(chosen[:n_FP], dtype=np.int32)

    return pair_FP


def _resample_local_fp_pairs(pair_neighbors_np, pair_FP_np, Y_mx, low_dist_thres, rng):
    """Resample further pairs using embedding locality (pure MLX GPU)."""
    if pair_FP_np.size == 0 or low_dist_thres <= 0:
        return pair_FP_np

    n = Y_mx.shape[0]
    n_FP = pair_FP_np.shape[0] // n
    threshold_sq = float(low_dist_thres) ** 2
    # Oversample to ensure enough valid candidates after filtering
    sample_size = max(n_FP * 8, 64)

    # Random candidates on GPU
    candidates = mx.random.randint(0, n, shape=(n, sample_size))  # (n, S)

    # Mask: not self
    src_ids = mx.arange(n).reshape(n, 1)  # (n, 1)
    mask = candidates != src_ids

    # Mask: not neighbors
    n_neighbors = pair_neighbors_np.shape[0] // n if pair_neighbors_np.size else 0
    if n_neighbors > 0:
        nb_cols = mx.array(pair_neighbors_np[:, 1].reshape(n, n_neighbors))  # (n, K)
        for k in range(n_neighbors):
            mask = mask & (candidates != nb_cols[:, k:k + 1])

    # Mask: within distance threshold
    diff = Y_mx[candidates] - Y_mx[src_ids]         # (n, S, dim)
    dist_sq = mx.sum(diff * diff, axis=2)            # (n, S)
    mask = mask & (dist_sq <= threshold_sq)

    # Sort valid candidates first per row using a large-value trick:
    # invalid candidates get sort key = sample_size, valid ones get their column index
    col_idx = mx.broadcast_to(mx.arange(sample_size).reshape(1, sample_size), (n, sample_size))
    sort_key = mx.where(mask, col_idx, mx.array(sample_size))
    order = mx.argsort(sort_key, axis=1)  # valid first per row

    # Gather sorted candidates, take first n_FP per row
    sorted_cands = mx.take_along_axis(candidates, order, axis=1)[:, :n_FP]  # (n, n_FP)
    sorted_valid = mx.take_along_axis(mask, order, axis=1)[:, :n_FP]        # (n, n_FP)

    # Fall back to existing FP pairs where we didn't find enough valid candidates
    existing_dst = mx.array(pair_FP_np[:, 1].reshape(n, n_FP))
    result_dst = mx.where(sorted_valid, sorted_cands, existing_dst)
    mx.eval(result_dst)

    updated = pair_FP_np.copy()
    updated[:, 1] = np.array(result_dst).ravel()
    return updated


def _eigh_cpu(matrix_mx):
    """Compute eigendecomposition on CPU with an MLX→NumPy fallback."""
    linalg_eigh = getattr(mx.linalg, "eigh", None)
    if linalg_eigh is not None:
        eigvals, eigvecs = linalg_eigh(matrix_mx, stream=mx.cpu)
        mx.eval(eigvals, eigvecs)
        return eigvals, eigvecs

    matrix_np = np.array(matrix_mx, dtype=np.float32)
    eigvals_np, eigvecs_np = np.linalg.eigh(matrix_np)
    eigvals = mx.array(eigvals_np.astype(np.float32))
    eigvecs = mx.array(eigvecs_np.astype(np.float32))
    mx.eval(eigvals, eigvecs)
    return eigvals, eigvecs


def _pca_init(X_mx, n_components):
    """PCA initialisation via covariance eigendecomposition or SVD."""
    mean = mx.mean(X_mx, axis=0, keepdims=True)
    X_centered = X_mx - mean
    # Use SVD on the centred data
    # For large n, compute on covariance matrix
    n, d = X_centered.shape
    if n > d:
        cov = (X_centered.T @ X_centered) / (n - 1)
        eigvals, eigvecs = _eigh_cpu(cov)
        # eigh returns ascending order, take last n_components
        idx = mx.argsort(eigvals)
        eigvecs = eigvecs[:, idx[-n_components:]]
        eigvecs = eigvecs[:, ::-1]  # descending
        Y = X_centered @ eigvecs
    else:
        U, S, _ = mx.linalg.svd(X_centered, stream=mx.cpu)
        Y = U[:, :n_components] * S[:n_components]

    return Y * 0.01


def _pair_indices(pair_array):
    """Convert a pair array into contiguous MLX source/destination indices."""
    src = mx.array(pair_array[:, 0].copy())
    dst = mx.array(pair_array[:, 1].copy())
    return src, dst


def _step_with_mn(Y, m, v, src_nn, dst_nn, src_mn, dst_mn, src_fp, dst_fp, w_nb, w_mn, w_fp, lr_t):
    """One optimisation step with mid-near pairs."""
    grad = mx.zeros_like(Y)

    # NN: attractive
    diff = Y[src_nn] - Y[dst_nn]
    d = mx.sum(diff * diff, axis=1, keepdims=True) + 1.0
    g = (w_nb * 20.0 / ((10.0 + d) * (10.0 + d))) * diff
    grad = grad.at[src_nn].add(g)
    grad = grad.at[dst_nn].add(-g)

    # MN: attractive
    diff = Y[src_mn] - Y[dst_mn]
    d = mx.sum(diff * diff, axis=1, keepdims=True) + 1.0
    g = (w_mn * 20000.0 / ((10000.0 + d) * (10000.0 + d))) * diff
    grad = grad.at[src_mn].add(g)
    grad = grad.at[dst_mn].add(-g)

    # FP: repulsive
    diff = Y[src_fp] - Y[dst_fp]
    d = mx.sum(diff * diff, axis=1, keepdims=True) + 1.0
    g = (w_fp * 2.0 / ((1.0 + d) * (1.0 + d))) * diff
    grad = grad.at[src_fp].add(-g)
    grad = grad.at[dst_fp].add(g)

    # Adam
    m = 0.9 * m + 0.1 * grad
    v = 0.999 * v + 0.001 * (grad * grad)
    Y = Y - lr_t * m / (mx.sqrt(v) + 1e-7)
    return Y, m, v


def _step_no_mn(Y, m, v, src_nn, dst_nn, src_fp, dst_fp, w_nb, w_fp, lr_t):
    """One optimisation step without mid-near pairs."""
    grad = mx.zeros_like(Y)

    # NN: attractive
    diff = Y[src_nn] - Y[dst_nn]
    d = mx.sum(diff * diff, axis=1, keepdims=True) + 1.0
    g = (w_nb * 20.0 / ((10.0 + d) * (10.0 + d))) * diff
    grad = grad.at[src_nn].add(g)
    grad = grad.at[dst_nn].add(-g)

    # FP: repulsive
    diff = Y[src_fp] - Y[dst_fp]
    d = mx.sum(diff * diff, axis=1, keepdims=True) + 1.0
    g = (w_fp * 2.0 / ((1.0 + d) * (1.0 + d))) * diff
    grad = grad.at[src_fp].add(-g)
    grad = grad.at[dst_fp].add(g)

    # Adam
    m = 0.9 * m + 0.1 * grad
    v = 0.999 * v + 0.001 * (grad * grad)
    Y = Y - lr_t * m / (mx.sqrt(v) + 1e-7)
    return Y, m, v


def _step_no_mn_local(Y, m, v, src_nn, dst_nn, src_fp, dst_fp, w_nb, w_fp, lr_t, nn_scale):
    """One LocalMAP optimisation step with local graph adjustment."""
    grad = mx.zeros_like(Y)

    # NN: attractive with extra local adjustment weighting.
    diff = Y[src_nn] - Y[dst_nn]
    d = mx.sum(diff * diff, axis=1, keepdims=True) + 1.0
    g = (w_nb * 20.0 * nn_scale / (((10.0 + d) * (10.0 + d)) * mx.sqrt(d))) * diff
    grad = grad.at[src_nn].add(g)
    grad = grad.at[dst_nn].add(-g)

    # FP: repulsive
    diff = Y[src_fp] - Y[dst_fp]
    d = mx.sum(diff * diff, axis=1, keepdims=True) + 1.0
    g = (w_fp * 2.0 / ((1.0 + d) * (1.0 + d))) * diff
    grad = grad.at[src_fp].add(-g)
    grad = grad.at[dst_fp].add(g)

    # Adam
    m = 0.9 * m + 0.1 * grad
    v = 0.999 * v + 0.001 * (grad * grad)
    Y = Y - lr_t * m / (mx.sqrt(v) + 1e-7)
    return Y, m, v


_STEP_WITH_MN = mx.compile(_step_with_mn)
_STEP_NO_MN = mx.compile(_step_no_mn)
_STEP_NO_MN_LOCAL = mx.compile(_step_no_mn_local)


class PaCMAP:
    """PaCMAP dimensionality reduction using MLX on Apple Silicon.

    Parameters
    ----------
    n_components : int, default=2
    n_neighbors : int, default=10
    MN_ratio : float, default=0.5
    FP_ratio : float, default=2.0
    lr : float, default=1.0
    num_iters : tuple of 3 ints, default=(100, 100, 250)
    random_state : int or None, default=None
    verbose : bool, default=False
    apply_pca : bool, default=True
    """

    def __init__(
        self,
        n_components=2,
        n_neighbors=10,
        MN_ratio=0.5,
        FP_ratio=2.0,
        lr=1.0,
        num_iters=(100, 100, 250),
        random_state=None,
        verbose=False,
        apply_pca=True,
        knn_method: str = "auto",
    ):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.MN_ratio = MN_ratio
        self.FP_ratio = FP_ratio
        self.lr = lr
        if isinstance(num_iters, int):
            self.num_iters = (100, 100, num_iters)
        else:
            self.num_iters = tuple(num_iters)
        self.random_state = random_state
        self.verbose = verbose
        self.apply_pca = apply_pca
        self.knn_method = knn_method
        self.embedding_ = None

    def _log(self, msg):
        if self.verbose:
            print(msg)

    def _preprocess(self, X_np):
        """Preprocess: if dim > 100, reduce to 100 via SVD. Otherwise normalise."""
        n, dim = X_np.shape

        if dim > 100 and self.apply_pca:
            # Centre and reduce via truncated SVD in MLX
            xmean = np.mean(X_np, axis=0)
            X_np = X_np - xmean
            X_mx = mx.array(X_np)
            # Compute top-100 components via covariance eigendecomposition
            cov = (X_mx.T @ X_mx) / (n - 1)
            mx.eval(cov)
            eigvals, eigvecs = _eigh_cpu(cov)
            # Take top 100
            idx = mx.argsort(eigvals)[-100:]
            idx = idx[::-1]
            proj = eigvecs[:, idx]
            X_mx = X_mx @ proj
            mx.eval(X_mx)
            self._pca_solution = True
            self._tsvd_proj = proj
            self._xmean = xmean
            self._log("Applied PCA, dimensionality reduced to 100")
            return X_mx

        # Normalise when PCA preprocessing is disabled or unnecessary.
        xmin = np.min(X_np)
        X_np = X_np - xmin
        xmax = np.max(X_np)
        X_np = X_np / xmax
        xmean = np.mean(X_np, axis=0)
        X_np = X_np - xmean
        self._pca_solution = False
        self._xmin = xmin
        self._xmax = xmax
        self._xmean = xmean
        self._log("X normalised")
        return mx.array(X_np)

    def _decide_num_pairs(self, n):
        """Determine pair counts with PaCMAP-compatible constraints."""
        if self.n_neighbors is None:
            if n <= 10_000:
                n_neighbors = 10
            else:
                n_neighbors = int(round(10 + 15 * (np.log10(n) - 4)))
        else:
            n_neighbors = int(self.n_neighbors)

        n_MN = int(round(n_neighbors * self.MN_ratio))
        n_FP = int(round(n_neighbors * self.FP_ratio))

        if n - 1 < n_neighbors:
            self._log("Sample size is smaller than requested neighbours; reducing n_neighbors")
        n_neighbors = min(n_neighbors, n - 1)

        if n - 1 < n_MN:
            self._log("Sample size is smaller than requested mid-near pairs; reducing n_MN")
        n_MN = min(n_MN, n - 1)

        if n - 1 < n_FP:
            self._log("Sample size is smaller than requested further pairs; reducing n_FP")
        n_FP = min(n_FP, max(0, n - 1 - n_neighbors))

        if n_neighbors + n_MN + n_FP >= n:
            self._log(
                "Sample size is smaller than total assigned points; reorganising n_neighbors, n_MN, n_FP"
            )
            denom = 1.0 + self.MN_ratio + self.FP_ratio
            n_neighbors = max(1, int(n / denom))
            n_MN = int(n_neighbors * self.MN_ratio)
            n_FP = int(n_neighbors * self.FP_ratio)
            n_neighbors = min(n_neighbors, n - 1)
            n_MN = min(n_MN, n - 1)
            n_FP = min(n_FP, max(0, n - 1 - n_neighbors))

        if n_neighbors < 1:
            raise ValueError("The number of nearest neighbours can't be less than 1")

        return n_neighbors, n_MN, n_FP

    def _fit_transform_impl(self, X, init="pca", epoch_callback=None, low_dist_thres=None):
        """Shared implementation for PaCMAP and LocalMAP."""
        start_time = time.time()
        X_np = np.array(X, dtype=np.float32)
        n, _ = X_np.shape
        if n <= 0:
            raise ValueError("The sample size must be larger than 0")
        rng = np.random.default_rng(self.random_state)
        if self.random_state is not None:
            mx.random.seed(self.random_state)
        localmap = low_dist_thres is not None

        # Preprocess
        X_mx = self._preprocess(X_np.copy())
        mx.eval(X_mx)
        X_proc_np = np.array(X_mx)

        # Determine pair counts
        n_neighbors, n_MN, n_FP = self._decide_num_pairs(n)
        self._log(
            f"n={n}, dim={X_mx.shape[1]}, n_neighbors={n_neighbors}, n_MN={n_MN}, n_FP={n_FP}"
        )

        # KNN
        self._log("Computing KNN...")
        t0 = time.time()
        from mlx_vis._knn import compute_knn

        knn_k = min(n - 1, n_neighbors + 50)
        knn_indices_np, knn_distances_np = compute_knn(
            np.array(X_mx),
            knn_k,
            method=self.knn_method,
            verbose=self.verbose,
            random_state=self.random_state,
        )
        self._log(f"KNN done in {time.time() - t0:.1f}s")

        # Sample pairs (all in numpy, pre-computed)
        self._log("Sampling neighbour pairs...")
        pair_neighbors = _sample_neighbors(knn_distances_np, knn_indices_np, n_neighbors)

        self._log("Sampling MN pairs...")
        pair_MN = _sample_MN_pairs(X_mx, n_MN, rng)

        self._log("Sampling FP pairs...")
        pair_FP = _sample_FP_pairs(n, pair_neighbors, n_neighbors, n_FP, rng)

        self._log(f"Pairs: NN={pair_neighbors.shape}, MN={pair_MN.shape}, FP={pair_FP.shape}")

        # Initialise embedding
        if init == "pca":
            if self._pca_solution:
                Y = mx.array(X_proc_np[:, : self.n_components]) * 0.01
            else:
                Y = _pca_init(X_mx, self.n_components)
        elif init == "random":
            Y = mx.array(rng.normal(size=(n, self.n_components)).astype(np.float32)) * 0.0001
        else:
            raise ValueError(f"Unknown init: {init}")
        mx.eval(Y)

        # Adam state
        m = mx.zeros_like(Y)
        v = mx.zeros_like(Y)
        beta1 = 0.9
        beta2 = 0.999
        lr = self.lr

        num_iters_total = sum(self.num_iters)
        w_MN_init = 1000.0
        phase1, phase2, _ = self.num_iters
        local_nn_scale = mx.array(float(low_dist_thres) / 2.0) if localmap else None

        src_nn, dst_nn = _pair_indices(pair_neighbors)
        src_mn, dst_mn = _pair_indices(pair_MN)
        src_fp, dst_fp = _pair_indices(pair_FP)
        mx.eval(src_nn, dst_nn, src_mn, dst_mn, src_fp, dst_fp)

        if epoch_callback is not None:
            epoch_callback(0, np.array(Y))

        self._log("Starting optimisation...")

        for itr in range(num_iters_total):
            lr_t = lr * math.sqrt(1.0 - beta2 ** (itr + 1)) / (1.0 - beta1 ** (itr + 1))
            lr_mx = mx.array(lr_t)

            if itr < phase1:
                w_MN = (1.0 - itr / phase1) * w_MN_init + (itr / phase1) * 3.0
                Y, m, v = _STEP_WITH_MN(
                    Y,
                    m,
                    v,
                    src_nn,
                    dst_nn,
                    src_mn,
                    dst_mn,
                    src_fp,
                    dst_fp,
                    mx.array(2.0),
                    mx.array(w_MN),
                    mx.array(1.0),
                    lr_mx,
                )
            elif itr < phase1 + phase2:
                Y, m, v = _STEP_WITH_MN(
                    Y,
                    m,
                    v,
                    src_nn,
                    dst_nn,
                    src_mn,
                    dst_mn,
                    src_fp,
                    dst_fp,
                    mx.array(3.0),
                    mx.array(3.0),
                    mx.array(1.0),
                    lr_mx,
                )
            elif localmap:
                Y, m, v = _STEP_NO_MN_LOCAL(
                    Y,
                    m,
                    v,
                    src_nn,
                    dst_nn,
                    src_fp,
                    dst_fp,
                    mx.array(1.0),
                    mx.array(1.0),
                    lr_mx,
                    local_nn_scale,
                )
            else:
                Y, m, v = _STEP_NO_MN(
                    Y,
                    m,
                    v,
                    src_nn,
                    dst_nn,
                    src_fp,
                    dst_fp,
                    mx.array(1.0),
                    mx.array(1.0),
                    lr_mx,
                )

            if localmap and itr > phase1 + phase2 and itr % 10 == 0 and pair_FP.size:
                mx.eval(Y, m, v)
                pair_FP = _resample_local_fp_pairs(pair_neighbors, pair_FP, Y, low_dist_thres, rng)
                src_fp, dst_fp = _pair_indices(pair_FP)
                mx.eval(src_fp, dst_fp)

            if epoch_callback is not None:
                mx.eval(Y, m, v)
                epoch_callback(itr + 1, np.array(Y))
            elif (itr + 1) % 10 == 0:
                mx.eval(Y, m, v)

        mx.eval(Y)
        elapsed = time.time() - start_time
        self._log(f"Elapsed time: {elapsed:.2f}s")

        self.embedding_ = np.array(Y)
        return self.embedding_

    def fit_transform(self, X, init="pca", epoch_callback=None):
        """Fit and return the low-dimensional embedding.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
        init : str, default="pca"
        epoch_callback : callable, optional
            Called as epoch_callback(epoch, Y_numpy) at init and each iteration.

        Returns
        -------
        Y : numpy.ndarray of shape (n_samples, n_components)
        """
        return self._fit_transform_impl(X, init=init, epoch_callback=epoch_callback)


class LocalMAP(PaCMAP):
    """LocalMAP extends PaCMAP with local graph adjustment in the final phase."""

    def __init__(
        self,
        n_components=2,
        n_neighbors=10,
        MN_ratio=0.5,
        FP_ratio=2.0,
        lr=1.0,
        num_iters=(100, 100, 250),
        random_state=None,
        verbose=False,
        apply_pca=True,
        knn_method: str = "auto",
        low_dist_thres=10.0,
    ):
        super().__init__(
            n_components=n_components,
            n_neighbors=n_neighbors,
            MN_ratio=MN_ratio,
            FP_ratio=FP_ratio,
            lr=lr,
            num_iters=num_iters,
            random_state=random_state,
            verbose=verbose,
            apply_pca=apply_pca,
            knn_method=knn_method,
        )
        self.low_dist_thres = float(low_dist_thres)

    def fit_transform(self, X, init="pca", epoch_callback=None):
        """Fit LocalMAP and return the low-dimensional embedding."""
        return self._fit_transform_impl(
            X,
            init=init,
            epoch_callback=epoch_callback,
            low_dist_thres=self.low_dist_thres,
        )
