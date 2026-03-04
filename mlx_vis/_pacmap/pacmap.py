"""PaCMAP (Pairwise Controlled Manifold Approximation) in pure MLX."""

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
            0.0
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
            p_idx, p_dist, p_start, p_end = prev
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
    """Sample neighbor pairs using scaled distance (vectorized numpy)."""
    n, k_extra = knn_distances_np.shape
    # sigma = mean of distances to 4th-6th neighbors
    sig = np.maximum(np.mean(knn_distances_np[:, 3:6], axis=1), 1e-10)
    
    # Scale distances: d^2 / (sig[i] * sig[neighbor])
    sig_nb = sig[knn_indices_np]  # (n, k_extra)
    scaled = knn_distances_np ** 2 / (sig[:, None] * sig_nb)  # (n, k_extra)
    
    # Pick top n_neighbors by scaled distance
    sorted_idx = np.argsort(scaled, axis=1)[:, :n_neighbors]  # (n, n_neighbors)
    picked = np.take_along_axis(knn_indices_np, sorted_idx, axis=1)  # (n, n_neighbors)
    
    src = np.repeat(np.arange(n, dtype=np.int32), n_neighbors)
    dst = picked.ravel().astype(np.int32)
    return np.stack([src, dst], axis=1)


def _sample_MN_pairs(X_mx, n_MN, rng):
    """Sample mid-near pairs on GPU. For each point, sample 6 random, pick 2nd closest."""
    n = X_mx.shape[0]
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
    """Sample further pairs (vectorized). Random non-neighbors."""
    # For 70K points with only 10 neighbors each, random sampling almost never
    # hits a neighbor. Just sample randomly and accept the tiny collision rate.
    all_src = np.repeat(np.arange(n, dtype=np.int32), n_FP)
    # Sample with offset to avoid self
    offsets = rng.integers(1, n, size=n * n_FP)
    all_dst = ((all_src + offsets) % n).astype(np.int32)
    
    pair_FP = np.stack([all_src, all_dst], axis=1)
    return pair_FP


def _pca_init(X_mx, n_components):
    """PCA initialization via SVD in MLX."""
    mean = mx.mean(X_mx, axis=0, keepdims=True)
    X_centered = X_mx - mean
    # Use SVD on the centered data
    # For large n, compute on covariance matrix
    n, d = X_centered.shape
    if n > d:
        cov = (X_centered.T @ X_centered) / (n - 1)
        eigvals, eigvecs = mx.linalg.eigh(cov, stream=mx.cpu)
        # eigh returns ascending order, take last n_components
        idx = mx.argsort(eigvals)
        eigvecs = eigvecs[:, idx[-n_components:]]
        eigvecs = eigvecs[:, ::-1]  # descending
        Y = X_centered @ eigvecs
    else:
        U, S, Vt = mx.linalg.svd(X_centered, stream=mx.cpu)
        Y = U[:, :n_components] * S[:n_components]
    
    return Y * 0.01


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
        self.embedding_ = None
    
    def _log(self, msg):
        if self.verbose:
            print(msg)
    
    def _preprocess(self, X_np):
        """Preprocess: if dim > 100, reduce to 100 via SVD. Otherwise normalize."""
        n, dim = X_np.shape
        
        if dim > 100 and self.apply_pca:
            # Center and reduce via truncated SVD in MLX
            xmean = np.mean(X_np, axis=0)
            X_np = X_np - xmean
            X_mx = mx.array(X_np)
            # Compute top-100 components via covariance eigendecomposition
            cov = (X_mx.T @ X_mx) / (n - 1)
            mx.eval(cov)
            eigvals, eigvecs = mx.linalg.eigh(cov, stream=mx.cpu)
            mx.eval(eigvals, eigvecs)
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
        else:
            # Normalize
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
            self._log("X normalized")
            return mx.array(X_np)
    
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
        start_time = time.time()
        X_np = np.array(X, dtype=np.float32)
        n, dim = X_np.shape
        
        rng = np.random.default_rng(self.random_state)
        
        # Preprocess
        X_mx = self._preprocess(X_np.copy())
        mx.eval(X_mx)
        X_proc_np = np.array(X_mx)
        
        # Determine pair counts
        n_neighbors = min(self.n_neighbors, n - 1)
        n_MN = int(self.MN_ratio * n_neighbors)
        n_FP = int(self.FP_ratio * n_neighbors)
        n_MN = min(n_MN, n - 1)
        n_FP = min(n_FP, n - 1)
        
        self._log(f"n={n}, dim={X_mx.shape[1]}, n_neighbors={n_neighbors}, n_MN={n_MN}, n_FP={n_FP}")
        
        # KNN
        self._log("Computing KNN...")
        t0 = time.time()
        knn_indices, knn_distances = _brute_knn(X_mx, n_neighbors + 50)
        knn_indices_np = np.array(knn_indices, dtype=np.int32)
        knn_distances_np = np.array(knn_distances, dtype=np.float32)
        self._log(f"KNN done in {time.time()-t0:.1f}s")
        
        # Sample pairs (all in numpy, pre-computed)
        self._log("Sampling neighbor pairs...")
        pair_neighbors = _sample_neighbors(knn_distances_np, knn_indices_np, n_neighbors)
        
        self._log("Sampling MN pairs...")
        pair_MN = _sample_MN_pairs(X_mx, n_MN, rng)
        
        self._log("Sampling FP pairs...")
        pair_FP = _sample_FP_pairs(n, pair_neighbors, n_neighbors, n_FP, rng)
        
        self._log(f"Pairs: NN={pair_neighbors.shape}, MN={pair_MN.shape}, FP={pair_FP.shape}")
        
        # Initialize embedding
        if init == "pca":
            if self._pca_solution:
                Y = mx.array(X_proc_np[:, :self.n_components]) * 0.01
            else:
                Y = _pca_init(X_mx, self.n_components)
        elif init == "random":
            Y = mx.array(rng.normal(size=(n, self.n_components)).astype(np.float32)) * 0.0001
        else:
            raise ValueError(f"Unknown init: {init}")
        mx.eval(Y)
        
        # Convert pairs to MLX
        pair_nn_mx = mx.array(pair_neighbors)
        pair_mn_mx = mx.array(pair_MN)
        pair_fp_mx = mx.array(pair_FP)
        
        # Adam state
        m = mx.zeros_like(Y)
        v = mx.zeros_like(Y)
        beta1 = 0.9
        beta2 = 0.999
        lr = self.lr
        
        num_iters_total = sum(self.num_iters)
        w_MN_init = 1000.0
        phase1, phase2, _ = self.num_iters
        
        # Pre-build unified pair arrays for each phase to minimize per-iteration work.
        # All pairs: concat source indices, target indices, and per-pair constants
        # NN: attractive, coeff_num=20, denom_offset=10, sign=+1 (grad[i] += w*diff)
        # MN: attractive, coeff_num=20000, denom_offset=10000, sign=+1
        # FP: repulsive, coeff_num=2, denom_offset=1, sign=-1 (grad[i] -= w*diff)
        
        n_nn = pair_neighbors.shape[0]
        n_mn = pair_MN.shape[0]
        n_fp = pair_FP.shape[0]
        
        # Source and target indices for scatter (both directions: i and j)
        # For each pair (i,j) we add to both i and j, so double the indices
        # Direction: for attractive pairs, grad[i] += coeff*diff, grad[j] -= coeff*diff
        #           for repulsive pairs, grad[i] -= coeff*diff, grad[j] += coeff*diff
        # Unified: src_indices gather from, scatter to src_scatter and dst_scatter
        
        src_nn = mx.array(pair_neighbors[:, 0])
        dst_nn = mx.array(pair_neighbors[:, 1])
        src_mn = mx.array(pair_MN[:, 0])
        dst_mn = mx.array(pair_MN[:, 1])
        src_fp = mx.array(pair_FP[:, 0])
        dst_fp = mx.array(pair_FP[:, 1])
        
        if epoch_callback is not None:
            epoch_callback(0, np.array(Y))

        self._log("Starting optimization...")
        
        # Pre-extract column slices as contiguous arrays to avoid repeated indexing
        src_nn = mx.array(pair_neighbors[:, 0].copy())
        dst_nn = mx.array(pair_neighbors[:, 1].copy())
        src_mn = mx.array(pair_MN[:, 0].copy())
        dst_mn = mx.array(pair_MN[:, 1].copy())
        src_fp = mx.array(pair_FP[:, 0].copy())
        dst_fp = mx.array(pair_FP[:, 1].copy())
        mx.eval(src_nn, dst_nn, src_mn, dst_mn, src_fp, dst_fp)
        
        def step_with_mn(Y, m, v, w_nb, w_mn, w_fp, lr_t):
            """One optimization step with MN pairs (phase 1 & 2)."""
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
        
        def step_no_mn(Y, m, v, w_nb, w_fp, lr_t):
            """One optimization step without MN pairs (phase 3)."""
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
        
        # Don't compile - closures with scatter-add are tricky in mx.compile
        step_with_mn_c = step_with_mn
        step_no_mn_c = step_no_mn
        
        for itr in range(num_iters_total):
            if itr < phase1:
                w_MN = (1.0 - itr / phase1) * w_MN_init + (itr / phase1) * 3.0
                lr_t = lr * math.sqrt(1.0 - beta2 ** (itr + 1)) / (1.0 - beta1 ** (itr + 1))
                Y, m, v = step_with_mn_c(Y, m, v, 2.0, w_MN, 1.0, lr_t)
            elif itr < phase1 + phase2:
                lr_t = lr * math.sqrt(1.0 - beta2 ** (itr + 1)) / (1.0 - beta1 ** (itr + 1))
                Y, m, v = step_with_mn_c(Y, m, v, 3.0, 3.0, 1.0, lr_t)
            else:
                lr_t = lr * math.sqrt(1.0 - beta2 ** (itr + 1)) / (1.0 - beta1 ** (itr + 1))
                Y, m, v = step_no_mn_c(Y, m, v, 1.0, 1.0, lr_t)
            
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
