"""t-SNE implementation in pure MLX for Apple Silicon."""

import mlx.core as mx
import numpy as np


def _chunked_knn(X_mx, k):
    """Brute-force KNN using chunked pairwise distances on GPU."""
    n = X_mx.shape[0]
    chunk_size = min(n, max(1000, 500_000_000 // (n * 4)))
    
    sq_norms = mx.sum(X_mx * X_mx, axis=1)
    mx.eval(sq_norms)
    
    all_indices = []
    all_distances = []
    prev = None
    
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        sq_chunk = sq_norms[start:end, None]
        
        dists = mx.maximum(
            sq_chunk + sq_norms[None, :] - 2.0 * (X_mx[start:end] @ X_mx.T),
            0.0
        )
        
        if end - start == n:
            dists = dists + mx.eye(n) * 1e30
        else:
            arange_chunk = mx.arange(start, end)[:, None]
            arange_all = mx.arange(n)[None, :]
            dists = dists + (arange_chunk == arange_all).astype(mx.float32) * 1e30
        
        indices = mx.argsort(dists, axis=1)[:, :k]
        gathered = mx.sqrt(mx.take_along_axis(dists, indices, axis=1))
        
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


def _searchsorted(sorted_array: mx.array, values: mx.array) -> mx.array:
    """Vectorized binary search on GPU."""
    n_arr = sorted_array.shape[0]
    m = values.shape[0]
    lo = mx.zeros((m,), dtype=mx.int32)
    hi = mx.full((m,), n_arr, dtype=mx.int32)
    for _ in range(int(np.ceil(np.log2(max(n_arr, 2)))) + 1):
        mid = (lo + hi) // 2
        mid_clamped = mx.minimum(mid, n_arr - 1)
        go_right = sorted_array[mid_clamped] < values
        lo = mx.where(go_right, mid + 1, lo)
        hi = mx.where(go_right, hi, mid)
    return lo


class TSNE:
    """t-SNE dimensionality reduction using MLX on Metal GPU.

    Uses sparse P matrix (KNN), exact repulsive forces via chunked GPU computation,
    and momentum-based gradient descent with adaptive gains.

    Parameters:
        n_components: Embedding dimension (default 2).
        perplexity: Effective number of neighbors (default 30.0).
        learning_rate: GD learning rate (default 200.0).
        n_iter: Number of iterations (default 1000).
        early_exaggeration: P multiplier during early phase (default 12.0).
        early_exaggeration_iter: Duration of early exaggeration (default 250).
        random_state: Random seed.
        verbose: Print every N iters (0 = silent).
        pca_dim: PCA preprocessing dim (default 50). None to skip.
    """

    def __init__(
        self,
        n_components: int = 2,
        perplexity: float = 30.0,
        learning_rate: float = 200.0,
        n_iter: int = 1000,
        early_exaggeration: float = 12.0,
        early_exaggeration_iter: int = 250,
        random_state: int | None = None,
        verbose: int = 0,
        pca_dim: int | None = 50,
    ):
        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.early_exaggeration = early_exaggeration
        self.early_exaggeration_iter = early_exaggeration_iter
        self.random_state = random_state
        self.verbose = verbose
        self.pca_dim = pca_dim
        self.embedding_ = None

    def fit_transform(self, X, epoch_callback=None) -> np.ndarray:
        if isinstance(X, mx.array):
            X = np.array(X)
        X = np.asarray(X, dtype=np.float32)
        n, dim = X.shape

        # PCA preprocessing
        if self.pca_dim is not None and dim > self.pca_dim:
            X_mx = mx.array(X)
            mean = mx.mean(X_mx, axis=0)
            X_centered = X_mx - mean
            cov = (X_centered.T @ X_centered) / (n - 1)
            mx.eval(cov)
            eigvals, eigvecs = mx.linalg.eigh(cov, stream=mx.cpu)
            mx.eval(eigvals, eigvecs)
            proj = eigvecs[:, -self.pca_dim:][:, ::-1]
            X_pca = X_centered @ proj
            mx.eval(X_pca)
            X_for_knn = np.array(X_pca)
        else:
            X_for_knn = X

        # KNN
        k = min(int(3 * self.perplexity), n - 1)
        X_mx = mx.array(X_for_knn)
        knn_indices, knn_dists = _chunked_knn(X_mx, k)
        mx.eval(knn_indices, knn_dists)

        # Build P matrix
        edge_from, edge_to, edge_weights = self._build_p(knn_indices, knn_dists, n)
        mx.eval(edge_from, edge_to, edge_weights)

        # PCA init
        if self.random_state is not None:
            mx.random.seed(self.random_state)
        mean = mx.mean(X_mx, axis=0)
        X_centered = X_mx - mean
        cov = (X_centered.T @ X_centered) / (n - 1)
        mx.eval(cov)
        eigvals, eigvecs = mx.linalg.eigh(cov, stream=mx.cpu)
        mx.eval(eigvals, eigvecs)
        components = eigvecs[:, -self.n_components:][:, ::-1]
        Y = (X_centered @ components) * 1e-4
        mx.eval(Y)

        # Optimize
        Y = self._optimize(edge_from, edge_to, edge_weights, Y, n, epoch_callback)
        mx.eval(Y)
        self.embedding_ = np.array(Y)
        return self.embedding_

    def _build_p(self, knn_indices, knn_dists, n):
        """Build symmetrized sparse P matrix with sum(P) = 1."""
        k = knn_indices.shape[1]

        # Binary search for bandwidth (vectorized)
        lo = mx.full((n,), 1e-20)
        hi = mx.full((n,), 1e4)
        beta = mx.ones((n,))
        target_H = float(np.log(self.perplexity))
        
        sq_dists = knn_dists * knn_dists
        mx.eval(sq_dists)

        for _ in range(64):
            logits = -beta[:, None] * sq_dists
            logits_max = mx.max(logits, axis=1, keepdims=True)
            shifted = logits - logits_max
            exp_l = mx.exp(shifted)
            sum_exp = mx.sum(exp_l, axis=1)
            
            p = exp_l / sum_exp[:, None]
            H = mx.log(sum_exp) - mx.sum(p * shifted, axis=1)

            converged = mx.abs(H - target_H) < 1e-5
            too_high = (H > target_H) & ~converged
            too_low = (H < target_H) & ~converged

            new_lo = mx.where(too_high, beta, lo)
            new_hi = mx.where(too_low, beta, hi)
            beta = mx.where(too_high, mx.where(new_hi < 1e4, (beta + new_hi) / 2.0, beta * 2.0), beta)
            beta = mx.where(too_low, mx.where(new_lo > 1e-20, (new_lo + beta) / 2.0, beta / 2.0), beta)
            lo = new_lo
            hi = new_hi
            mx.eval(beta)
            if bool(mx.all(converged)):
                break

        # Final conditional P(j|i)
        logits = -beta[:, None] * sq_dists
        logits_max = mx.max(logits, axis=1, keepdims=True)
        exp_l = mx.exp(logits - logits_max)
        sum_exp = mx.sum(exp_l, axis=1, keepdims=True)
        weights = exp_l / sum_exp
        mx.eval(weights)

        # Sparse edges
        rows_np = np.repeat(np.arange(n, dtype=np.int32), k)
        cols_np = np.array(knn_indices).ravel().astype(np.int32)
        vals_np = np.array(weights.reshape(-1))

        # Symmetrize via searchsorted
        rows_mx = mx.array(rows_np)
        cols_mx = mx.array(cols_np)
        vals_mx = mx.array(vals_np)
        
        fwd_keys = rows_mx.astype(mx.int64) * n + cols_mx.astype(mx.int64)
        rev_keys = cols_mx.astype(mx.int64) * n + rows_mx.astype(mx.int64)
        sort_idx = mx.argsort(fwd_keys)
        sorted_keys = fwd_keys[sort_idx]
        sorted_vals = vals_mx[sort_idx]
        pos = _searchsorted(sorted_keys, rev_keys)
        pos = mx.minimum(pos, sorted_keys.shape[0] - 1)
        matched = sorted_keys[pos] == rev_keys
        w_rev = mx.where(matched, sorted_vals[pos], 0.0)
        
        # p_ij = (p(j|i) + p(i|j)) / (2n)
        w_sym = (vals_mx + w_rev) / (2.0 * n)
        mx.eval(w_sym)

        # Make both directions explicit, dedup on GPU
        all_rows = mx.concatenate([rows_mx, cols_mx])
        all_cols = mx.concatenate([cols_mx, rows_mx])
        all_vals = mx.concatenate([w_sym, w_sym])
        
        keys = all_rows.astype(mx.int64) * n + all_cols.astype(mx.int64)
        sort_idx2 = mx.argsort(keys)
        sorted_keys2 = keys[sort_idx2]
        # Mark first occurrence: keep only where key differs from previous
        is_first = mx.concatenate([mx.array([True]), sorted_keys2[1:] != sorted_keys2[:-1]])
        # Compact unique indices using argsort on is_first (descending)
        # Alternative: just use the mask to select
        is_first_np = np.array(is_first)
        unique_positions = np.nonzero(is_first_np)[0]
        final_idx = sort_idx2[mx.array(unique_positions)]
        mx.eval(final_idx)
        
        return all_rows[final_idx], all_cols[final_idx], all_vals[final_idx]

    @staticmethod
    @mx.compile
    def _repulsive_grad_full(Y, eye_mask):
        """Compiled repulsive gradient. eye_mask = 1 - eye(n), precomputed."""
        sq_norms = mx.sum(Y * Y, axis=1)
        dsq = mx.maximum(sq_norms[:, None] + sq_norms[None, :] - 2.0 * (Y @ Y.T), 0.0)
        kernel = eye_mask / (1.0 + dsq)
        Z = mx.sum(kernel)
        ksq = kernel * kernel
        rep_grad = mx.sum(ksq, axis=1, keepdims=True) * Y - ksq @ Y
        return Z, rep_grad

    @staticmethod
    @mx.compile
    def _chunk_kernel(Y_chunk, Y_full, sq_norms_chunk, sq_norms_full, self_mask):
        """Compiled chunk repulsive: compute Z_chunk and rep_grad_chunk."""
        dsq = mx.maximum(sq_norms_chunk[:, None] + sq_norms_full[None, :] - 2.0 * (Y_chunk @ Y_full.T), 0.0)
        kernel = self_mask / (1.0 + dsq)
        Z_chunk = mx.sum(kernel)
        ksq = kernel * kernel
        rg = mx.sum(ksq, axis=1, keepdims=True) * Y_chunk - ksq @ Y_full
        return Z_chunk, rg

    def _repulsive_grad_chunked(self, Y, n, chunk_size, self_masks):
        """Chunked repulsive gradient with compiled chunks."""
        Z = mx.array(0.0)
        rep_grad = mx.zeros_like(Y)
        sq_norms = mx.sum(Y * Y, axis=1)
        
        for i, start in enumerate(range(0, n, chunk_size)):
            end = min(start + chunk_size, n)
            Z_c, rg = self._chunk_kernel(
                Y[start:end], Y, sq_norms[start:end], sq_norms, self_masks[i]
            )
            Z = Z + Z_c
            rep_grad = rep_grad.at[start:end].add(rg)
            mx.eval(Z, rep_grad)
        
        return Z, rep_grad

    @staticmethod
    def _fft_repulsive(Y, n, nodes, denom, fft_state):
        """Full MLX GPU FFT repulsive gradient.
        
        Scatter/gather via sparse interpolation matrix (matmul on GPU).
        FFT convolution via mx.fft on GPU. Zero numpy in hot path.
        
        fft_state: mutable dict for caching interpolation indices across epochs.
        Returns (Z, rep_grad) as mx.arrays.
        """
        n_interp = 3
        
        # Grid setup
        y_min = mx.min(Y, axis=0)
        y_max = mx.max(Y, axis=0)
        mx.eval(y_min, y_max)
        y_min_f = y_min.tolist()
        y_max_f = y_max.tolist()
        cr = [y_max_f[d] - y_min_f[d] for d in range(2)]
        pad = [c * 0.01 + 1e-8 for c in cr]
        bl = [y_min_f[d] - pad[d] for d in range(2)]
        cr = [cr[d] + 2 * pad[d] for d in range(2)]
        
        n_boxes = max(50, int(np.ceil(max(cr))))
        ng = n_boxes * n_interp
        
        # Scale points to [0, n_boxes] on GPU
        bl_mx = mx.array(bl, dtype=mx.float32)
        cr_mx = mx.array(cr, dtype=mx.float32)
        y_scaled = (Y - bl_mx) / cr_mx * n_boxes
        
        box_idx = mx.clip(mx.floor(y_scaled).astype(mx.int32), 0, n_boxes - 1)
        y_rel = y_scaled - box_idx.astype(mx.float32)
        
        # Lagrange weights on GPU: (n, 3) per dim
        nodes_mx = mx.array(nodes, dtype=mx.float32)
        denom_mx = mx.array(denom, dtype=mx.float32)
        
        # For n_interp=3: L_j(x) = prod_{k!=j}(x - nodes[k]) / denom[j]
        # Compute all differences: (n, 3) = y_rel[:, d:d+1] - nodes[None, :]
        diff_x = y_rel[:, 0:1] - nodes_mx[None, :]  # (n, 3)
        diff_y = y_rel[:, 1:2] - nodes_mx[None, :]  # (n, 3)
        
        # L_0 = diff[1]*diff[2]/denom[0], L_1 = diff[0]*diff[2]/denom[1], L_2 = diff[0]*diff[1]/denom[2]
        wx0 = diff_x[:, 1] * diff_x[:, 2] / denom_mx[0]
        wx1 = diff_x[:, 0] * diff_x[:, 2] / denom_mx[1]
        wx2 = diff_x[:, 0] * diff_x[:, 1] / denom_mx[2]
        wy0 = diff_y[:, 1] * diff_y[:, 2] / denom_mx[0]
        wy1 = diff_y[:, 0] * diff_y[:, 2] / denom_mx[1]
        wy2 = diff_y[:, 0] * diff_y[:, 1] / denom_mx[2]
        
        # Build sparse interpolation matrix as dense gather/scatter indices
        # Each point maps to 9 grid nodes (3x3 tensor product)
        # flat_idx: (n, 9), weights: (n, 9)
        bx = box_idx[:, 0]  # (n,)
        by = box_idx[:, 1]  # (n,)
        
        gx0 = bx * n_interp
        gx1 = gx0 + 1
        gx2 = gx0 + 2
        gy0 = by * n_interp
        gy1 = gy0 + 1
        gy2 = gy0 + 2
        
        # 9 flat indices: (n, 9)
        flat_idx = mx.stack([
            gx0 * ng + gy0, gx0 * ng + gy1, gx0 * ng + gy2,
            gx1 * ng + gy0, gx1 * ng + gy1, gx1 * ng + gy2,
            gx2 * ng + gy0, gx2 * ng + gy1, gx2 * ng + gy2,
        ], axis=1)  # (n, 9)
        
        # 9 weights: (n, 9)
        interp_w = mx.stack([
            wx0 * wy0, wx0 * wy1, wx0 * wy2,
            wx1 * wy0, wx1 * wy1, wx1 * wy2,
            wx2 * wy0, wx2 * wy1, wx2 * wy2,
        ], axis=1)  # (n, 9)
        
        mx.eval(flat_idx, interp_w)
        
        # Charges: (n, 4) = [1, 1, Y[:,0], Y[:,1]]
        ones = mx.ones((n,), dtype=mx.float32)
        charges = mx.stack([ones, ones, Y[:, 0], Y[:, 1]], axis=1)  # (n, 4)
        
        # Step 1: Scatter charges to grid via 1D scatter_add
        # For each of 9 interp nodes per point: grid[flat_idx] += interp_w * charge
        # Expand: weighted_charges (n, 9, 4) = interp_w[:, :, None] * charges[:, None, :]
        weighted_charges = interp_w[:, :, None] * charges[:, None, :]  # (n, 9, 4)
        
        # Flatten to (n*9, 4) with indices (n*9,)
        wc_flat = weighted_charges.reshape(n * 9, 4)
        idx_flat = flat_idx.reshape(n * 9)
        
        # Scatter add
        w_grid_flat = mx.zeros((ng * ng, 4), dtype=mx.float32)
        w_grid_flat = w_grid_flat.at[idx_flat].add(wc_flat)
        
        w_grid = w_grid_flat.reshape(ng, ng, 4)
        
        # Step 2: FFT convolution on GPU
        M = ng
        hx = cr[0] / M
        hy = cr[1] / M
        
        ax = mx.arange(2 * M, dtype=mx.float32)
        ax = mx.where(ax >= M, ax - 2 * M, ax)
        dsq_grid = (ax[:, None] * hx) ** 2 + (ax[None, :] * hy) ** 2
        k1 = 1.0 / (1.0 + dsq_grid)
        fk1 = mx.fft.rfft2(k1)
        fk2 = mx.fft.rfft2(k1 * k1)
        
        # Pad and convolve 4 terms -- batch as (4, 2M, 2M)
        w_padded = mx.zeros((4, 2 * M, 2 * M), dtype=mx.float32)
        # Transpose w_grid to (4, M, M) for batched FFT
        wg_t = mx.transpose(w_grid, (2, 0, 1))  # (4, M, M)
        w_padded = w_padded.at[:, :M, :M].add(wg_t)
        
        fw = mx.fft.rfft2(w_padded)  # (4, 2M, M+1)
        
        # Term 0 uses fk1, terms 1-3 use fk2
        fk_batch = mx.stack([fk1, fk2, fk2, fk2])  # (4, 2M, M+1)
        result = mx.fft.irfft2(fw * fk_batch, s=(2 * M, 2 * M))  # (4, 2M, 2M)
        pot = result[:, :M, :M]  # (4, M, M)
        
        # Step 3: Gather potentials back to points
        # pot_flat: (4, M*M) -> transpose to (M*M, 4) for gathering
        pot_flat = pot.reshape(4, M * M).T  # (M*M, 4)
        
        # Gather: for each point, sum over 9 interp nodes
        # gathered = pot_flat[flat_idx] -> (n, 9, 4)
        gathered = pot_flat[idx_flat.reshape(n * 9)]  # (n*9, 4)
        gathered = gathered.reshape(n, 9, 4)
        
        # Weighted sum: phi = sum_j interp_w[:, j] * gathered[:, j, :]
        phi = mx.sum(interp_w[:, :, None] * gathered, axis=1)  # (n, 4)
        
        # Z = sum(phi[:, 0]) - n (remove self-interaction)
        Z = mx.sum(phi[:, 0]) - n
        
        # Repulsive gradient
        rep_grad = mx.stack([
            Y[:, 0] * phi[:, 1] - phi[:, 2],
            Y[:, 1] * phi[:, 1] - phi[:, 3],
        ], axis=1)
        
        return Z, rep_grad

    def _optimize(self, edge_from, edge_to, edge_weights, Y, n, epoch_callback=None):
        """Exact t-SNE gradient with momentum + adaptive gains.
        
        Attractive: sparse (KNN edges). Repulsive: exact all-pairs on GPU.
        Uses matmul-based repulsive (no 3D broadcast tensors).
        """
        n_epochs = self.n_iter
        lr = self.learning_rate

        velocity = mx.zeros_like(Y)
        gains = mx.ones_like(Y)
        min_gain = 0.01

        weights_exag = edge_weights * self.early_exaggeration
        mx.eval(weights_exag)

        # Method selection: FFT for large n (2D only), exact for small n
        use_fft = (n >= 16000) and (Y.shape[1] == 2)
        
        # Full path (compiled): n^2 * 4B < 1GB. Chunked path for larger.
        full_limit = 1_000_000_000
        use_full = (n * n * 4) < full_limit and not use_fft
        chunk_size = min(n, max(512, 2_000_000_000 // (n * 4)))
        
        if use_fft:
            eye_mask = None
            self_masks = None
            # Precompute Lagrange nodes/denom on GPU (constant across epochs)
            h = 1.0 / 3
            fft_nodes = np.array([(0.5 + j) * h for j in range(3)], dtype=np.float32)
            fft_denom = np.ones(3, dtype=np.float32)
            for j in range(3):
                for k in range(3):
                    if k != j:
                        fft_denom[j] *= (fft_nodes[j] - fft_nodes[k])
            fft_state = {}
            if self.verbose > 0:
                print(f"Using FFT-accelerated repulsive on GPU (n={n})")
        elif use_full:
            eye_mask = 1.0 - mx.eye(n)
            mx.eval(eye_mask)
            self_masks = None
        else:
            # Precompute self-interaction masks: (cs, n) with 0 on diagonal
            self_masks = []
            for start in range(0, n, chunk_size):
                end = min(start + chunk_size, n)
                cs = end - start
                mask = 1.0 - (mx.arange(start, end)[:, None] == mx.arange(n)[None, :]).astype(mx.float32)
                mx.eval(mask)
                self_masks.append(mask)

        if epoch_callback is not None:
            epoch_callback(0, np.array(Y))

        for epoch in range(n_epochs):
            momentum = 0.5 if epoch < 250 else 0.8
            w = weights_exag if epoch < self.early_exaggeration_iter else edge_weights

            # Attractive (sparse)
            diff_a = Y[edge_from] - Y[edge_to]
            dsq_a = mx.sum(diff_a * diff_a, axis=1, keepdims=True)
            f_attr = 4.0 * w[:, None] * diff_a / (1.0 + dsq_a)
            
            grad = mx.zeros_like(Y)
            grad = grad.at[edge_from].add(f_attr)

            # Repulsive
            if use_fft:
                Z, rep_grad = self._fft_repulsive(Y, n, fft_nodes, fft_denom, fft_state)
            elif use_full:
                Z, rep_grad = self._repulsive_grad_full(Y, eye_mask)
            else:
                Z, rep_grad = self._repulsive_grad_chunked(Y, n, chunk_size, self_masks)
            
            grad = grad - (4.0 / Z) * rep_grad

            # Phase transition reset
            if epoch == self.early_exaggeration_iter:
                gains = mx.ones_like(Y)
                velocity = mx.zeros_like(Y)

            # Adaptive gains
            inc = (velocity * grad) < 0.0
            gains = mx.where(inc, gains + 0.2, gains * 0.8)
            gains = mx.maximum(gains, min_gain)
            
            velocity = momentum * velocity - lr * (gains * grad)
            Y = Y + velocity
            Y = Y - mx.mean(Y, axis=0)

            # Eval every epoch to prevent graph explosion
            mx.eval(Y, velocity, gains)
            if epoch_callback is not None:
                epoch_callback(epoch + 1, np.array(Y))

            if self.verbose > 0 and (epoch + 1) % self.verbose == 0:
                print(f"Epoch {epoch + 1}/{n_epochs}")

        return Y
