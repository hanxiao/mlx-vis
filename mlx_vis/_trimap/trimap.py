"""TriMap: dimensionality reduction using triplet constraints.

Reference: Amid & Warmuth, "TriMap: Large-scale Dimensionality Reduction
Using Triplets" (arXiv:1910.00204).

Entire optimization on Metal GPU via MLX. Triplet generation in numpy.
"""

import mlx.core as mx
import numpy as np
import time

from mlx_vis._nndescent.nndescent import NNDescent


class TriMap:
    """TriMap dimensionality reduction via triplet constraints.

    Parameters
    ----------
    n_components : int
        Dimension of the embedding (default 2).
    n_neighbors : int
        Number of nearest neighbors for graph construction (default 12).
    n_inliers : int
        Number of inlier neighbors per point for triplet generation (default 12).
    n_outliers : int
        Number of outlier points sampled per inlier (default 4).
    n_random : int
        Number of random triplets per point (default 3).
    n_iters : int
        Number of optimization iterations (default 400).
    lr : float
        Learning rate (default 1000).
    weight_temp : float
        Temperature for tempered log weight transform (default 0.5).
    pca_dim : int
        PCA target dimension if input dim exceeds this (default 100).
    random_state : int or None
        Random seed for reproducibility.
    verbose : bool
        Print progress information.
    """

    def __init__(
        self,
        n_components=2,
        n_neighbors=12,
        n_inliers=12,
        n_outliers=4,
        n_random=3,
        n_iters=400,
        lr=1000,
        weight_temp=0.5,
        pca_dim=100,
        random_state=None,
        verbose=False,
    ):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.n_inliers = n_inliers
        self.n_outliers = n_outliers
        self.n_random = n_random
        self.n_iters = n_iters
        self.lr = lr
        self.weight_temp = weight_temp
        self.pca_dim = pca_dim
        self.random_state = random_state
        self.verbose = verbose

    def fit_transform(self, X, epoch_callback=None):
        """Compute TriMap embedding.

        Args:
            X: Input data array of shape (n_samples, n_features).
            epoch_callback: Optional callback(epoch, Y_numpy) for animation.

        Returns:
            np.ndarray of shape (n_samples, n_components).
        """
        t0 = time.time()
        rng = np.random.RandomState(self.random_state)

        X = np.asarray(X, dtype=np.float32)
        n, dim = X.shape

        if self.verbose:
            print(f"TriMap: {n} points, {dim} dims")

        # PCA preprocessing
        if self.pca_dim is not None and dim > self.pca_dim:
            X_mx = mx.array(X)
            mean = mx.mean(X_mx, axis=0)
            X_centered = X_mx - mean
            cov = (X_centered.T @ X_centered) / (n - 1)
            mx.eval(cov)
            eigvals, eigvecs = mx.linalg.eigh(cov, stream=mx.cpu)
            mx.eval(eigvals, eigvecs)
            proj = eigvecs[:, -self.pca_dim :][:, ::-1]
            X_pca = X_centered @ proj
            mx.eval(X_pca)
            if self.verbose:
                var_retained = mx.sum(eigvals[-self.pca_dim :]) / mx.sum(eigvals)
                mx.eval(var_retained)
                print(
                    f"PCA: {dim} -> {self.pca_dim} dims, "
                    f"{float(var_retained) * 100:.1f}% variance retained"
                )
            X = np.array(X_pca)
            dim = self.pca_dim

        # k-NN via NNDescent
        if self.verbose:
            print("Computing k-NN...")
        t_knn = time.time()
        nn = NNDescent(
            k=self.n_neighbors,
            verbose=self.verbose,
            random_state=self.random_state if self.random_state is not None else 42,
        )
        knn_indices, knn_distances = nn.build(X)
        # knn_indices: (n, k) sorted neighbor indices (no self)
        # knn_distances: (n, k) Euclidean distances, sorted ascending
        if self.verbose:
            print(f"k-NN done in {time.time() - t_knn:.1f}s")

        # Sigma: mean Euclidean distance to 4th-6th nearest neighbors (0-indexed: 3,4,5)
        sigma = np.mean(knn_distances[:, 3:6], axis=1)  # (n,)
        sigma = np.maximum(sigma, 1e-10)

        # Generate triplets and weights
        if self.verbose:
            print("Generating triplets...")
        triplets, weights = self._generate_triplets(
            X, n, knn_indices, knn_distances, sigma, rng
        )
        if self.verbose:
            print(f"Generated {len(triplets)} triplets")

        # Initialize embedding via PCA (scaled by 0.01)
        X_mx = mx.array(X)
        mean = mx.mean(X_mx, axis=0)
        X_centered = X_mx - mean
        cov = (X_centered.T @ X_centered) / (n - 1)
        mx.eval(cov)
        eigvals, eigvecs = mx.linalg.eigh(cov, stream=mx.cpu)
        mx.eval(eigvals, eigvecs)
        init_proj = eigvecs[:, -self.n_components :][:, ::-1]
        Y = (X_centered @ init_proj) * 0.01
        mx.eval(Y)

        if epoch_callback is not None:
            epoch_callback(0, np.array(Y))

        # Convert to MLX for optimization
        triplets_mx = mx.array(triplets.astype(np.int32))
        weights_mx = mx.array(weights.astype(np.float32))

        # Delta-bar-delta optimization with momentum
        vel = mx.zeros_like(Y)
        gain = mx.ones_like(Y)
        mx.eval(vel, gain)

        for itr in range(1, self.n_iters + 1):
            grad = _compute_gradient(Y, triplets_mx, weights_mx, n)

            # Gain update: increase if gradient and velocity disagree
            same_sign = (vel * grad) > 0
            gain = mx.where(same_sign, gain * 0.8, gain + 0.2)
            gain = mx.maximum(gain, 0.01)

            momentum = 0.5 if itr <= 250 else 0.8
            vel = momentum * vel - self.lr * gain * grad
            Y = Y + vel

            if self.verbose and itr % 50 == 0:
                loss = _compute_loss(Y, triplets_mx, weights_mx)
                mx.eval(Y, vel, gain, loss)
                print(f"Iteration {itr}/{self.n_iters}, loss: {float(loss):.4f}")
            elif epoch_callback is not None:
                mx.eval(Y, vel, gain)
            elif itr % 10 == 0:
                mx.eval(Y, vel, gain)

            if epoch_callback is not None:
                epoch_callback(itr, np.array(Y))

        mx.eval(Y)
        self.embedding_ = np.array(Y)

        if self.verbose:
            print(f"TriMap done in {time.time() - t0:.1f}s")

        return self.embedding_

    def _generate_triplets(self, X, n, knn_indices, knn_distances, sigma, rng):
        """Generate triplets and compute weights.

        Returns:
            triplets: (T, 3) int32 array of (anchor, inlier, outlier) indices.
            weights: (T,) float32 array of transformed triplet weights.
        """
        n_inliers = min(self.n_inliers, knn_indices.shape[1])
        dist_sq = knn_distances**2

        # P[i][j_pos] = -d(i,j)^2 / (sigma_i * sigma_j) for KNN pairs
        sigma_ij = sigma[:, None] * sigma[knn_indices]  # (n, k)
        P_knn = -dist_sq / sigma_ij  # (n, k)

        # --- KNN triplets ---
        n_knn = n * n_inliers * self.n_outliers

        # Anchor: each point repeated n_inliers * n_outliers times
        anchors = np.repeat(np.arange(n, dtype=np.int32), n_inliers * self.n_outliers)

        # Inlier position in KNN list: 0..n_inliers-1, each repeated n_outliers
        inlier_pos = np.tile(
            np.repeat(np.arange(n_inliers), self.n_outliers), n
        )
        inliers = knn_indices[anchors, inlier_pos]

        # Outliers: random points
        outliers = rng.randint(0, n, size=n_knn).astype(np.int32)

        knn_triplets = np.stack([anchors, inliers, outliers], axis=1)

        # KNN weights: P[i,j] - P_outlier
        p_ij = P_knn[anchors, inlier_pos]

        # P_outlier = -d(i,k)^2 / (sigma_i * sigma_k)
        diff_ik = X[anchors] - X[outliers]
        d_ik_sq = np.sum(diff_ik**2, axis=1)
        p_outlier = -d_ik_sq / (sigma[anchors] * sigma[outliers])
        knn_weights = p_ij - p_outlier

        # --- Random triplets ---
        n_rand = n * self.n_random
        rand_anchors = np.repeat(np.arange(n, dtype=np.int32), self.n_random)
        rand_j = rng.randint(0, n, size=n_rand).astype(np.int32)
        rand_k = rng.randint(0, n, size=n_rand).astype(np.int32)

        random_triplets = np.stack([rand_anchors, rand_j, rand_k], axis=1)

        # Random weights: (P_ij - P_ik) * 0.1
        diff_ij_r = X[rand_anchors] - X[rand_j]
        d_ij_sq_r = np.sum(diff_ij_r**2, axis=1)
        p_ij_r = -d_ij_sq_r / (sigma[rand_anchors] * sigma[rand_j])

        diff_ik_r = X[rand_anchors] - X[rand_k]
        d_ik_sq_r = np.sum(diff_ik_r**2, axis=1)
        p_ik_r = -d_ik_sq_r / (sigma[rand_anchors] * sigma[rand_k])

        random_weights = (p_ij_r - p_ik_r) * 0.1

        # Combine
        triplets = np.concatenate([knn_triplets, random_triplets], axis=0)
        weights = np.concatenate([knn_weights, random_weights], axis=0)

        # Shift and transform: tempered_log(1 + w, t)
        weights -= weights.min()
        t = self.weight_temp
        weights = (np.power(1.0 + weights, 1.0 - t) - 1.0) / (1.0 - t)

        return triplets, weights.astype(np.float32)


def _compute_gradient(Y, triplets, weights, n):
    """Vectorized gradient computation over all triplets."""
    yi = Y[triplets[:, 0]]  # (T, d)
    yj = Y[triplets[:, 1]]  # (T, d)
    yk = Y[triplets[:, 2]]  # (T, d)

    diff_ij = yi - yj
    diff_ik = yi - yk

    d_ij = 1.0 + mx.sum(diff_ij**2, axis=1)  # (T,)
    d_ik = 1.0 + mx.sum(diff_ik**2, axis=1)  # (T,)

    w = weights / (d_ij + d_ik) ** 2  # (T,)
    w = w[:, None]  # (T, 1) for broadcasting

    grad_sim = diff_ij * (d_ik[:, None] * w)  # (T, d)
    grad_out = diff_ik * (d_ij[:, None] * w)  # (T, d)

    grad = mx.zeros_like(Y)
    grad = grad.at[triplets[:, 0]].add(grad_sim - grad_out)
    grad = grad.at[triplets[:, 1]].add(-grad_sim)
    grad = grad.at[triplets[:, 2]].add(grad_out)

    return grad


def _compute_loss(Y, triplets, weights):
    """Compute total triplet loss."""
    yi = Y[triplets[:, 0]]
    yj = Y[triplets[:, 1]]
    yk = Y[triplets[:, 2]]

    d_ij = 1.0 + mx.sum((yi - yj) ** 2, axis=1)
    d_ik = 1.0 + mx.sum((yi - yk) ** 2, axis=1)

    # loss = w * s_ik / (s_ij + s_ik) = w * d_ij / (d_ij + d_ik)
    loss = weights * d_ij / (d_ij + d_ik)
    return mx.sum(loss)
