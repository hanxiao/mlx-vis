"""CNE (Contrastive Neighbor Embedding) in pure MLX for Apple Silicon.

Reference: Damrich et al., "From t-SNE to UMAP with contrastive learning",
ICLR 2023. Unifies t-SNE and UMAP through contrastive losses on the
neighbor graph. InfoNCE loss gives t-SNE-like results, NEG loss gives
UMAP-like results.
"""

import time

import mlx.core as mx
import numpy as np

from mlx_vis._nndescent.nndescent import NNDescent


class CNE:
    """Contrastive Neighbor Embedding using MLX on Metal GPU.

    Parameters
    ----------
    n_components : int
        Dimension of the embedding (default 2).
    n_neighbors : int
        Number of nearest neighbors for graph construction (default 15).
    n_negatives : int
        Negative samples per positive edge (default 5).
        More negatives produce more t-SNE-like separation.
    loss : str
        Contrastive loss: "infonce" (t-SNE-like, default), "nce", or "neg" (UMAP-like).
    n_iter : int
        Number of optimization iterations (default 500).
    learning_rate : float
        Adam learning rate (default 1.0).
    batch_size : int or None
        Edge mini-batch size per iteration. None = all edges (default).
    pca_dim : int or None
        PCA target dimension for high-dimensional input (default 50).
    random_state : int or None
        Random seed for reproducibility.
    verbose : bool
        Print progress information.
    normalize : str or bool
        Input normalization before PCA/embedding (default False).
        False/None = no normalization, True/"standard" = z-score per feature,
        "minmax" = min-max scaling to [0,1] per feature.
    """

    def __init__(
        self,
        n_components=2,
        n_neighbors=15,
        n_negatives=5,
        loss="infonce",
        n_iter=500,
        learning_rate=1.0,
        batch_size=None,
        pca_dim=50,
        random_state=None,
        verbose=False,
        knn_method: str = "auto",
        normalize: str | bool = False,
    ):
        if loss not in ("infonce", "nce", "neg"):
            raise ValueError(f"Unknown loss: {loss!r}. Use 'infonce', 'nce', or 'neg'.")
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.n_negatives = n_negatives
        self.loss = loss
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.pca_dim = pca_dim
        self.random_state = random_state
        self.verbose = verbose
        self.knn_method = knn_method
        self.normalize = normalize
        self.embedding_ = None

    def fit_transform(self, X, epoch_callback=None):
        """Compute CNE embedding.

        Args:
            X: Input data array of shape (n_samples, n_features).
            epoch_callback: Optional callback(epoch, Y_numpy) for animation.

        Returns:
            np.ndarray of shape (n_samples, n_components).
        """
        t0 = time.time()

        if isinstance(X, mx.array):
            X = np.array(X)
        X = np.asarray(X, dtype=np.float32)

        from mlx_vis._normalize import normalize_input
        X = normalize_input(X, self.normalize)

        n, dim = X.shape

        if self.verbose:
            print(f"CNE: {n} points, {dim} dims, loss={self.loss}, "
                  f"k={self.n_neighbors}, m={self.n_negatives}")

        # PCA preprocessing
        if self.pca_dim is not None and dim > self.pca_dim:
            X = self._pca_reduce(X, n, dim)
            dim = X.shape[1]

        # k-NN
        if self.verbose:
            print("Computing k-NN...")
        t_knn = time.time()
        from mlx_vis._knn import compute_knn
        knn_indices, _ = compute_knn(
            X, self.n_neighbors, method=self.knn_method,
            verbose=self.verbose, random_state=self.random_state if self.random_state is not None else 42,
        )
        if self.verbose:
            print(f"k-NN done in {time.time() - t_knn:.1f}s")

        # Build symmetrized edge list
        edges = self._build_edges(knn_indices, n)
        n_edges = edges.shape[0]
        if self.verbose:
            print(f"Graph: {n_edges} edges (symmetrized)")

        # PCA initialization
        Y = self._pca_init(X, n)

        # Determine batch size
        batch_size = self.batch_size
        if batch_size is None or batch_size >= n_edges:
            batch_size = n_edges

        # Convert to MLX
        edges_mx = mx.array(edges)
        if self.random_state is not None:
            mx.random.seed(self.random_state)

        if epoch_callback is not None:
            epoch_callback(0, np.array(Y))

        # Optimize
        Y = self._optimize(Y, edges_mx, n, n_edges, batch_size, epoch_callback)

        mx.eval(Y)
        self.embedding_ = np.array(Y)

        if self.verbose:
            print(f"CNE done in {time.time() - t0:.1f}s")

        return self.embedding_

    def _pca_reduce(self, X, n, dim):
        """Reduce dimensionality via PCA."""
        if self.verbose:
            print(f"PCA: {dim} -> {self.pca_dim} dims...")
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
        if self.verbose:
            total_var = mx.sum(eigvals).item()
            retained_var = mx.sum(eigvals[-self.pca_dim:]).item()
            print(f"Variance retained: {retained_var / total_var * 100:.1f}%")
        return np.array(X_pca)

    def _pca_init(self, X, n):
        """Initialize embedding via PCA (scaled by 0.01)."""
        X_mx = mx.array(X)
        mean = mx.mean(X_mx, axis=0)
        X_centered = X_mx - mean
        cov = (X_centered.T @ X_centered) / (n - 1)
        mx.eval(cov)
        eigvals, eigvecs = mx.linalg.eigh(cov, stream=mx.cpu)
        mx.eval(eigvals, eigvecs)
        proj = eigvecs[:, -self.n_components:][:, ::-1]
        Y = (X_centered @ proj) * 0.01
        mx.eval(Y)
        if self.verbose:
            print("PCA initialization done")
        return Y

    @staticmethod
    def _build_edges(knn_indices, n):
        """Build symmetrized edge list from k-NN indices.

        For each directed edge i->j in the k-NN graph, add both (i,j) and (j,i),
        then deduplicate.

        Returns:
            np.ndarray of shape (E, 2), dtype int32.
        """
        k = knn_indices.shape[1]
        src = np.repeat(np.arange(n, dtype=np.int32), k)
        dst = knn_indices.ravel().astype(np.int32)

        # Stack both directions
        all_src = np.concatenate([src, dst])
        all_dst = np.concatenate([dst, src])

        # Deduplicate via canonical edge encoding
        keys = all_src.astype(np.int64) * n + all_dst.astype(np.int64)
        _, unique_idx = np.unique(keys, return_index=True)
        edges = np.stack([all_src[unique_idx], all_dst[unique_idx]], axis=1)

        # Remove self-loops
        mask = edges[:, 0] != edges[:, 1]
        return edges[mask]

    @staticmethod
    @mx.compile
    def _infonce_grad(Y, ei, ej, neg_indices):
        """Compiled InfoNCE gradient computation."""
        yi = Y[ei]
        yj = Y[ej]
        yk = Y[neg_indices]

        diff_pos = yi - yj
        d_pos = 1.0 + mx.sum(diff_pos * diff_pos, axis=-1)
        s_pos = 1.0 / d_pos

        diff_neg = yi[:, None, :] - yk
        d_neg = 1.0 + mx.sum(diff_neg * diff_neg, axis=-1)
        s_neg = 1.0 / d_neg

        S_neg = mx.sum(s_neg, axis=1)
        Z = s_pos + S_neg
        w_pos = 1.0 - s_pos / Z
        w_neg = s_neg / Z[:, None]

        g_attr = w_pos[:, None] * 2.0 * s_pos[:, None] * diff_pos
        g_rep = -mx.sum(
            w_neg[:, :, None] * 2.0 * s_neg[:, :, None] * diff_neg, axis=1,
        )

        grad_per_edge = g_attr + g_rep
        neg_grad = w_neg[:, :, None] * 2.0 * s_neg[:, :, None] * diff_neg

        grad = mx.zeros_like(Y)
        grad = grad.at[ei].add(grad_per_edge)
        grad = grad.at[ej].add(-g_attr)
        grad = grad.at[neg_indices.reshape(-1)].add(neg_grad.reshape(-1, grad.shape[1]))

        return grad

    @staticmethod
    @mx.compile
    def _nce_grad(Y, ei, ej, neg_indices, inv_m):
        """Compiled NCE gradient computation."""
        yi = Y[ei]
        yj = Y[ej]
        yk = Y[neg_indices]

        diff_pos = yi - yj
        d_pos = 1.0 + mx.sum(diff_pos * diff_pos, axis=-1)
        s_pos = 1.0 / d_pos

        diff_neg = yi[:, None, :] - yk
        d_neg = 1.0 + mx.sum(diff_neg * diff_neg, axis=-1)
        s_neg = 1.0 / d_neg

        g_attr = (2.0 * inv_m * s_pos / (s_pos + inv_m))[:, None] * diff_pos
        g_rep = mx.sum(
            -2.0 * (s_neg * s_neg / (s_neg + inv_m))[:, :, None] * diff_neg,
            axis=1,
        )

        neg_grad = 2.0 * (s_neg * s_neg / (s_neg + inv_m))[:, :, None] * diff_neg

        grad = mx.zeros_like(Y)
        grad = grad.at[ei].add(g_attr + g_rep)
        grad = grad.at[ej].add(-g_attr)
        grad = grad.at[neg_indices.reshape(-1)].add(neg_grad.reshape(-1, grad.shape[1]))

        return grad

    @staticmethod
    @mx.compile
    def _neg_grad(Y, ei, ej, neg_indices):
        """Compiled NEG gradient computation."""
        yi = Y[ei]
        yj = Y[ej]
        yk = Y[neg_indices]

        diff_pos = yi - yj
        d_pos = 1.0 + mx.sum(diff_pos * diff_pos, axis=-1)
        s_pos = 1.0 / d_pos

        diff_neg = yi[:, None, :] - yk
        d_neg = 1.0 + mx.sum(diff_neg * diff_neg, axis=-1)
        s_neg = 1.0 / d_neg

        g_attr = (2.0 * s_pos)[:, None] * diff_pos
        one_minus_s = mx.maximum(1.0 - s_neg, 1e-8)
        w_neg = s_neg * s_neg / one_minus_s
        g_rep = mx.sum(-2.0 * w_neg[:, :, None] * diff_neg, axis=1)

        neg_grad = 2.0 * w_neg[:, :, None] * diff_neg

        grad = mx.zeros_like(Y)
        grad = grad.at[ei].add(g_attr + g_rep)
        grad = grad.at[ej].add(-g_attr)
        grad = grad.at[neg_indices.reshape(-1)].add(neg_grad.reshape(-1, grad.shape[1]))

        return grad

    def _optimize(self, Y, edges_mx, n, n_edges, batch_size, epoch_callback):
        """Adam optimization with contrastive loss."""
        m = mx.zeros_like(Y)
        v = mx.zeros_like(Y)
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        lr = self.learning_rate
        n_neg = self.n_negatives

        # Select compiled gradient function
        if self.loss == "infonce":
            grad_fn = self._infonce_grad
        elif self.loss == "nce":
            inv_m_mx = mx.array(1.0 / n_neg)
            grad_fn = None  # handled separately due to extra param
        else:
            grad_fn = self._neg_grad

        if self.verbose:
            print("Starting optimization...")

        for itr in range(1, self.n_iter + 1):
            # Mini-batch edge sampling
            if batch_size < n_edges:
                idx = mx.random.randint(0, n_edges, (batch_size,))
                batch_edges = edges_mx[idx]
            else:
                batch_edges = edges_mx

            ei = batch_edges[:, 0]
            ej = batch_edges[:, 1]

            # Sample negatives uniformly
            neg_indices = mx.random.randint(0, n, (batch_edges.shape[0], n_neg))

            # Compiled gradient computation
            if self.loss == "nce":
                grad = self._nce_grad(Y, ei, ej, neg_indices, inv_m_mx)
            else:
                grad = grad_fn(Y, ei, ej, neg_indices)

            # Scale gradient for mini-batch
            if batch_size < n_edges:
                grad = grad * (n_edges / batch_size)

            # Adam update
            m = beta1 * m + (1.0 - beta1) * grad
            v = beta2 * v + (1.0 - beta2) * (grad * grad)
            m_hat = m / (1.0 - beta1 ** itr)
            v_hat = v / (1.0 - beta2 ** itr)
            Y = Y - lr * m_hat / (mx.sqrt(v_hat) + eps)

            if epoch_callback is not None:
                mx.eval(Y, m, v)
                epoch_callback(itr, np.array(Y))
            elif itr % 10 == 0 or itr == self.n_iter:
                mx.eval(Y, m, v)

            if self.verbose and itr % 50 == 0:
                print(f"Iteration {itr}/{self.n_iter}")

        return Y
