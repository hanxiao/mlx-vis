"""MMAE (Manifold-Matching Autoencoder) in pure MLX.

Implements the Manifold-Matching regularization from:
    Cheret et al., "Manifold-Matching Autoencoders", arXiv:2603.16568, 2026.

The key idea: align pairwise Euclidean distances in the latent space to those
in a reference space (raw input or PCA-reduced input) via MSE on distance
matrices computed per minibatch. This preserves global geometry and, by the
stability theorem, topology.
"""

import time
from functools import partial

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from mlx_vis.pca import PCA


class _MLP(nn.Module):
    """MLP with ReLU activations and layer normalization."""

    def __init__(self, dims):
        super().__init__()
        self.linears = [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        self.norms = [nn.LayerNorm(dims[i + 1]) for i in range(len(dims) - 2)]

    def __call__(self, x):
        for i, linear in enumerate(self.linears[:-1]):
            x = linear(x)
            x = self.norms[i](x)
            x = nn.relu(x)
        x = self.linears[-1](x)
        return x


class _Autoencoder(nn.Module):
    """Symmetric autoencoder: encoder and decoder are mirror architectures."""

    def __init__(self, input_dim, hidden_dims, latent_dim):
        super().__init__()
        enc_dims = [input_dim] + list(hidden_dims) + [latent_dim]
        dec_dims = [latent_dim] + list(reversed(hidden_dims)) + [input_dim]
        self.encoder = _MLP(enc_dims)
        self.decoder = _MLP(dec_dims)

    def __call__(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return z, recon

    def encode(self, x):
        return self.encoder(x)


def _pairwise_euclidean(Z):
    """Pairwise Euclidean distance matrix for a batch of vectors."""
    sq_norms = mx.sum(Z * Z, axis=1)
    dists_sq = sq_norms[:, None] + sq_norms[None, :] - 2.0 * (Z @ Z.T)
    return mx.sqrt(mx.maximum(dists_sq, 0.0) + 1e-12)


class MMAE:
    """Manifold-Matching Autoencoder for dimensionality reduction.

    Trains an autoencoder with a regularization term that aligns pairwise
    distances in the latent space to those in a reference space (raw input
    or a PCA projection). This preserves global metric structure without
    requiring persistent homology computation.

    Parameters
    ----------
    n_components : int, default=2
        Dimensionality of the latent (output) embedding.
    n_epochs : int, default=100
        Number of training epochs.
    batch_size : int, default=512
        Minibatch size for training.
    lr : float, default=1e-3
        Learning rate for AdamW optimizer.
    weight_decay : float, default=1e-4
        Weight decay (L2 regularization).
    lambda_mm : float, default=1.0
        Weight of the manifold-matching regularization relative to
        reconstruction loss.
    hidden_dims : tuple of int, default=(512, 256, 128)
        Hidden layer sizes for encoder (decoder mirrors this).
    pca_dim : int or None, default=None
        If set and input dim > pca_dim, PCA-reduce the reference space
        to this many dimensions. Helps with curse of dimensionality on
        high-dimensional data. When None, uses raw input distances.
    reference : numpy.ndarray or None, default=None
        External reference embedding of shape (n_samples, k) to use as
        the target space for MM-reg. When provided, overrides pca_dim
        and raw input as the reference. This allows MMAE to "copy" any
        embedding (UMAP, t-SNE, PCA, etc.) while gaining an encoder
        for out-of-sample extension::

            # Give UMAP out-of-sample extension via MMAE
            Y_umap = UMAP().fit_transform(X)
            Y_mmae = MMAE(reference=Y_umap).fit_transform(X)
            # Now the encoder can embed new data with UMAP-like layout

        Can also be passed at fit_transform() time, which overrides
        the init-time reference.
    random_state : int or None, default=None
        Seed for reproducibility.
    verbose : bool, default=False
        Print progress during training.
    knn_method : str, default="auto"
        Unused - kept for API compatibility with other algorithms.
    """

    def __init__(
        self,
        n_components=2,
        n_epochs=100,
        batch_size=512,
        lr=1e-3,
        weight_decay=1e-4,
        lambda_mm=1.0,
        hidden_dims=(512, 256, 128),
        pca_dim=None,
        random_state=None,
        verbose=False,
        reference=None,
        knn_method="auto",
    ):
        self.n_components = n_components
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.lambda_mm = lambda_mm
        self.hidden_dims = hidden_dims
        self.pca_dim = pca_dim
        self.random_state = random_state
        self.verbose = verbose
        self.reference = reference
        self.knn_method = knn_method
        self.embedding_ = None

    def _log(self, msg):
        if self.verbose:
            print(msg)

    def _build_reference(self, X_np, n, D, reference=None):
        """Build the reference embedding for MM-reg."""
        if reference is not None:
            ref = np.asarray(reference, dtype=np.float32)
            if ref.shape[0] != n:
                raise ValueError(
                    f"reference has {ref.shape[0]} samples, expected {n}"
                )
            k = ref.shape[1]
            self._log(f"Using external reference ({k}D)")
            return mx.array(ref)
        pca_dim = self.pca_dim
        if pca_dim is not None and D > pca_dim:
            E_np = PCA(n_components=pca_dim).fit_transform(X_np)
            self._log(f"PCA reference: {D}D -> {pca_dim}D")
            return mx.array(E_np)
        else:
            self._log(f"Using raw input ({D}D) as reference space")
            return mx.array(X_np)

    def _encode_all(self, model, X, batch_size):
        """Encode all data points, chunked to manage memory."""
        n = X.shape[0]
        parts = []
        for i in range(0, n, batch_size):
            z = model.encode(X[i : i + batch_size])
            parts.append(z)
            if len(parts) % 4 == 0:
                mx.eval(*parts[-4:])
        if parts:
            mx.eval(*parts)
        return mx.concatenate(parts, axis=0)

    def fit_transform(self, X, reference=None, epoch_callback=None):
        """Fit the autoencoder and return the latent embedding.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
        reference : numpy.ndarray or None, optional
            External reference embedding of shape (n_samples, k).
            Overrides the init-time reference if provided.
        epoch_callback : callable, optional
            Called as epoch_callback(epoch, Y_numpy) after initialization
            (epoch=0) and after each training epoch.

        Returns
        -------
        Y : numpy.ndarray of shape (n_samples, n_components)
        """
        start_time = time.time()
        X_np = np.array(X, dtype=np.float32)
        n, D = X_np.shape

        rng = np.random.default_rng(self.random_state)
        if self.random_state is not None:
            mx.random.seed(self.random_state)

        # Reference space for MM-reg (external, PCA-reduced, or raw)
        ref = reference if reference is not None else self.reference
        E_mx = self._build_reference(X_np, n, D, reference=ref)

        # Normalize input for stable training
        X_mx = mx.array(X_np)
        x_mean = mx.mean(X_mx, axis=0)
        X_train = X_mx - x_mean
        x_scale = mx.sqrt(mx.mean(X_train * X_train)) + 1e-8
        X_train = X_train / x_scale
        mx.eval(X_train)

        # Build model
        model = _Autoencoder(D, self.hidden_dims, self.n_components)
        mx.eval(model.parameters())

        optimizer = optim.AdamW(
            learning_rate=self.lr, weight_decay=self.weight_decay
        )

        lambda_mm = self.lambda_mm
        batch_size = min(self.batch_size, n)

        def loss_fn(model, x_batch, e_batch):
            z, recon = model(x_batch)
            recon_loss = mx.mean((x_batch - recon) ** 2)
            D_Z = _pairwise_euclidean(z)
            D_E = _pairwise_euclidean(e_batch)
            mm_loss = mx.mean((D_Z - D_E) ** 2)
            return recon_loss + lambda_mm * mm_loss

        loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

        state = [model.state, optimizer.state]

        @partial(mx.compile, inputs=state, outputs=state)
        def step(x_batch, e_batch):
            loss, grads = loss_and_grad_fn(model, x_batch, e_batch)
            optimizer.update(model, grads)
            return loss

        self._log(
            f"Architecture: {D} -> {list(self.hidden_dims)} -> {self.n_components} "
            f"-> {list(reversed(self.hidden_dims))} -> {D}"
        )
        self._log(f"Training: n={n}, batch_size={batch_size}, n_epochs={self.n_epochs}")

        # Epoch 0 callback
        if epoch_callback is not None:
            model.eval()
            Y_init = self._encode_all(model, X_train, batch_size)
            mx.eval(Y_init)
            epoch_callback(0, np.array(Y_init))

        model.train()
        for epoch in range(self.n_epochs):
            indices = rng.permutation(n)
            epoch_loss = 0.0
            n_batches = 0

            for b in range(0, n, batch_size):
                batch_idx = indices[b : b + batch_size]
                if len(batch_idx) < 2:
                    continue
                batch_idx_mx = mx.array(batch_idx)
                x_batch = X_train[batch_idx_mx]
                e_batch = E_mx[batch_idx_mx]

                loss = step(x_batch, e_batch)
                mx.eval(loss)
                epoch_loss += float(loss)
                n_batches += 1

            if epoch_callback is not None:
                Y_cur = self._encode_all(model, X_train, batch_size)
                mx.eval(Y_cur)
                epoch_callback(epoch + 1, np.array(Y_cur))

            if self.verbose and (epoch + 1) % 10 == 0:
                avg_loss = epoch_loss / max(n_batches, 1)
                self._log(f"Epoch {epoch + 1}/{self.n_epochs}, loss={avg_loss:.6f}")

        # Final embedding
        model.eval()
        Y = self._encode_all(model, X_train, batch_size)
        mx.eval(Y)

        elapsed = time.time() - start_time
        self._log(f"Elapsed time: {elapsed:.2f}s")

        self.embedding_ = np.array(Y)
        return self.embedding_
