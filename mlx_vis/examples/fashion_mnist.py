"""Fashion-MNIST embedding visualization with UMAP, t-SNE, PaCMAP.

Usage:
    python -m mlx_vis.examples.fashion_mnist --method umap --theme dark
    python -m mlx_vis.examples.fashion_mnist --method all --fps 120
"""

import argparse
import os
import time

import numpy as np


LABELS = [
    "T-shirt", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Boot",
]
SEED = 42
VIZ_N = 15000


def load_fashion_mnist():
    from sklearn.datasets import fetch_openml
    print("Loading Fashion-MNIST 70K...")
    fm = fetch_openml("Fashion-MNIST", version=1, as_frame=False, parser="auto")
    X = fm.data.astype(np.float32) / 255.0
    y = fm.target.astype(np.int32)
    return X, y


def run_umap(X, y, viz_idx, args):
    from mlx_vis import UMAP
    from mlx_vis.plot import scatter, animate

    snaps, snap_times = [], []
    t0 = time.time()

    def cb(epoch, Y_np):
        snaps.append(Y_np[viz_idx])
        snap_times.append(time.time() - t0)

    n_epochs = 500
    print(f"Running UMAP ({n_epochs} epochs)...")
    Y = UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
             random_state=SEED, n_epochs=n_epochs, verbose=True).fit_transform(X, epoch_callback=cb)
    t_total = time.time() - t0
    print(f"UMAP done: {t_total:.2f}s, {len(snaps)} snapshots")

    y_viz = y[viz_idx]
    outdir = args.outdir

    # static plot
    fig = scatter(Y[viz_idx], labels=y_viz, theme=args.theme, title="umap-mlx  Fashion-MNIST 70K",
                  save=os.path.join(outdir, "umap_scatter.png"))
    print(f"Saved {outdir}/umap_scatter.png")

    # animation
    total_f = animate(snaps, labels=y_viz, timestamps=snap_times,
                      method_name="umap-mlx", dataset_name="Fashion-MNIST 70K x 784",
                      fps=args.fps, theme=args.theme,
                      save=os.path.join(outdir, "umap_animation.mp4"))
    fpath = os.path.join(outdir, "umap_animation.mp4")
    mb = os.path.getsize(fpath) / 1024 / 1024
    print(f"Saved {fpath} ({total_f} frames, {mb:.1f} MB)")


def run_tsne(X, y, viz_idx, args):
    from mlx_vis import TSNE
    from mlx_vis.plot import scatter, animate

    snaps, snap_times = [], []
    t0 = time.time()

    def cb(epoch, Y_np):
        snaps.append(Y_np[viz_idx].copy())
        snap_times.append(time.time() - t0)

    n_iter = 750
    print(f"Running t-SNE ({n_iter} iterations)...")
    Y = TSNE(n_components=2, perplexity=30, n_iter=n_iter,
             random_state=SEED, verbose=250).fit_transform(X, epoch_callback=cb)
    t_total = time.time() - t0
    print(f"t-SNE done: {t_total:.2f}s, {len(snaps)} snapshots")

    y_viz = y[viz_idx]
    outdir = args.outdir

    fig = scatter(Y[viz_idx], labels=y_viz, theme=args.theme, title="tsne-mlx  Fashion-MNIST 70K",
                  save=os.path.join(outdir, "tsne_scatter.png"))
    print(f"Saved {outdir}/tsne_scatter.png")

    total_f = animate(snaps, labels=y_viz, timestamps=snap_times,
                      method_name="tsne-mlx", dataset_name="Fashion-MNIST 70K x 784",
                      fps=args.fps, theme=args.theme,
                      save=os.path.join(outdir, "tsne_animation.mp4"))
    fpath = os.path.join(outdir, "tsne_animation.mp4")
    mb = os.path.getsize(fpath) / 1024 / 1024
    print(f"Saved {fpath} ({total_f} frames, {mb:.1f} MB)")


def run_pacmap(X, y, viz_idx, args):
    from mlx_vis import PaCMAP
    from mlx_vis.plot import scatter, animate

    snaps, snap_times = [], []
    t0 = time.time()

    def cb(epoch, Y_np):
        snaps.append(Y_np[viz_idx])
        snap_times.append(time.time() - t0)

    print("Running PaCMAP (450 iterations)...")
    Y = PaCMAP(n_components=2, n_neighbors=10, random_state=SEED,
               verbose=True).fit_transform(X, epoch_callback=cb)
    t_total = time.time() - t0
    print(f"PaCMAP done: {t_total:.2f}s, {len(snaps)} snapshots")

    y_viz = y[viz_idx]
    outdir = args.outdir

    fig = scatter(Y[viz_idx], labels=y_viz, theme=args.theme, title="pacmap-mlx  Fashion-MNIST 70K",
                  save=os.path.join(outdir, "pacmap_scatter.png"))
    print(f"Saved {outdir}/pacmap_scatter.png")

    total_f = animate(snaps, labels=y_viz, timestamps=snap_times,
                      method_name="pacmap-mlx", dataset_name="Fashion-MNIST 70K x 784",
                      fps=args.fps, theme=args.theme,
                      save=os.path.join(outdir, "pacmap_animation.mp4"))
    fpath = os.path.join(outdir, "pacmap_animation.mp4")
    mb = os.path.getsize(fpath) / 1024 / 1024
    print(f"Saved {fpath} ({total_f} frames, {mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Fashion-MNIST embedding visualization")
    parser.add_argument("--method", choices=["umap", "tsne", "pacmap", "all"], default="all")
    parser.add_argument("--theme", choices=["dark", "light"], default="dark")
    parser.add_argument("--fps", type=int, default=120)
    parser.add_argument("--outdir", default=".")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    X, y = load_fashion_mnist()
    np.random.seed(SEED)
    viz_idx = np.random.choice(len(X), VIZ_N, replace=False)

    methods = {
        "umap": run_umap,
        "tsne": run_tsne,
        "pacmap": run_pacmap,
    }

    if args.method == "all":
        for name, fn in methods.items():
            fn(X, y, viz_idx, args)
    else:
        methods[args.method](X, y, viz_idx, args)


if __name__ == "__main__":
    main()
