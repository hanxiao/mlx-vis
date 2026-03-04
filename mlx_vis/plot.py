"""Visualization module for mlx-vis embeddings."""

import numpy as np


# Dark theme: bright saturated colors that pop on black
_PALETTE_DARK = np.array([
    [0.376, 0.631, 0.961, 1.0],  # bright blue
    [0.961, 0.396, 0.322, 1.0],  # vermilion
    [0.353, 0.839, 0.502, 1.0],  # emerald
    [0.788, 0.490, 0.910, 1.0],  # lavender
    [1.000, 0.757, 0.224, 1.0],  # gold
    [0.259, 0.827, 0.808, 1.0],  # cyan
    [0.969, 0.518, 0.643, 1.0],  # pink
    [0.886, 0.671, 0.380, 1.0],  # sand
    [0.490, 0.847, 0.690, 1.0],  # mint
    [0.714, 0.714, 0.443, 1.0],  # khaki
], dtype=np.float32)

# Light theme: deeper muted tones that read well on white
_PALETTE_LIGHT = np.array([
    [0.216, 0.380, 0.655, 1.0],  # navy blue
    [0.808, 0.259, 0.204, 1.0],  # brick red
    [0.173, 0.569, 0.318, 1.0],  # forest green
    [0.545, 0.306, 0.682, 1.0],  # grape
    [0.816, 0.557, 0.082, 1.0],  # dark gold
    [0.110, 0.569, 0.569, 1.0],  # dark teal
    [0.776, 0.310, 0.416, 1.0],  # berry
    [0.624, 0.443, 0.216, 1.0],  # brown
    [0.263, 0.557, 0.408, 1.0],  # sage
    [0.482, 0.451, 0.290, 1.0],  # olive
], dtype=np.float32)


def _get_square_lims(emb, margin=0.1):
    cx = (emb[:, 0].min() + emb[:, 0].max()) / 2
    cy = (emb[:, 1].min() + emb[:, 1].max()) / 2
    span = max(emb[:, 0].max() - emb[:, 0].min(),
               emb[:, 1].max() - emb[:, 1].min())
    hs = span / 2 * (1 + margin)
    return (cx - hs, cx + hs), (cy - hs, cy + hs)


def _resolve_colors(labels, colors, n_points, theme):
    """Resolve point colors from labels, explicit colors, or defaults."""
    import matplotlib.pyplot as plt

    if colors is not None:
        if isinstance(colors, str):
            cmap = plt.get_cmap(colors)
            if labels is not None:
                norm = labels.astype(float)
                norm = (norm - norm.min()) / max(norm.max() - norm.min(), 1)
                return cmap(norm)
            return cmap(np.linspace(0, 1, n_points))
        return np.asarray(colors)

    if labels is not None:
        n_classes = int(labels.max() - labels.min() + 1)
        if n_classes <= 10:
            palette = _PALETTE_DARK if theme == "dark" else _PALETTE_LIGHT
            norm = (labels - labels.min()).astype(int)
            return palette[norm % 10]
        cmap = plt.get_cmap("Spectral")
        norm = labels.astype(float)
        norm = (norm - norm.min()) / max(norm.max() - norm.min(), 1)
        return cmap(norm)

    if theme == "dark":
        return np.full((n_points, 4), [0.4, 0.7, 0.9, 1.0], dtype=np.float32)
    return np.full((n_points, 4), [0.2, 0.4, 0.7, 1.0], dtype=np.float32)


def scatter(Y, labels=None, theme="dark", colors=None, point_size=1.5,
            alpha=0.6, title=None, figsize=(10, 10), save=None, dpi=150):
    """Static scatter plot using matplotlib (CPU fallback)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Y = np.asarray(Y)
    c = _resolve_colors(labels, colors, len(Y), theme)
    c = np.array(c, dtype=np.float64)
    c[:, 3] = alpha

    bg = "black" if theme == "dark" else "white"
    fg = "white" if theme == "dark" else "black"

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    fig.set_facecolor(bg)
    ax.set_facecolor(bg)

    xlim, ylim = _get_square_lims(Y)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.scatter(Y[:, 0], Y[:, 1], s=point_size, c=c, edgecolors="none")

    if title:
        ax.set_title(title, color=fg, fontsize=14, pad=10, fontfamily="monospace")

    fig.tight_layout(pad=0.5)

    if save:
        fig.savefig(save, dpi=dpi, facecolor=bg, bbox_inches="tight")

    return fig


def scatter_gpu(Y, labels=None, theme="dark", colors=None, point_size=2,
                alpha=0.6, title=None, save=None, width=1000, height=1000):
    """Static scatter plot rendered on Metal GPU via MLX.

    Args:
        Y: (n, 2) array of embedding coordinates.
        labels: optional int array for coloring by class.
        theme: "dark" (black bg) or "light" (white bg).
        colors: custom RGBA array or matplotlib colormap name.
        point_size: point radius in pixels.
        alpha: point transparency.
        title: optional title string (ignored in GPU path).
        save: path to save PNG.
        width: image width in pixels.
        height: image height in pixels.
    """
    from .render import render_frame
    import subprocess

    Y = np.asarray(Y)
    c = _resolve_colors(labels, colors, len(Y), theme)
    c = np.array(c, dtype=np.float32)
    c[:, 3] = alpha

    bg_val = 0.0 if theme == "dark" else 1.0
    bg_color = np.array([bg_val, bg_val, bg_val, 1.0], dtype=np.float32)

    xlim, ylim = _get_square_lims(Y)

    pixels = render_frame(
        Y, c, width, height, xlim, ylim,
        point_radius=point_size,
        bg_color=bg_color,
    )

    if save:
        # write raw RGBA to PNG via ffmpeg (no PIL/matplotlib dependency)
        cmd = [
            "ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgba",
            "-s", f"{width}x{height}", "-i", "pipe:",
            "-frames:v", "1", save,
        ]
        proc = subprocess.run(cmd, input=pixels.tobytes(),
                              capture_output=True, timeout=10)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg error: {proc.stderr.decode()}")

    return pixels


def animate(snapshots, labels=None, timestamps=None, method_name="",
            dataset_name="", fps=120, theme="dark", colors=None,
            point_size=1.5, alpha=0.6, init_hold=0.5, end_hold=2.0,
            save="animation.mp4", bitrate=8000, figsize=(10, 10)):
    """Render animation using matplotlib (CPU fallback)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.animation as mpl_anim

    snapshots = [np.asarray(s) for s in snapshots]
    n_snap = len(snapshots)
    n_points = len(snapshots[0])
    n_epochs = n_snap - 1

    c = _resolve_colors(labels, colors, n_points, theme)
    c = np.array(c, dtype=np.float64)
    c[:, 3] = alpha

    bg = "black" if theme == "dark" else "white"
    fg = "white" if theme == "dark" else "black"

    xlim, ylim = _get_square_lims(snapshots[-1])

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    fig.set_facecolor(bg)
    ax.set_facecolor(bg)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.axis("off")

    sc = ax.scatter([], [], s=point_size)
    ttl = ax.set_title("", color=fg, fontsize=14, pad=10, fontfamily="monospace")

    init_f = int(init_hold * fps)
    hold_f = int(end_hold * fps)
    total_f = init_f + n_snap + hold_f

    def _title_text(idx):
        parts = []
        if method_name:
            parts.append(method_name)
        if dataset_name:
            parts.append(dataset_name)
        if idx == 0:
            parts.append("init")
        elif idx < n_snap - 1:
            parts.append(f"epoch {idx}/{n_epochs}")
        else:
            parts.append("done")
        if timestamps:
            t = timestamps[min(idx, len(timestamps) - 1)]
            parts.append(f"t={t:.1f}s" if idx >= n_snap - 1 else f"t={t:.2f}s")
        return "  ".join(parts)

    def update(frame):
        if frame < init_f:
            idx = 0
        elif frame < init_f + n_snap:
            idx = frame - init_f
        else:
            idx = n_snap - 1
        sc.set_offsets(snapshots[idx])
        sc.set_color(c)
        ttl.set_text(_title_text(idx))
        return sc, ttl

    anim = mpl_anim.FuncAnimation(fig, update, frames=total_f, blit=True,
                                   interval=1000 // fps)
    writer = mpl_anim.FFMpegWriter(fps=fps, bitrate=bitrate,
                                    extra_args=["-pix_fmt", "yuv420p"])
    anim.save(save, writer=writer)
    plt.close(fig)
    return total_f


def animate_gpu(snapshots, labels=None, timestamps=None, method_name="",
                dataset_name="", fps=120, theme="dark", colors=None,
                point_size=1.5, alpha=0.6, init_hold=0.5, end_hold=2.0,
                save="animation.mp4", width=1000, height=1000, bitrate=8000):
    """GPU-accelerated animation using MLX Metal + ffmpeg pipe.

    Renders on Metal GPU without matplotlib.
    Uses h264_videotoolbox for hardware video encoding on Mac.

    Returns total frames rendered.
    """
    import subprocess
    from mlx_vis.render import render_frame

    snapshots_np = [np.asarray(s) for s in snapshots]
    n_snap = len(snapshots_np)
    n_points = len(snapshots_np[0])
    n_epochs = n_snap - 1

    c = _resolve_colors(labels, colors, n_points, theme)
    c = np.array(c, dtype=np.float32)
    c[:, 3] = alpha

    xlim, ylim = _get_square_lims(snapshots_np[-1])

    bg_val = 0.0 if theme == "dark" else 1.0
    bg = np.array([bg_val, bg_val, bg_val, 1.0], dtype=np.float32)

    point_radius = max(1.0, point_size)

    init_f = int(init_hold * fps)
    hold_f = int(end_hold * fps)
    total_f = init_f + n_snap + hold_f

    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgba",
        "-s", f"{width}x{height}", "-r", str(fps),
        "-i", "pipe:",
        "-c:v", "h264_videotoolbox",
        "-b:v", f"{bitrate}k",
        "-pix_fmt", "yuv420p",
        save,
    ]

    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    for frame_i in range(total_f):
        if frame_i < init_f:
            idx = 0
        elif frame_i < init_f + n_snap:
            idx = frame_i - init_f
        else:
            idx = n_snap - 1

        frame = render_frame(snapshots_np[idx], c, width, height,
                             xlim, ylim, point_radius=point_radius,
                             bg_color=bg)
        proc.stdin.write(frame.tobytes())

        if (frame_i + 1) % 100 == 0 or frame_i == total_f - 1:
            print(f"  frame {frame_i + 1}/{total_f}")

    proc.stdin.close()
    proc.wait()

    return total_f
