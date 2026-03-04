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


def scatter(Y, labels=None, theme="dark", colors=None, point_size=2,
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
    import mlx.core as mx
    from .render import render_frame
    import subprocess

    Y = np.asarray(Y)
    c = _resolve_colors(labels, colors, len(Y), theme)
    c = np.array(c, dtype=np.float32)
    c[:, 3] = alpha

    bg_val = 0.0 if theme == "dark" else 1.0
    bg_color = mx.array([bg_val, bg_val, bg_val, 1.0], dtype=mx.float32)

    xlim, ylim = _get_square_lims(Y)

    frame = render_frame(
        mx.array(Y, dtype=mx.float32),
        mx.array(c, dtype=mx.float32),
        width, height, xlim, ylim,
        point_radius=int(point_size),
        bg_color=bg_color,
    )
    mx.eval(frame)
    pixels = np.array(frame)

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
                save="animation.mp4", width=1000, height=1000, bitrate=8000):
    """GPU-accelerated animation using MLX Metal + ffmpeg pipe.

    Same interface as animate() but renders on GPU without matplotlib.
    Uses h264_videotoolbox for hardware video encoding on Mac.

    Returns total frames rendered.
    """
    import subprocess
    import mlx.core as mx
    from mlx_vis.render import render_frame

    snapshots_np = [np.asarray(s) for s in snapshots]
    n_snap = len(snapshots_np)
    n_points = len(snapshots_np[0])
    n_epochs = n_snap - 1

    # resolve colors
    c = _resolve_colors(labels, colors, n_points, theme)
    c = np.array(c, dtype=np.float32)
    c[:, 3] = alpha
    colors_mx = mx.array(c)

    # compute axis limits from final snapshot
    xlim, ylim = _get_square_lims(snapshots_np[-1])

    # convert all snapshots to mx.array upfront
    snapshots_mx = [mx.array(s.astype(np.float32)) for s in snapshots_np]

    # background color
    if theme == "dark":
        bg = mx.array([0.0, 0.0, 0.0, 1.0], dtype=mx.float32)
    else:
        bg = mx.array([1.0, 1.0, 1.0, 1.0], dtype=mx.float32)

    # point radius from point_size (sqrt scaling, minimum 1px)
    point_radius = max(1, int(round(np.sqrt(point_size) * 1.2)))

    # title text overlay via ffmpeg drawtext
    fg_hex = "ffffff" if theme == "dark" else "000000"

    # frame counts
    init_f = int(init_hold * fps)
    hold_f = int(end_hold * fps)
    total_f = init_f + n_snap + hold_f

    # build title text for each unique snapshot index
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
            if idx >= n_snap - 1:
                parts.append(f"t={t:.1f}s")
            else:
                parts.append(f"t={t:.2f}s")
        return "  ".join(parts)

    # build ffmpeg command with drawtext filter for title
    vf_parts = []
    # drawtext for title
    title_text = _title_text(0)  # will update per-frame via metadata, but ffmpeg drawtext is static
    # Instead: we burn text into frames by sending via ffmpeg per-segment, or skip text for simplicity
    # For v1: use a single ffmpeg with no text overlay (title can be added post-hoc)

    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-pix_fmt", "rgba",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "pipe:",
        "-c:v", "h264_videotoolbox",
        "-b:v", f"{bitrate}k",
        "-pix_fmt", "yuv420p",
        save,
    ]

    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    for frame_i in range(total_f):
        # determine snapshot index
        if frame_i < init_f:
            idx = 0
        elif frame_i < init_f + n_snap:
            idx = frame_i - init_f
        else:
            idx = n_snap - 1

        # render
        frame = render_frame(snapshots_mx[idx], colors_mx, width, height,
                             xlim, ylim, point_radius=point_radius,
                             bg_color=bg)
        mx.eval(frame)
        proc.stdin.write(np.array(frame, copy=False).tobytes())

        if (frame_i + 1) % 100 == 0 or frame_i == total_f - 1:
            print(f"  frame {frame_i + 1}/{total_f}")

    proc.stdin.close()
    proc.wait()

    return total_f
