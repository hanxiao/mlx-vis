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
                alpha=0.4, title=None, save=None, width=1000, height=1000):
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
                point_size=2, alpha=1.0, init_hold=0.5, end_hold=2.0,
                save="animation.mp4", width=1000, height=1000, bitrate=8000):
    """GPU-accelerated animation using MLX Metal + ffmpeg pipe.

    Renders on Metal GPU without matplotlib.
    Uses h264_videotoolbox for hardware video encoding on Mac.

    Returns total frames rendered.
    """
    import subprocess
    import mlx.core as mx
    from mlx_vis.render import _render_frame_mlx, _circle_template, _TEMPLATE_CACHE

    snapshots_np = [np.asarray(s, dtype=np.float32) for s in snapshots]
    n_snap = len(snapshots_np)
    n_points = len(snapshots_np[0])

    c = _resolve_colors(labels, colors, n_points, theme)
    c = np.array(c, dtype=np.float32)
    c[:, 3] = alpha

    # Use global limits across ALL frames (not just the last one).
    # This is critical for FlowMatch where source and target have different ranges.
    all_points = np.concatenate(snapshots_np, axis=0)
    xlim, ylim = _get_square_lims(all_points)

    bg_val = 0.0 if theme == "dark" else 1.0

    point_radius = max(1.0, point_size)

    init_f = int(init_hold * fps)
    hold_f = int(end_hold * fps)
    total_f = init_f + n_snap + hold_f

    # pre-convert constant inputs to mx.array once (avoid per-frame conversion)
    colors_mx = mx.array(c)
    bg_mx = mx.array([bg_val, bg_val, bg_val, 1.0], dtype=mx.float32)
    xmin, xmax = float(xlim[0]), float(xlim[1])
    ymin, ymax = float(ylim[0]), float(ylim[1])

    rk = round(point_radius * 10)
    if rk not in _TEMPLATE_CACHE:
        _TEMPLATE_CACHE[rk] = _circle_template(point_radius)
    offsets, weights = _TEMPLATE_CACHE[rk]

    def _render(idx):
        Y_mx = mx.array(snapshots_np[idx])
        return _render_frame_mlx(Y_mx, colors_mx, offsets, weights,
                                 width, height, xmin, xmax, ymin, ymax, bg_mx)

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

    # init hold: render once, reuse bytes for all hold frames
    init_buf = _render(0)
    mx.eval(init_buf)
    init_bytes = np.array(init_buf).tobytes()
    for _ in range(init_f):
        proc.stdin.write(init_bytes)

    # pipeline: render frame N+1 on GPU while piping frame N to ffmpeg
    # first snapshot frame already rendered above; write it
    proc.stdin.write(init_bytes)

    if n_snap > 1:
        frame = _render(1)
        mx.async_eval(frame)
        for i in range(2, n_snap):
            next_frame = _render(i)
            mx.async_eval(next_frame)
            mx.eval(frame)
            proc.stdin.write(np.array(frame).tobytes())
            frame = next_frame
            done = init_f + i
            if done % 100 == 0:
                print(f"  frame {done}/{total_f}")

        # last changing frame
        mx.eval(frame)
        last_bytes = np.array(frame).tobytes()
        proc.stdin.write(last_bytes)
    else:
        last_bytes = init_bytes

    # end hold: reuse last frame bytes
    for _ in range(hold_f):
        proc.stdin.write(last_bytes)

    print(f"  frame {total_f}/{total_f}")

    proc.stdin.close()
    proc.wait()

    return total_f


def morph_gpu(Y_source, Y_target, labels=None, n_steps=300,
              fps=60, theme="dark", colors=None, point_size=2, alpha=1.0,
              init_hold=1.0, end_hold=2.0, save="morph.mp4",
              width=1000, height=1000, bitrate=8000):
    """Smooth morphing animation between two paired 2D embeddings.

    Procrustes-aligns both embeddings (center, scale, rotate) then
    generates intermediate frames via linear interpolation on GPU.
    All computation (interpolation + rendering) runs on Metal GPU.

    Parameters
    ----------
    Y_source : array-like (n, 2)
        Source embedding (e.g. UMAP output).
    Y_target : array-like (n, 2)
        Target embedding (e.g. t-SNE output). Must be paired with source:
        Y_source[i] and Y_target[i] correspond to the same input sample.
    labels : array-like (n,), optional
        Integer labels for coloring.
    n_steps : int
        Number of interpolation steps (default 300).
    fps : int
        Frames per second (default 60).
    save : str
        Output video path.

    Returns
    -------
    int : total frames rendered.
    """
    import subprocess
    import mlx.core as mx
    from mlx_vis.render import _render_frame_mlx, _circle_template, _TEMPLATE_CACHE

    # --- Procrustes alignment ---
    X0 = np.asarray(Y_source, dtype=np.float64)
    X1 = np.asarray(Y_target, dtype=np.float64)
    n = len(X0)

    # Center
    X0 = X0 - X0.mean(axis=0)
    X1 = X1 - X1.mean(axis=0)

    # Per-axis unit std
    X0 = X0 / np.maximum(X0.std(axis=0), 1e-8)
    X1 = X1 / np.maximum(X1.std(axis=0), 1e-8)

    # Optimal rotation via SVD
    M = X0.T @ X1
    U, _, Vt = np.linalg.svd(M)
    d = np.linalg.det(U @ Vt)
    R = U @ np.diag([1.0, d]) @ Vt
    X1 = X1 @ R.T

    # --- Setup rendering ---
    X0_mx = mx.array(X0.astype(np.float32))
    X1_mx = mx.array(X1.astype(np.float32))

    # Global limits from both endpoints
    all_pts = np.concatenate([X0, X1], axis=0).astype(np.float32)
    xlim, ylim = _get_square_lims(all_pts)

    c = _resolve_colors(labels, colors, n, theme)
    c = np.array(c, dtype=np.float32)
    c[:, 3] = alpha

    bg_val = 0.0 if theme == "dark" else 1.0
    colors_mx = mx.array(c)
    bg_mx = mx.array([bg_val, bg_val, bg_val, 1.0], dtype=mx.float32)
    xmin, xmax = float(xlim[0]), float(xlim[1])
    ymin, ymax = float(ylim[0]), float(ylim[1])

    point_radius = max(1.0, point_size)
    rk = round(point_radius * 10)
    if rk not in _TEMPLATE_CACHE:
        _TEMPLATE_CACHE[rk] = _circle_template(point_radius)
    offsets, weights = _TEMPLATE_CACHE[rk]

    init_f = int(init_hold * fps)
    hold_f = int(end_hold * fps)
    total_f = init_f + (n_steps + 1) + hold_f

    # Precompute step delta on GPU (avoids per-frame scalar creation)
    delta = (X1_mx - X0_mx) / n_steps if n_steps > 0 else mx.zeros_like(X0_mx)
    mx.eval(delta)

    def _render_at(Y):
        return _render_frame_mlx(Y, colors_mx, offsets, weights,
                                 width, height, xmin, xmax, ymin, ymax, bg_mx)

    # --- Encode video ---
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

    # Init hold
    Y_cur = X0_mx
    frame0 = _render_at(Y_cur)
    mx.eval(frame0)
    buf0 = np.array(frame0).tobytes()
    for _ in range(init_f):
        proc.stdin.write(buf0)

    # Interpolation frames with GPU pipeline: Y += delta each step
    proc.stdin.write(buf0)  # step 0
    if n_steps > 0:
        Y_cur = Y_cur + delta
        frame = _render_at(Y_cur)
        mx.async_eval(frame)

        for step in range(2, n_steps + 1):
            Y_cur = Y_cur + delta
            next_frame = _render_at(Y_cur)
            mx.async_eval(next_frame)
            mx.eval(frame)
            proc.stdin.write(np.array(frame).tobytes())
            frame = next_frame
            done = init_f + step
            if done % 100 == 0:
                print(f"  frame {done}/{total_f}")

        mx.eval(frame)
        last_bytes = np.array(frame).tobytes()
        proc.stdin.write(last_bytes)
    else:
        last_bytes = buf0

    # End hold
    for _ in range(hold_f):
        proc.stdin.write(last_bytes)

    print(f"  frame {total_f}/{total_f}")
    proc.stdin.close()
    proc.wait()

    return total_f


def morph_all_effect(embeddings, labels=None, n_steps=200, fps=60,
                     theme="dark", colors=None, point_size=1.5, alpha=1.0,
                     hold=1.0, fade=0.5, decay=0.90, fade_decay=0.82,
                     save="morph_all.mp4", width=1000, height=1000,
                     bitrate=5000, order="similarity"):
    """Morph carousel across multiple embeddings with comet trails and ease-in-out.

    Cycles through all embeddings with smooth transitions, fading trails,
    and cubic ease-in-out timing. All rendering runs on Metal GPU.

    Parameters
    ----------
    embeddings : dict[str, array-like (n, 2)]
        Named embeddings to morph between. All must share the same sample
        ordering (embeddings[name][i] corresponds to the same input sample).
    labels : array-like (n,), optional
        Integer labels for coloring.
    n_steps : int
        Interpolation steps per transition (default 200).
    fps : int
        Frames per second (default 60).
    hold : float
        Seconds to hold each static endpoint (default 1.0).
    fade : float
        Seconds for trail fade-out at each endpoint (default 0.5).
    decay : float
        Trail decay factor per frame during morph (default 0.90).
    fade_decay : float
        Faster decay during fade-out phase (default 0.82).
    save : str
        Output video path.
    order : str
        "similarity" for greedy nearest-neighbor ordering (default),
        or "given" to use dict insertion order.

    Returns
    -------
    int : total frames rendered.
    """
    import subprocess
    import mlx.core as mx

    names = list(embeddings.keys())
    n_methods = len(names)
    if n_methods < 2:
        raise ValueError("Need at least 2 embeddings")

    # Procrustes-align all to first
    raw = {n: np.asarray(embeddings[n], dtype=np.float64) for n in names}
    n = len(raw[names[0]])

    def _norm(Y):
        Y = Y - Y.mean(0)
        Y = Y / np.maximum(Y.std(0), 1e-8)
        return Y

    normed = {n: _norm(raw[n]) for n in names}
    ref = normed[names[0]]
    aligned = {names[0]: ref.astype(np.float32)}
    for name in names[1:]:
        M = ref.T @ normed[name]
        U, _, Vt = np.linalg.svd(M)
        d = np.linalg.det(U @ Vt)
        R = U @ np.diag([1.0, d]) @ Vt
        aligned[name] = (normed[name] @ R.T).astype(np.float32)

    # Determine order
    if order == "similarity" and n_methods > 2:
        dist_matrix = np.zeros((n_methods, n_methods))
        for i in range(n_methods):
            for j in range(i + 1, n_methods):
                mse = np.mean((aligned[names[i]] - aligned[names[j]]) ** 2)
                dist_matrix[i, j] = dist_matrix[j, i] = mse
        visited = [0]
        remaining = set(range(1, n_methods))
        while remaining:
            last = visited[-1]
            nearest = min(remaining, key=lambda x: dist_matrix[last, x])
            visited.append(nearest)
            remaining.remove(nearest)
        ordered = [names[i] for i in visited]
    else:
        ordered = names

    print(f"  order: {' -> '.join(ordered)}")

    # Rendering setup
    all_pts = np.concatenate(list(aligned.values()), axis=0)
    xlim, ylim = _get_square_lims(all_pts)
    xmin, xmax = float(xlim[0]), float(xlim[1])
    ymin, ymax = float(ylim[0]), float(ylim[1])

    c = _resolve_colors(labels, colors, n, theme)
    c = np.array(c, dtype=np.float32)
    c[:, 3] = alpha
    colors_mx = mx.array(c)

    bg_val = 0.0 if theme == "dark" else 1.0
    bg_mx = mx.array([bg_val, bg_val, bg_val, 1.0], dtype=mx.float32)

    from mlx_vis.render import _render_frame_mlx, _circle_template, _TEMPLATE_CACHE
    point_radius = max(1.0, point_size)
    rk = round(point_radius * 10)
    if rk not in _TEMPLATE_CACHE:
        _TEMPLATE_CACHE[rk] = _circle_template(point_radius)
    offsets, weights = _TEMPLATE_CACHE[rk]

    hold_f = int(hold * fps)
    fade_f = int(fade * fps)
    pairs = [(ordered[i], ordered[(i + 1) % len(ordered)]) for i in range(len(ordered))]
    total_f = len(pairs) * (hold_f + n_steps + fade_f)

    def _ease(t):
        if t < 0.5:
            return 4 * t * t * t
        return 1 - (-2 * t + 2) ** 3 / 2

    def _render(Y):
        return _render_frame_mlx(Y, colors_mx, offsets, weights,
                                 width, height, xmin, xmax, ymin, ymax, bg_mx)

    # Encode
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

    accum = mx.zeros((height, width, 4), dtype=mx.float32)
    done = 0

    for src_name, dst_name in pairs:
        X0 = mx.array(aligned[src_name])
        X1 = mx.array(aligned[dst_name])

        # Hold
        f = _render(X0)
        mx.eval(f)
        buf = np.array(f).tobytes()
        accum = f.astype(mx.float32) / 255.0
        for _ in range(hold_f):
            proc.stdin.write(buf)
        done += hold_f

        # Morph with trails + ease-in-out
        for step in range(n_steps):
            t = _ease((step + 1) / n_steps)
            Y = (1.0 - t) * X0 + t * X1
            f = _render(Y)
            f_float = f.astype(mx.float32) / 255.0
            trail = accum * decay
            accum = mx.maximum(trail, f_float)
            out_frame = (mx.clip(accum, 0.0, 1.0) * 255).astype(mx.uint8)
            mx.eval(out_frame)
            proc.stdin.write(np.array(out_frame).tobytes())
        done += n_steps

        # Fade-out trails
        endpoint = _render(X1)
        endpoint_float = endpoint.astype(mx.float32) / 255.0
        mx.eval(endpoint_float)
        for _ in range(fade_f):
            accum = accum * fade_decay
            accum = mx.maximum(accum, endpoint_float)
            out_frame = (mx.clip(accum, 0.0, 1.0) * 255).astype(mx.uint8)
            mx.eval(out_frame)
            proc.stdin.write(np.array(out_frame).tobytes())
        done += fade_f

        if done % 100 == 0 or done == total_f:
            print(f"  {src_name} -> {dst_name} ({done}/{total_f})")

    proc.stdin.close()
    proc.wait()
    print(f"  {done}/{total_f} frames, saved to {save}")

    return total_f
