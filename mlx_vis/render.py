"""GPU-accelerated frame renderer using MLX Metal."""

import mlx.core as mx
import numpy as np


def _circle_template(radius):
    """Pre-compute circle pixel offsets and distance-based weights."""
    r = int(np.ceil(radius))
    offsets = []
    weights = []
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            dist_sq = dx * dx + dy * dy
            if dist_sq <= radius * radius:
                offsets.append((dy, dx))
                # smooth gaussian-ish falloff for anti-aliasing
                w = max(0.0, 1.0 - np.sqrt(dist_sq) / radius)
                weights.append(w)
    return (mx.array(np.array(offsets, dtype=np.int32)),
            mx.array(np.array(weights, dtype=np.float32)))


# cache keyed by radius
_TEMPLATE_CACHE = {}


def render_frame(Y, colors, width, height, xlim, ylim, point_radius=1.5,
                 bg_color=None):
    """Render scattered points to an RGBA pixel buffer on Metal GPU.

    Uses mx.at[].add() for alpha-blended scatter accumulation.
    Returns (height, width, 4) uint8 numpy array.
    """
    if not isinstance(Y, mx.array):
        Y = mx.array(np.asarray(Y, dtype=np.float32))
    if not isinstance(colors, mx.array):
        colors = mx.array(np.asarray(colors, dtype=np.float32))

    if bg_color is None:
        bg_color = mx.array([0.0, 0.0, 0.0, 1.0], dtype=mx.float32)
    elif not isinstance(bg_color, mx.array):
        bg_color = mx.array(np.asarray(bg_color, dtype=np.float32))

    # get cached circle template
    rk = round(point_radius * 10)  # cache key
    if rk not in _TEMPLATE_CACHE:
        _TEMPLATE_CACHE[rk] = _circle_template(point_radius)
    offsets, weights = _TEMPLATE_CACHE[rk]
    k = offsets.shape[0]

    n = Y.shape[0]
    xmin, xmax = float(xlim[0]), float(xlim[1])
    ymin, ymax = float(ylim[0]), float(ylim[1])

    # map embedding coords to pixel coords
    px = ((Y[:, 0] - xmin) / (xmax - xmin) * (width - 1)).astype(mx.int32)
    py = ((ymax - Y[:, 1]) / (ymax - ymin) * (height - 1)).astype(mx.int32)

    # expand by circle offsets: (n, k)
    all_y = mx.clip(py[:, None] + offsets[None, :, 0], 0, height - 1)
    all_x = mx.clip(px[:, None] + offsets[None, :, 1], 0, width - 1)
    flat_idx = (all_y * width + all_x).reshape(-1)

    # premultiply colors by alpha, then modulate by distance weight
    premul_rgb = colors[:, :3] * colors[:, 3:4]  # (n, 3)
    premul_a = colors[:, 3:4]  # (n, 1)

    # expand to (n, k, 4): [r*a*w, g*a*w, b*a*w, a*w]
    w = weights[None, :]  # (1, k)
    rgb_expanded = (premul_rgb[:, None, :] * w[:, :, None]).reshape(-1, 3)  # (n*k, 3)
    a_expanded = (premul_a * w).reshape(-1, 1)  # (n*k, 1)
    vals = mx.concatenate([rgb_expanded, a_expanded], axis=1)  # (n*k, 4)

    # scatter-add into accumulation buffer
    total = height * width
    accum = mx.zeros((total, 4), dtype=mx.float32)
    accum = accum.at[flat_idx].add(vals)

    # composite over background
    alpha = mx.clip(accum[:, 3:4], 0.0, 1.0)
    # avoid division by zero
    safe_alpha = mx.maximum(alpha, 1e-6)
    fg_rgb = mx.clip(accum[:, :3] / safe_alpha, 0.0, 1.0)

    result_rgb = fg_rgb * alpha + bg_color[:3] * (1.0 - alpha)
    result_a = mx.ones((total, 1), dtype=mx.float32)
    result = mx.concatenate([result_rgb, result_a], axis=1)

    buf = (mx.clip(result, 0.0, 1.0) * 255).astype(mx.uint8)
    mx.eval(buf)
    return np.array(buf.reshape(height, width, 4))
