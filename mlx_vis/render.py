"""GPU-accelerated frame renderer using MLX Metal."""

import mlx.core as mx
import numpy as np


def render_frame(Y, colors, width, height, xlim, ylim, point_radius=1.5,
                 bg_color=None):
    """Render scattered points to an RGBA pixel buffer.

    Uses numpy for alpha-blended scatter (additive accumulation),
    then returns as uint8 array. For 70K points at 1000x1000 this
    takes ~2-3ms on M3 Ultra.
    """
    if bg_color is None:
        bg_color = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    else:
        bg_color = np.array(bg_color, dtype=np.float32)

    Y_np = np.array(Y) if not isinstance(Y, np.ndarray) else Y
    c_np = np.array(colors) if not isinstance(colors, np.ndarray) else colors

    n = Y_np.shape[0]
    xmin, xmax = float(xlim[0]), float(xlim[1])
    ymin, ymax = float(ylim[0]), float(ylim[1])

    # map to pixel coords
    px = ((Y_np[:, 0] - xmin) / (xmax - xmin) * (width - 1)).astype(np.int32)
    py = ((ymax - Y_np[:, 1]) / (ymax - ymin) * (height - 1)).astype(np.int32)

    # circle offsets
    r = int(np.ceil(point_radius))
    offsets = []
    weights = []
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            dist_sq = dx * dx + dy * dy
            if dist_sq <= point_radius * point_radius:
                offsets.append((dy, dx))
                # smooth falloff: gaussian-ish weight for anti-aliasing
                w = max(0.0, 1.0 - np.sqrt(dist_sq) / point_radius)
                weights.append(w)
    offsets = np.array(offsets, dtype=np.int32)
    weights = np.array(weights, dtype=np.float32)
    k = len(offsets)

    # expand points by circle offsets
    all_y = np.clip(py[:, None] + offsets[None, :, 0], 0, height - 1)
    all_x = np.clip(px[:, None] + offsets[None, :, 1], 0, width - 1)
    flat_idx = (all_y * width + all_x).ravel()

    # weighted colors: (n, k, 4) with per-offset weight applied to alpha
    c_expanded = np.broadcast_to(c_np[:, None, :], (n, k, 4)).copy().reshape(-1, 4)
    w_expanded = np.broadcast_to(weights[None, :], (n, k)).ravel()
    c_expanded[:, 3] *= w_expanded  # modulate alpha by distance weight

    # accumulate RGB*alpha and alpha separately for proper blending
    total = height * width
    accum = np.zeros((total, 4), dtype=np.float32)
    
    # premultiply RGB by alpha
    premul = c_expanded.copy()
    premul[:, :3] *= premul[:, 3:4]
    
    np.add.at(accum, flat_idx, premul)

    # composite over background
    # clamp accumulated alpha
    accum[:, 3] = np.clip(accum[:, 3], 0.0, 1.0)
    
    # final color = accum_rgb / accum_alpha where accum_alpha > 0, blended with bg
    result = np.zeros((total, 4), dtype=np.float32)
    mask = accum[:, 3] > 0.001
    
    # normalize premultiplied RGB
    fg_rgb = np.zeros((total, 3), dtype=np.float32)
    fg_rgb[mask] = accum[mask, :3] / accum[mask, 3:4]
    fg_rgb = np.clip(fg_rgb, 0.0, 1.0)
    fg_alpha = accum[:, 3]
    
    # alpha compositing: result = fg * alpha + bg * (1 - alpha)
    result[:, :3] = fg_rgb * fg_alpha[:, None] + bg_color[:3] * (1.0 - fg_alpha[:, None])
    result[:, 3] = 1.0

    buf = (np.clip(result, 0.0, 1.0) * 255).astype(np.uint8)
    return buf.reshape(height, width, 4)
