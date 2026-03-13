"""NNDescent approximate k-NN graph construction in pure MLX for Apple Silicon.

Reference: Dong et al. "Efficient K-Nearest Neighbor Graph Construction for Generic
Similarity Measures" (WWW 2011).

Entire pipeline on Metal GPU via MLX. No numpy in the hot path.
"""

import mlx.core as mx
import numpy as np
import time


class NNDescent:
    """Approximate k-NN graph via NNDescent on Metal GPU."""

    def __init__(
        self,
        k: int = 15,
        n_iters: int = 20,
        max_candidates: int | None = None,
        delta: float = 0.015,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        self.k = k
        self.n_iters = n_iters
        self.max_candidates = max_candidates
        self.delta = delta
        self.random_state = random_state
        self.verbose = verbose
        self.neighbor_graph = None

    def build(self, X) -> tuple[np.ndarray, np.ndarray]:
        """Build approximate k-NN graph.

        Returns:
            (indices, distances): both (n, k) numpy arrays. Euclidean distances.
        """
        if isinstance(X, np.ndarray):
            X = mx.array(X.astype(np.float32))

        n, d = X.shape
        k = min(self.k, n - 1)
        mc = self.max_candidates or k

        if self.random_state is not None:
            np.random.seed(self.random_state)

        # Precompute squared norms
        sq_norms = mx.sum(X * X, axis=1)  # (n,)

        # Random init
        idx_np = np.random.randint(0, n - 1, (n, k), dtype=np.int32)
        i_vals = np.arange(n, dtype=np.int32)[:, None]
        idx_np = np.where(idx_np >= i_vals, idx_np + 1, idx_np).astype(np.int32)
        indices = mx.array(idx_np)

        # Initial distances
        dists = _gather_dists(X, sq_norms, indices)  # (n, k)

        # Sort
        si = mx.argsort(dists, axis=1)
        indices = mx.take_along_axis(indices, si, axis=1)
        dists = mx.take_along_axis(dists, si, axis=1)
        mx.eval(indices, dists, sq_norms)

        # Low-dim random projection for early iters: fewer FLOPs, similar neighbor ranking.
        # Iter 0 (random init): 100d is as good as 784d - all neighbors are random anyway.
        # Iter 1 (graph still noisy, uf~1.0): 300d provides better approximation.
        # After each low-dim iter, recompute dists in full-dim before continuing.
        proj_dim = min(100, d)
        proj_dim2 = min(300, d)
        if proj_dim < d:
            P = mx.array(np.random.randn(d, proj_dim).astype(np.float32) / np.sqrt(proj_dim))
            X_low = X @ P          # (n, proj_dim)
            sq_norms_low = mx.sum(X_low * X_low, axis=1)
            mx.eval(X_low, sq_norms_low)
        else:
            X_low, sq_norms_low = X, sq_norms
        # P2 is computed lazily inside iter 1 to avoid persistent VRAM.
        P2 = mx.array(np.random.randn(d, proj_dim2).astype(np.float32) / np.sqrt(proj_dim2)) if proj_dim2 < d else None

        t0 = time.time()
        update_frac = 1.0  # initialize before first iteration
        flags = mx.ones((n, k), dtype=mx.bool_)  # all NEW on first iter
        for it in range(self.n_iters):
            # ---- NEW/OLD flag tracking (GNND-style selective update) ----
            # Only use NEW neighbors as sources for nn_of_nn when graph is converging.
            # Skip OLD-OLD cross-matching: old neighbors' neighborhoods already explored.
            # Only activate when update_frac < 0.5 - early iters still need full exploration.

            mc_nn = max(3, int(min(k, mc) * min(1.0, update_frac ** 0.5 * 3)))
            j_nn = max(k // 2, min(k, int(k * min(1.0, update_frac ** 0.5 * 3))))

            if update_frac < 0.5:
                # Sort indices to put NEW neighbors first for easy slicing
                new_order = mx.argsort((~flags).astype(mx.int32), axis=1)  # NEW=0 sorts first
                new_first = mx.take_along_axis(indices, new_order, axis=1)
                mc_new = max(3, int(k * update_frac))

                # Active-point optimization: skip fully converged points (no NEW neighbors).
                # Points with all-OLD neighbors have no new candidates to explore.
                active_mask = mx.any(flags, axis=1)  # (n,) - True if any neighbor is NEW
                active_idx = mx.array(
                    np.where(np.array(active_mask))[0].astype(np.int32)
                )
                n_active = active_idx.shape[0]

                if n_active < n:
                    # Sparse path: compute only for active points
                    nn_sub_a = new_first[active_idx, :mc_new]
                    nn_of_nn_a = indices[nn_sub_a.reshape(-1)].reshape(n_active, mc_new, k)[:, :, :j_nn]

                    if update_frac >= 0.10:
                        src_all = mx.broadcast_to(mx.arange(n)[:, None], (n, k)).reshape(-1)
                        dst_all = indices.reshape(-1)
                        rev_order = mx.argsort(dst_all)
                        rev_src = src_all[rev_order]
                        rev_dst = dst_all[rev_order]
                        is_new_group = mx.concatenate([
                            mx.ones((1,), dtype=mx.int32),
                            (rev_dst[1:] != rev_dst[:-1]).astype(mx.int32)
                        ])
                        global_pos = mx.arange(n * k)
                        group_start_markers = mx.where(is_new_group.astype(mx.bool_), global_pos, 0)
                        group_starts = mx.cummax(group_start_markers)
                        within_pos = global_pos - group_starts
                        keep = within_pos < k
                        flat_idx = rev_dst * k + within_pos
                        flat_idx = mx.where(keep, flat_idx, 0)
                        rev_src_kept = mx.where(keep, rev_src, 0)
                        rev_cands_full = mx.zeros((n * k,), dtype=mx.int32).at[flat_idx].add(rev_src_kept).reshape(n, k)
                        new_cands_a = mx.concatenate([
                            nn_of_nn_a.reshape(n_active, mc_new * j_nn),
                            rev_cands_full[active_idx],
                        ], axis=1)
                    else:
                        new_cands_a = nn_of_nn_a.reshape(n_active, mc_new * j_nn)

                    new_dists_a = _gather_dists_active(X, sq_norms, new_cands_a, active_idx)

                    all_cands_a = mx.concatenate([indices[active_idx], new_cands_a], axis=1)
                    all_dists_a = mx.concatenate([dists[active_idx], new_dists_a], axis=1)
                    self_mask_a = all_cands_a == active_idx[:, None]
                    all_dists_a = mx.where(self_mask_a, 1e30, all_dists_a)
                    cand_sort = mx.argsort(all_cands_a, axis=1)
                    sorted_c = mx.take_along_axis(all_cands_a, cand_sort, axis=1)
                    sorted_d = mx.take_along_axis(all_dists_a, cand_sort, axis=1)
                    is_dup = mx.concatenate([
                        mx.zeros((n_active, 1), dtype=mx.bool_),
                        sorted_c[:, 1:] == sorted_c[:, :-1]
                    ], axis=1)
                    sorted_d = mx.where(is_dup, 1e30, sorted_d)
                    top_idx = mx.argpartition(sorted_d, kth=k - 1, axis=1)[:, :k]
                    top_dists_a = mx.take_along_axis(sorted_d, top_idx, axis=1)
                    sub_sort = mx.argsort(top_dists_a, axis=1)
                    top_idx = mx.take_along_axis(top_idx, sub_sort, axis=1)
                    new_idx_a = mx.take_along_axis(sorted_c, top_idx, axis=1)
                    new_dist_a = mx.take_along_axis(sorted_d, top_idx, axis=1)
                    mx.eval(new_idx_a, new_dist_a)

                    # Scatter active results back: use zeros+add scatter then where-combine
                    new_idx_full = mx.zeros((n, k), dtype=mx.int32).at[active_idx].add(new_idx_a)
                    new_dist_full = mx.zeros((n, k)).at[active_idx].add(new_dist_a)
                    new_indices = mx.where(active_mask[:, None], new_idx_full, indices)
                    new_dists = mx.where(active_mask[:, None], new_dist_full, dists)
                    mx.eval(new_indices, new_dists)
                else:
                    # All points active: fall through to dense path
                    nn_sub = new_first[:, :mc_new]
                    nn_of_nn = indices[nn_sub.reshape(-1)].reshape(n, mc_new, k)[:, :, :j_nn]
                    if update_frac >= 0.10:
                        src_all = mx.broadcast_to(mx.arange(n)[:, None], (n, k)).reshape(-1)
                        dst_all = indices.reshape(-1)
                        rev_order = mx.argsort(dst_all)
                        rev_src = src_all[rev_order]
                        rev_dst = dst_all[rev_order]
                        is_new_group = mx.concatenate([
                            mx.ones((1,), dtype=mx.int32),
                            (rev_dst[1:] != rev_dst[:-1]).astype(mx.int32)
                        ])
                        global_pos = mx.arange(n * k)
                        group_start_markers = mx.where(is_new_group.astype(mx.bool_), global_pos, 0)
                        group_starts = mx.cummax(group_start_markers)
                        within_pos = global_pos - group_starts
                        keep = within_pos < k
                        flat_idx = rev_dst * k + within_pos
                        flat_idx = mx.where(keep, flat_idx, 0)
                        rev_src_kept = mx.where(keep, rev_src, 0)
                        rev_cands = mx.zeros((n * k,), dtype=mx.int32).at[flat_idx].add(rev_src_kept).reshape(n, k)
                        new_cands = mx.concatenate([nn_of_nn.reshape(n, mc_new * j_nn), rev_cands], axis=1)
                    else:
                        new_cands = nn_of_nn.reshape(n, mc_new * j_nn)
                    new_dists_comp = _gather_dists(X, sq_norms, new_cands)
                    all_cands = mx.concatenate([indices, new_cands], axis=1)
                    all_dists = mx.concatenate([dists, new_dists_comp], axis=1)
                    self_mask = all_cands == mx.arange(n)[:, None]
                    all_dists = mx.where(self_mask, 1e30, all_dists)
                    cand_sort = mx.argsort(all_cands, axis=1)
                    sorted_c = mx.take_along_axis(all_cands, cand_sort, axis=1)
                    sorted_d = mx.take_along_axis(all_dists, cand_sort, axis=1)
                    is_dup = mx.concatenate([
                        mx.zeros((n, 1), dtype=mx.bool_),
                        sorted_c[:, 1:] == sorted_c[:, :-1]
                    ], axis=1)
                    sorted_d = mx.where(is_dup, 1e30, sorted_d)
                    top_idx = mx.argpartition(sorted_d, kth=k - 1, axis=1)[:, :k]
                    top_dists = mx.take_along_axis(sorted_d, top_idx, axis=1)
                    sub_sort = mx.argsort(top_dists, axis=1)
                    top_idx = mx.take_along_axis(top_idx, sub_sort, axis=1)
                    new_indices = mx.take_along_axis(sorted_c, top_idx, axis=1)
                    new_dists = mx.take_along_axis(sorted_d, top_idx, axis=1)
                    mx.eval(new_indices, new_dists)
            else:
                # Full update: all n points, use top mc_nn neighbors as sources
                mc_new = mc_nn
                nn_sub = indices[:, :mc_new]
                nn_of_nn = indices[nn_sub.reshape(-1)].reshape(n, mc_new, k)[:, :, :j_nn]

                # Reverse candidates: for edge (i -> j), i is a candidate for j
                # Skip when nearly converged (update_frac < 0.10) - saves argsort(n*k)
                if update_frac >= 0.10:
                    src_all = mx.broadcast_to(mx.arange(n)[:, None], (n, k)).reshape(-1)
                    dst_all = indices.reshape(-1)

                    rev_order = mx.argsort(dst_all)
                    rev_src = src_all[rev_order]
                    rev_dst = dst_all[rev_order]

                    # Within-group position via cumsum trick
                    is_new_group = mx.concatenate([
                        mx.ones((1,), dtype=mx.int32),
                        (rev_dst[1:] != rev_dst[:-1]).astype(mx.int32)
                    ])
                    # Running count within each group
                    global_pos = mx.arange(n * k)
                    group_start_markers = mx.where(is_new_group.astype(mx.bool_), global_pos, 0)
                    # Forward-fill group starts using cummax
                    group_starts = mx.cummax(group_start_markers)
                    within_pos = global_pos - group_starts

                    # Keep only first k per group, scatter into (n, k) array
                    keep = within_pos < k
                    flat_idx = rev_dst * k + within_pos
                    flat_idx = mx.where(keep, flat_idx, 0)
                    rev_src_kept = mx.where(keep, rev_src, 0)

                    rev_cands = mx.zeros((n * k,), dtype=mx.int32).at[flat_idx].add(rev_src_kept).reshape(n, k)

                    # New candidates only (skip current indices - reuse known dists)
                    new_cands = mx.concatenate([
                        nn_of_nn.reshape(n, mc_new * j_nn),  # (n, mc_new*j_nn)
                        rev_cands,                            # (n, k)
                    ], axis=1)
                else:
                    # Converging: skip reverse candidates, use only nn-of-nn
                    new_cands = nn_of_nn.reshape(n, mc_new * j_nn)

                # Compute distances only for new candidates (reuse dists for current indices).
                # Early iters use low-dim projections: same neighbor ranking, fewer FLOPs.
                if it == 0:
                    Xd, sq_d = X_low, sq_norms_low
                elif it == 1 and P2 is not None:
                    # Compute 300d projection on-the-fly (avoids persistent VRAM storage).
                    Xd = X @ P2
                    sq_d = mx.sum(Xd * Xd, axis=1)
                    mx.eval(Xd, sq_d)
                else:
                    Xd, sq_d = X, sq_norms
                # fp16 matmul for iters 1-4 (noisy iters: 300d-fp16 for iter 1,
                # full-dim fp16 for iters 2-4). fp16 errors corrected by fp32 recompute after iter 4.
                # Iter 1 (300d): 300d already approximate, fp16 adds negligible error.
                if it in (1, 2, 3, 4):
                    new_dists_comp = _gather_dists_half(Xd, sq_d, new_cands)
                    cur_dists = _gather_dists_half(Xd, sq_d, indices)
                else:
                    new_dists_comp = _gather_dists(Xd, sq_d, new_cands)
                    cur_dists = _gather_dists(Xd, sq_d, indices) if it == 0 else dists
                all_cands = mx.concatenate([indices, new_cands], axis=1)
                all_dists = mx.concatenate([cur_dists, new_dists_comp], axis=1)

                # Mask self-references
                self_mask = all_cands == mx.arange(n)[:, None]
                all_dists = mx.where(self_mask, 1e30, all_dists)

                # Deduplicate per row: sort by candidate index, mark consecutive dups
                cand_sort = mx.argsort(all_cands, axis=1)
                sorted_c = mx.take_along_axis(all_cands, cand_sort, axis=1)
                sorted_d = mx.take_along_axis(all_dists, cand_sort, axis=1)

                is_dup = mx.concatenate([
                    mx.zeros((n, 1), dtype=mx.bool_),
                    sorted_c[:, 1:] == sorted_c[:, :-1]
                ], axis=1)
                sorted_d = mx.where(is_dup, 1e30, sorted_d)

                # Select top k directly from sorted space (skip unsort)
                top_idx = mx.argpartition(sorted_d, kth=k - 1, axis=1)[:, :k]
                top_dists = mx.take_along_axis(sorted_d, top_idx, axis=1)
                sub_sort = mx.argsort(top_dists, axis=1)
                top_idx = mx.take_along_axis(top_idx, sub_sort, axis=1)

                new_indices = mx.take_along_axis(sorted_c, top_idx, axis=1)
                new_dists = mx.take_along_axis(sorted_d, top_idx, axis=1)

                mx.eval(new_indices, new_dists)

                # Recompute fp32 distances only after fp16 iter 4:
                # - iters 0-1 (low-dim): no recompute needed since iters 2-4 (fp16) recompute
                #   cur_dists fresh; stored dists are never used as cur_dists until iter 5
                # - fp16 iter 4: recompute corrects accumulated error before dense fp32 iter 5
                if it == 4:
                    new_dists = _gather_dists(X, sq_norms, new_indices)
                    mx.eval(new_dists)

            # Track which neighbors changed (NEW) vs stayed (OLD)
            flags = new_indices != indices

            # Count updates
            changed = int(mx.sum(flags))
            update_frac = changed / (n * k)

            indices = new_indices
            dists = new_dists

            if self.verbose:
                elapsed = time.time() - t0
                print(f"Iter {it+1}/{self.n_iters}: {changed} updates "
                      f"({update_frac:.4f}), {elapsed:.2f}s")

            if update_frac < self.delta:
                if self.verbose:
                    print(f"Converged at iteration {it+1}")
                break

        final_dists = mx.sqrt(mx.maximum(dists, 0.0))
        mx.eval(indices, final_dists)
        self.neighbor_graph = (np.array(indices), np.array(final_dists))
        return self.neighbor_graph


def _rp_tree_init(X, n, k, d, n_trees=8, leaf_size=None):
    """Initialize k-NN graph using random projection trees.

    Build n_trees random projection trees. Points in the same leaf are
    candidate neighbors. Fill remaining slots with random points.

    All on MLX GPU.
    """
    if leaf_size is None:
        leaf_size = max(k * 2, 64)

    # Store candidates per point as a set (numpy, then convert)
    # For efficiency, collect leaf memberships and generate pairs
    cand_indices = np.full((n, k), -1, dtype=np.int32)
    cand_count = np.zeros(n, dtype=np.int32)

    X_np = np.array(X)

    for _ in range(n_trees):
        # Build one RP tree: recursively split data
        leaves = _build_rp_tree(X_np, np.arange(n), leaf_size, d)

        # For each leaf, all points in the leaf are candidates for each other
        for leaf in leaves:
            if len(leaf) <= 1:
                continue
            for i in range(len(leaf)):
                p = leaf[i]
                for j in range(len(leaf)):
                    if i == j:
                        continue
                    q = leaf[j]
                    if cand_count[p] < k:
                        cand_indices[p, cand_count[p]] = q
                        cand_count[p] += 1

    # Fill remaining slots with random points
    for i in range(n):
        while cand_count[i] < k:
            r = np.random.randint(0, n - 1)
            if r >= i:
                r += 1
            cand_indices[i, cand_count[i]] = r
            cand_count[i] += 1

    return mx.array(cand_indices)


def _build_rp_tree(X, point_ids, leaf_size, d):
    """Recursively build a random projection tree. Returns list of leaf arrays."""
    if len(point_ids) <= leaf_size:
        return [point_ids]

    # Random hyperplane: pick two random points, split by midpoint
    i, j = np.random.choice(len(point_ids), 2, replace=False)
    p1, p2 = X[point_ids[i]], X[point_ids[j]]
    normal = p2 - p1
    norm = np.linalg.norm(normal)
    if norm < 1e-10:
        # Degenerate: split randomly
        mid = len(point_ids) // 2
        return (_build_rp_tree(X, point_ids[:mid], leaf_size, d) +
                _build_rp_tree(X, point_ids[mid:], leaf_size, d))

    normal /= norm
    midpoint = (p1 + p2) / 2

    # Project all points
    projections = (X[point_ids] - midpoint) @ normal
    left_mask = projections <= 0

    left_ids = point_ids[left_mask]
    right_ids = point_ids[~left_mask]

    # Avoid empty splits
    if len(left_ids) == 0 or len(right_ids) == 0:
        mid = len(point_ids) // 2
        return (_build_rp_tree(X, point_ids[:mid], leaf_size, d) +
                _build_rp_tree(X, point_ids[mid:], leaf_size, d))

    return (_build_rp_tree(X, left_ids, leaf_size, d) +
            _build_rp_tree(X, right_ids, leaf_size, d))


def _gather_dists(X, sq_norms, col_ids):
    """Squared distances from each point i to col_ids[i, :].

    Chunked to limit memory. All MLX, single eval at end.

    Args:
        X: (n, d) data
        sq_norms: (n,) precomputed ||x||^2
        col_ids: (n, c) neighbor indices

    Returns:
        (n, c) squared distances
    """
    n, c = col_ids.shape
    d = X.shape[1]

    # Chunk to keep intermediate (cs, c, d) under ~2000MB
    max_cs = max(1, 500_000_000 // (c * d))
    max_cs = min(max_cs, n)

    if max_cs >= n:
        # Single chunk - no eval boundary
        flat = col_ids.reshape(-1)
        X_tgt = X[flat].reshape(n, c, d)
        X_src = X
        dots = mx.einsum('id,icd->ic', X_src, X_tgt)
        return mx.maximum(sq_norms[:, None] + sq_norms[flat].reshape(n, c) - 2.0 * dots, 0.0)

    chunks = []
    for s in range(0, n, max_cs):
        e = min(s + max_cs, n)
        cs = e - s
        flat = col_ids[s:e].reshape(-1)
        X_tgt = X[flat].reshape(cs, c, d)
        X_src = X[s:e]
        dots = mx.einsum('id,icd->ic', X_src, X_tgt)
        chunk_d = mx.maximum(sq_norms[s:e][:, None] + sq_norms[flat].reshape(cs, c) - 2.0 * dots, 0.0)
        mx.eval(chunk_d)
        chunks.append(chunk_d)
    return mx.concatenate(chunks, axis=0)


def _gather_dists_half(X, sq_norms, col_ids):
    """Like _gather_dists but uses fp16 matmul (1.8x faster) for noisy early iters.

    sq_norms stays in fp32 for numerical stability of the final subtraction.
    """
    n, c = col_ids.shape
    d = X.shape[1]
    max_cs = max(1, 500_000_000 // (c * d))
    max_cs = min(max_cs, n)
    Xh = X.astype(mx.float16)
    if max_cs >= n:
        flat = col_ids.reshape(-1)
        X_tgt = Xh[flat].reshape(n, c, d)
        dots = mx.einsum('id,icd->ic', Xh, X_tgt).astype(mx.float32)
        return mx.maximum(sq_norms[:, None] + sq_norms[flat].reshape(n, c) - 2.0 * dots, 0.0)
    chunks = []
    for s in range(0, n, max_cs):
        e = min(s + max_cs, n)
        cs = e - s
        flat = col_ids[s:e].reshape(-1)
        X_tgt = Xh[flat].reshape(cs, c, d)
        X_src = Xh[s:e]
        dots = mx.einsum('id,icd->ic', X_src, X_tgt).astype(mx.float32)
        chunk_d = mx.maximum(sq_norms[s:e][:, None] + sq_norms[flat].reshape(cs, c) - 2.0 * dots, 0.0)
        mx.eval(chunk_d)
        chunks.append(chunk_d)
    return mx.concatenate(chunks, axis=0)


def _gather_dists_active(X, sq_norms, col_ids, active_idx):
    """Squared distances for a sparse subset of source rows.

    Like _gather_dists but only computes for active_idx rows.
    X: (n, d) full data, sq_norms: (n,) full norms
    col_ids: (n_active, c) neighbor indices (global)
    active_idx: (n_active,) row indices into X
    """
    n_src, c = col_ids.shape
    d = X.shape[1]

    max_cs = max(1, 500_000_000 // (c * d))
    max_cs = min(max_cs, n_src)

    if max_cs >= n_src:
        flat = col_ids.reshape(-1)
        X_tgt = X[flat].reshape(n_src, c, d)
        X_src = X[active_idx]
        sq_src = sq_norms[active_idx]
        dots = mx.einsum('id,icd->ic', X_src, X_tgt)
        return mx.maximum(sq_src[:, None] + sq_norms[flat].reshape(n_src, c) - 2.0 * dots, 0.0)

    chunks = []
    for s in range(0, n_src, max_cs):
        e = min(s + max_cs, n_src)
        cs = e - s
        flat = col_ids[s:e].reshape(-1)
        X_tgt = X[flat].reshape(cs, c, d)
        X_src = X[active_idx[s:e]]
        sq_src = sq_norms[active_idx[s:e]]
        dots = mx.einsum('id,icd->ic', X_src, X_tgt)
        chunk_d = mx.maximum(sq_src[:, None] + sq_norms[flat].reshape(cs, c) - 2.0 * dots, 0.0)
        mx.eval(chunk_d)
        chunks.append(chunk_d)
    return mx.concatenate(chunks, axis=0)
