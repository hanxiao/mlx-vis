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

        # Memory budget for row chunking.
        # Each row costs roughly: total_cols * 8 (argsort intermediates) + mc_new*k
        # (nn_of_nn gather) elements. Distance computation is sub-chunked separately.
        # Empirical: row_cs=2000 peaks ~2.7GB, row_cs=6580 peaks ~6GB for k=91/d=784.
        # 300M budget -> row_cs~3950 for k=91 -> ~4.2GB peak.  k=15 stays single-shot.
        _MEM_BUDGET = 300_000_000

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

                # Use active points or all points
                use_rev = update_frac >= 0.10
                if n_active < n:
                    src_rows = active_idx  # (n_active,) global row indices
                    src_new_first = new_first[active_idx]  # (n_active, k)
                    n_src = n_active
                else:
                    src_rows = None  # means all rows
                    src_new_first = new_first
                    n_src = n

                if use_rev:
                    rev_cands = _compute_reverse_candidates(indices, n, k)
                else:
                    rev_cands = None

                # Row-chunked update for the converging path
                total_new_cols = mc_new * j_nn + (k if use_rev else 0)
                total_cols = k + total_new_cols
                elems_per_row = total_cols * 8 + mc_new * k
                row_cs = max(1, _MEM_BUDGET // max(elems_per_row, 1))
                row_cs = min(row_cs, n_src)

                result_indices = []
                result_dists = []

                for s in range(0, n_src, row_cs):
                    e = min(s + row_cs, n_src)
                    bs = e - s
                    if src_rows is not None:
                        global_rows = src_rows[s:e]  # global indices
                    else:
                        global_rows = mx.arange(s, e)

                    # nn_of_nn for this chunk
                    nn_sub = src_new_first[s:e, :mc_new]
                    nn_of_nn_chunk = indices[nn_sub.reshape(-1)].reshape(bs, mc_new, k)[:, :, :j_nn]

                    if rev_cands is not None:
                        nc_chunk = mx.concatenate([
                            nn_of_nn_chunk.reshape(bs, mc_new * j_nn),
                            rev_cands[global_rows] if src_rows is not None else rev_cands[s:e],
                        ], axis=1)
                    else:
                        nc_chunk = nn_of_nn_chunk.reshape(bs, mc_new * j_nn)

                    # Compute distances for new candidates
                    nc = nc_chunk.shape[1]
                    dist_max_cs = max(1, 200_000_000 // (nc * d))
                    dist_max_cs = min(dist_max_cs, bs)

                    dist_chunks = []
                    for ds in range(0, bs, dist_max_cs):
                        de = min(ds + dist_max_cs, bs)
                        dbs = de - ds
                        flat = nc_chunk[ds:de].reshape(-1)
                        X_tgt = X[flat].reshape(dbs, nc, d)
                        if src_rows is not None:
                            X_src = X[global_rows[ds:de]]
                            sq_src = sq_norms[global_rows[ds:de]]
                        else:
                            X_src = X[s + ds:s + de]
                            sq_src = sq_norms[s + ds:s + de]
                        dots = mx.einsum('id,icd->ic', X_src, X_tgt)
                        cd = mx.maximum(
                            sq_src[:, None] + sq_norms[flat].reshape(dbs, nc) - 2.0 * dots,
                            0.0
                        )
                        mx.eval(cd)
                        dist_chunks.append(cd)
                    new_d = dist_chunks[0] if len(dist_chunks) == 1 else mx.concatenate(dist_chunks, axis=0)

                    # Merge with existing neighbors
                    if src_rows is not None:
                        cur_idx = indices[global_rows]
                        cur_d = dists[global_rows]
                    else:
                        cur_idx = indices[s:e]
                        cur_d = dists[s:e]

                    ac = mx.concatenate([cur_idx, nc_chunk], axis=1)
                    ad = mx.concatenate([cur_d, new_d], axis=1)
                    self_mask = ac == global_rows[:, None]
                    ad = mx.where(self_mask, 1e30, ad)

                    ri, rd = _dedup_topk_block(ac, ad, bs, k)
                    mx.eval(ri, rd)
                    result_indices.append(ri)
                    result_dists.append(rd)

                new_idx_result = mx.concatenate(result_indices, axis=0)
                new_dist_result = mx.concatenate(result_dists, axis=0)
                mx.eval(new_idx_result, new_dist_result)

                if src_rows is not None:
                    # Scatter active results back
                    new_idx_full = mx.zeros((n, k), dtype=mx.int32).at[active_idx].add(new_idx_result)
                    new_dist_full = mx.zeros((n, k)).at[active_idx].add(new_dist_result)
                    new_indices = mx.where(active_mask[:, None], new_idx_full, indices)
                    new_dists = mx.where(active_mask[:, None], new_dist_full, dists)
                    mx.eval(new_indices, new_dists)
                else:
                    new_indices = new_idx_result
                    new_dists = new_dist_result
            else:
                # Full update: all n points, use top mc_nn neighbors as sources
                mc_new = mc_nn

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
                use_half = it in (1, 2, 3, 4)
                recompute_cur = (it == 0) or use_half

                # Reverse candidates: for edge (i -> j), i is a candidate for j
                # Skip when nearly converged (update_frac < 0.10) - saves argsort(n*k)
                use_rev = update_frac >= 0.10

                # Total candidate columns per row
                total_new_cols = mc_new * j_nn + (k if use_rev else 0)
                total_cols = k + total_new_cols
                # Row chunk sizing: the dedup/argsort pipeline creates ~8 arrays
                # of (bs, total_cols), plus the nn_of_nn gather of (bs, mc_new, k).
                # Distances are sub-chunked separately so don't dominate here.
                # Empirical calibration: row_cs=2000 peaks at ~2.7GB for k=91.
                elems_per_row = total_cols * 8 + mc_new * k
                row_cs = max(1, _MEM_BUDGET // max(elems_per_row, 1))
                row_cs = min(row_cs, n)

                if row_cs >= n:
                    # Small enough to do in one shot (e.g. k=15)
                    nn_sub = indices[:, :mc_new]
                    nn_of_nn = indices[nn_sub.reshape(-1)].reshape(n, mc_new, k)[:, :, :j_nn]

                    if use_rev:
                        rev_cands = _compute_reverse_candidates(indices, n, k)
                        new_cands = mx.concatenate([
                            nn_of_nn.reshape(n, mc_new * j_nn),
                            rev_cands,
                        ], axis=1)
                    else:
                        new_cands = nn_of_nn.reshape(n, mc_new * j_nn)

                    if use_half:
                        new_dists_comp = _gather_dists_half(Xd, sq_d, new_cands)
                        cur_dists = _gather_dists_half(Xd, sq_d, indices)
                    else:
                        new_dists_comp = _gather_dists(Xd, sq_d, new_cands)
                        cur_dists = _gather_dists(Xd, sq_d, indices) if recompute_cur else dists
                    all_cands = mx.concatenate([indices, new_cands], axis=1)
                    all_dists = mx.concatenate([cur_dists, new_dists_comp], axis=1)

                    self_mask = all_cands == mx.arange(n)[:, None]
                    all_dists = mx.where(self_mask, 1e30, all_dists)

                    new_indices, new_dists = _dedup_topk(all_cands, all_dists, n, k, _MEM_BUDGET)
                else:
                    # Row-chunked path: process nn_of_nn + dists + dedup in chunks
                    # to keep peak GPU memory bounded.

                    # Pre-compute reverse candidates (global operation)
                    if use_rev:
                        rev_cands = _compute_reverse_candidates(indices, n, k)
                    else:
                        rev_cands = None

                    result_indices = []
                    result_dists = []
                    dd = Xd.shape[1]
                    Xdh = Xd.astype(mx.float16) if use_half else None

                    for s in range(0, n, row_cs):
                        e = min(s + row_cs, n)
                        bs = e - s

                        # nn_of_nn for this chunk
                        nn_sub = indices[s:e, :mc_new]
                        nn_of_nn_chunk = indices[nn_sub.reshape(-1)].reshape(bs, mc_new, k)[:, :, :j_nn]

                        # Build new_cands for this chunk
                        if rev_cands is not None:
                            nc_chunk = mx.concatenate([
                                nn_of_nn_chunk.reshape(bs, mc_new * j_nn),
                                rev_cands[s:e]
                            ], axis=1)
                        else:
                            nc_chunk = nn_of_nn_chunk.reshape(bs, mc_new * j_nn)

                        # Compute distances for new candidates
                        nc = nc_chunk.shape[1]
                        dist_max_cs = max(1, 200_000_000 // (nc * dd))
                        dist_max_cs = min(dist_max_cs, bs)

                        dist_chunks = []
                        for ds in range(0, bs, dist_max_cs):
                            de = min(ds + dist_max_cs, bs)
                            dbs = de - ds
                            flat = nc_chunk[ds:de].reshape(-1)
                            if use_half:
                                X_tgt = Xdh[flat].reshape(dbs, nc, dd)
                                X_src = Xdh[s + ds:s + de]
                                dots = mx.einsum('id,icd->ic', X_src, X_tgt).astype(mx.float32)
                            else:
                                X_tgt = Xd[flat].reshape(dbs, nc, dd)
                                X_src = Xd[s + ds:s + de]
                                dots = mx.einsum('id,icd->ic', X_src, X_tgt)
                            cd = mx.maximum(
                                sq_d[s + ds:s + de, None] + sq_d[flat].reshape(dbs, nc) - 2.0 * dots,
                                0.0
                            )
                            mx.eval(cd)
                            dist_chunks.append(cd)
                        new_d = dist_chunks[0] if len(dist_chunks) == 1 else mx.concatenate(dist_chunks, axis=0)

                        # Current indices distances
                        if recompute_cur:
                            cur_flat = indices[s:e].reshape(-1)
                            if use_half:
                                X_tgt_c = Xdh[cur_flat].reshape(bs, k, dd)
                                X_src_c = Xdh[s:e]
                                dots_c = mx.einsum('id,icd->ic', X_src_c, X_tgt_c).astype(mx.float32)
                            else:
                                X_tgt_c = Xd[cur_flat].reshape(bs, k, dd)
                                X_src_c = Xd[s:e]
                                dots_c = mx.einsum('id,icd->ic', X_src_c, X_tgt_c)
                            cur_d = mx.maximum(
                                sq_d[s:e, None] + sq_d[cur_flat].reshape(bs, k) - 2.0 * dots_c,
                                0.0
                            )
                        else:
                            cur_d = dists[s:e]

                        # Merge and dedup
                        ac = mx.concatenate([indices[s:e], nc_chunk], axis=1)
                        ad = mx.concatenate([cur_d, new_d], axis=1)
                        self_mask = ac == mx.arange(s, e)[:, None]
                        ad = mx.where(self_mask, 1e30, ad)

                        ri, rd = _dedup_topk_block(ac, ad, bs, k)
                        mx.eval(ri, rd)
                        result_indices.append(ri)
                        result_dists.append(rd)

                    new_indices = mx.concatenate(result_indices, axis=0)
                    new_dists = mx.concatenate(result_dists, axis=0)
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


def _compute_reverse_candidates(indices, n, k):
    """Compute reverse candidates: for edge (i -> j), i is a candidate for j.

    Returns (n, k) array of reverse candidate source indices.
    """
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
    global_pos = mx.arange(n * k)
    group_start_markers = mx.where(is_new_group.astype(mx.bool_), global_pos, 0)
    group_starts = mx.cummax(group_start_markers)
    within_pos = global_pos - group_starts

    # Keep only first k per group, scatter into (n, k) array
    keep = within_pos < k
    flat_idx = rev_dst * k + within_pos
    flat_idx = mx.where(keep, flat_idx, 0)
    rev_src_kept = mx.where(keep, rev_src, 0)

    rev_cands = mx.zeros((n * k,), dtype=mx.int32).at[flat_idx].add(rev_src_kept).reshape(n, k)
    mx.eval(rev_cands)
    return rev_cands


def _dedup_topk_block(all_cands, all_dists, n_rows, k):
    """Deduplicate candidates and select top-k for a single block (no chunking)."""
    cand_sort = mx.argsort(all_cands, axis=1)
    sorted_c = mx.take_along_axis(all_cands, cand_sort, axis=1)
    sorted_d = mx.take_along_axis(all_dists, cand_sort, axis=1)

    is_dup = mx.concatenate([
        mx.zeros((n_rows, 1), dtype=mx.bool_),
        sorted_c[:, 1:] == sorted_c[:, :-1]
    ], axis=1)
    sorted_d = mx.where(is_dup, 1e30, sorted_d)

    top_idx = mx.argpartition(sorted_d, kth=k - 1, axis=1)[:, :k]
    top_dists = mx.take_along_axis(sorted_d, top_idx, axis=1)
    sub_sort = mx.argsort(top_dists, axis=1)
    top_idx = mx.take_along_axis(top_idx, sub_sort, axis=1)

    new_indices = mx.take_along_axis(sorted_c, top_idx, axis=1)
    new_dists = mx.take_along_axis(sorted_d, top_idx, axis=1)
    return new_indices, new_dists


def _dedup_topk(all_cands, all_dists, n_rows, k, mem_budget):
    """Deduplicate candidates and select top-k, chunked by rows if needed."""
    total_cols = all_cands.shape[1]
    # argsort + take_along_axis creates ~8 concurrent arrays of (row_cs, total_cols)
    row_cs = max(1, mem_budget // max(total_cols * 8, 1))
    row_cs = min(row_cs, n_rows)

    if row_cs >= n_rows:
        ri, rd = _dedup_topk_block(all_cands, all_dists, n_rows, k)
        mx.eval(ri, rd)
        return ri, rd

    result_indices = []
    result_dists = []
    for s in range(0, n_rows, row_cs):
        e = min(s + row_cs, n_rows)
        bs = e - s
        ri, rd = _dedup_topk_block(all_cands[s:e], all_dists[s:e], bs, k)
        mx.eval(ri, rd)
        result_indices.append(ri)
        result_dists.append(rd)

    new_indices = mx.concatenate(result_indices, axis=0)
    new_dists = mx.concatenate(result_dists, axis=0)
    mx.eval(new_indices, new_dists)
    return new_indices, new_dists


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

    # Chunk to keep intermediate (cs, c, d) under ~500MB
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
