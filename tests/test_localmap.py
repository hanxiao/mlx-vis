import unittest

import numpy as np
import mlx.core as mx

from mlx_vis import LocalMAP, PaCMAP
from mlx_vis._pacmap.pacmap import _resample_local_fp_pairs, _sample_FP_pairs
from mlx_vis.localmap import LocalMAP as ModuleLocalMAP
from mlx_vis.pacmap import LocalMAP as PacmapLocalMAP


class LocalMAPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(7)
        cls.X = rng.normal(size=(64, 12)).astype(np.float32)

    def test_public_exports(self):
        self.assertIs(LocalMAP, ModuleLocalMAP)
        self.assertIs(LocalMAP, PacmapLocalMAP)

    def test_fit_transform_returns_finite_embedding(self):
        model = LocalMAP(
            n_components=2,
            n_neighbors=8,
            MN_ratio=0.5,
            FP_ratio=1.5,
            num_iters=(2, 2, 8),
            random_state=0,
            knn_method="brute",
        )

        embedding = model.fit_transform(self.X, init="random")

        self.assertEqual(embedding.shape, (64, 2))
        self.assertTrue(np.isfinite(embedding).all())
        self.assertEqual(model.low_dist_thres, 10.0)

    def test_random_state_is_deterministic(self):
        kwargs = dict(
            n_components=2,
            n_neighbors=8,
            MN_ratio=0.5,
            FP_ratio=1.5,
            num_iters=(2, 2, 8),
            random_state=11,
            knn_method="brute",
            low_dist_thres=6.0,
        )

        emb_a = LocalMAP(**kwargs).fit_transform(self.X, init="random")
        emb_b = LocalMAP(**kwargs).fit_transform(self.X, init="random")

        self.assertTrue(np.allclose(emb_a, emb_b, atol=1e-3))

    def test_initial_fp_sampling_excludes_neighbours(self):
        n = 12
        n_neighbors = 3
        n_fp = 4

        pair_neighbors = np.empty((n * n_neighbors, 2), dtype=np.int32)
        for i in range(n):
            for j in range(n_neighbors):
                idx = i * n_neighbors + j
                pair_neighbors[idx, 0] = i
                pair_neighbors[idx, 1] = (i + j + 1) % n

        sampled = _sample_FP_pairs(
            n=n,
            pair_neighbors_np=pair_neighbors,
            n_neighbors=n_neighbors,
            n_FP=n_fp,
            rng=np.random.default_rng(1),
        )

        self.assertEqual(sampled.shape, (n * n_fp, 2))
        neighbour_rows = pair_neighbors[:, 1].reshape(n, n_neighbors)

        for i in range(n):
            row = sampled[i * n_fp:(i + 1) * n_fp, 1]
            reject = set(neighbour_rows[i].tolist()) | {i}
            self.assertEqual(len(np.unique(row)), n_fp)
            self.assertTrue(all(int(dst) not in reject for dst in row))

    def test_local_fp_resampling_uses_distance_threshold(self):
        n = 6
        n_neighbors = 2
        n_fp = 1

        pair_neighbors = np.empty((n * n_neighbors, 2), dtype=np.int32)
        for i in range(n):
            pair_neighbors[i * n_neighbors + 0] = (i, (i + 1) % n)
            pair_neighbors[i * n_neighbors + 1] = (i, (i + 2) % n)

        pair_fp = np.array(
            [
                [0, 5],
                [1, 4],
                [2, 5],
                [3, 0],
                [4, 1],
                [5, 2],
            ],
            dtype=np.int32,
        )

        Y = np.array(
            [
                [0.0, 0.0],
                [0.1, 0.0],
                [0.2, 0.0],
                [0.9, 0.0],
                [2.0, 0.0],
                [3.0, 0.0],
            ],
            dtype=np.float32,
        )

        low = _resample_local_fp_pairs(
            pair_neighbors_np=pair_neighbors,
            pair_FP_np=pair_fp,
            Y_mx=mx.array(Y),
            low_dist_thres=0.3,
            rng=np.random.default_rng(123),
        )
        high = _resample_local_fp_pairs(
            pair_neighbors_np=pair_neighbors,
            pair_FP_np=pair_fp,
            Y_mx=mx.array(Y),
            low_dist_thres=1.0,
            rng=np.random.default_rng(123),
        )

        self.assertEqual(low[0, 1], 5)
        self.assertEqual(high[0, 1], 3)

    def test_local_adjustment_changes_phase_three_embedding(self):
        common = dict(
            n_components=2,
            n_neighbors=8,
            MN_ratio=0.5,
            FP_ratio=1.5,
            num_iters=(2, 2, 12),
            random_state=3,
            knn_method="brute",
        )

        pacmap_embedding = PaCMAP(**common).fit_transform(self.X, init="random")
        localmap_embedding = LocalMAP(low_dist_thres=4.0, **common).fit_transform(self.X, init="random")

        self.assertGreater(np.max(np.abs(localmap_embedding - pacmap_embedding)), 1e-5)


if __name__ == "__main__":
    unittest.main()
