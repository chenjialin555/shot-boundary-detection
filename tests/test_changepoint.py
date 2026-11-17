import unittest

import numpy as np

from sbd.changepoint import bic_penalty, estimate_sigma2, pelt, segment_cost, segment_shots


def make_stream(n_segments=(60, 70, 50), dim=24, noise=0.002, seed=0):
    rng = np.random.default_rng(seed)
    parts, cuts, pos = [], [], 0
    for i, n in enumerate(n_segments):
        base = rng.dirichlet(np.full(dim, 0.5))
        if i > 0:
            cuts.append(pos)
        seg = np.clip(base + rng.normal(0, noise, (n, dim)), 0, None)
        seg /= seg.sum(axis=1, keepdims=True)
        parts.append(seg)
        pos += n
    return np.vstack(parts), cuts


class TestChangePoint(unittest.TestCase):
    def test_recovers_known_breaks(self):
        hists, cuts = make_stream()
        found = pelt(hists, penalty=bic_penalty(hists), min_size=10)
        self.assertEqual(len(found), len(cuts), f"期望 {cuts}，得到 {found}")
        for f, c in zip(found, cuts):
            self.assertLessEqual(abs(f - c), 3)

    def test_single_segment_returns_nothing(self):
        """单镜头视频：惩罚项必须让「0 个变点」成为最优解。"""
        hists, _ = make_stream(n_segments=(200,))
        self.assertEqual(pelt(hists, penalty=bic_penalty(hists), min_size=10), [])

    def test_tiny_input(self):
        self.assertEqual(pelt(np.zeros((3, 4))), [])

    def test_segment_cost(self):
        h = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
        from sbd.changepoint import _prefix

        p, q = _prefix(h)
        self.assertAlmostEqual(segment_cost(p, q, 0, 3), 0.0)
        h2 = np.array([[1.0, 0.0], [0.0, 1.0]])
        p2, q2 = _prefix(h2)
        self.assertAlmostEqual(segment_cost(p2, q2, 0, 2), 1.0)

    def test_sigma_estimate_ignores_cuts(self):
        hists, _ = make_stream(noise=0.001)
        s2 = estimate_sigma2(hists)
        self.assertGreater(s2, 0)
        self.assertLess(s2, 1e-2)

    def test_penalty_scales_with_dim(self):
        a, _ = make_stream(dim=8)
        b, _ = make_stream(dim=32)
        self.assertLess(bic_penalty(a), bic_penalty(b))

    def test_segment_shots_alias(self):
        hists, cuts = make_stream()
        self.assertEqual(segment_shots(hists, min_size=10), pelt(hists, penalty=bic_penalty(hists),
                                                                 min_size=10))


if __name__ == "__main__":
    unittest.main()
