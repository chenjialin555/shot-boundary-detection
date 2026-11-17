import unittest

import numpy as np

from sbd.distance import (
    METRICS,
    bhattacharyya,
    chi2,
    cosine,
    diff_series,
    intersection,
    l1,
    l2,
    resolve,
)


def rand_hist(seed=0, dim=64):
    rng = np.random.default_rng(seed)
    h = rng.random(dim)
    return h / h.sum()


class TestDistance(unittest.TestCase):
    def setUp(self):
        self.a = rand_hist(1)
        self.b = rand_hist(2)

    def test_identical_is_zero(self):
        for name, fn in METRICS.items():
            if name in ("l2", "chi2"):
                continue
            self.assertEqual(fn(self.a, self.a), 0.0, name)

    def test_symmetry(self):
        for name, fn in METRICS.items():
            self.assertAlmostEqual(fn(self.a, self.b), fn(self.b, self.a), places=9, msg=name)

    def test_bounded_metrics(self):
        for fn in (l1, intersection, bhattacharyya, cosine):
            v = fn(self.a, self.b)
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0)

    def test_l1_known_value(self):
        a = np.array([1.0, 0.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0, 0.0])
        self.assertAlmostEqual(l1(a, b), 1.0)
        self.assertAlmostEqual(intersection(a, b), 1.0)

    def test_disjoint_hists(self):
        a = np.array([0.5, 0.5, 0.0, 0.0])
        b = np.array([0.0, 0.0, 0.5, 0.5])
        self.assertAlmostEqual(bhattacharyya(a, b), 1.0)

    def test_l2_and_chi2_positive(self):
        self.assertGreater(l2(self.a, self.b), 0)
        self.assertGreater(chi2(self.a, self.b), 0)

    def test_diff_series_matches_scalar(self):
        rng = np.random.default_rng(3)
        hists = rng.random((20, 32))
        hists /= hists.sum(axis=1, keepdims=True)
        for metric in ("l1", "intersection", "l2", "chi2", "bhattacharyya", "cosine"):
            d = diff_series(hists, metric)
            self.assertEqual(d[0], 0.0)
            fn = resolve(metric)
            for i in range(1, len(hists)):
                self.assertAlmostEqual(d[i], fn(hists[i - 1], hists[i]), places=9)

    def test_unknown_metric(self):
        with self.assertRaises(ValueError):
            resolve("nope")

    def test_empty_and_single(self):
        self.assertEqual(len(diff_series(np.zeros((0, 4)))), 0)
        d = diff_series(np.ones((1, 4)) / 4)
        self.assertEqual(len(d), 1)
        self.assertEqual(d[0], 0.0)


if __name__ == "__main__":
    unittest.main()
