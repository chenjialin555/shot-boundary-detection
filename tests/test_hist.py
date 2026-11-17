import unittest

import numpy as np

from sbd.distance import l1
from sbd.hist import (
    frame_hist,
    hist_gray,
    hist_hsv,
    hist_rgb,
    normalize,
    rgb_to_hsv,
    smooth_hist,
)


class TestColorConversion(unittest.TestCase):
    def test_primaries(self):
        rgb = np.zeros((2, 2, 3), dtype=np.uint8)
        rgb[..., 0] = 255                       # 纯红
        hsv = rgb_to_hsv(rgb)
        self.assertAlmostEqual(float(hsv[0, 0, 0]), 0.0, places=3)
        self.assertAlmostEqual(float(hsv[0, 0, 1]), 1.0, places=3)

        rgb = np.zeros((2, 2, 3), dtype=np.uint8)
        rgb[..., 1] = 255                       # 纯绿
        self.assertAlmostEqual(float(rgb_to_hsv(rgb)[0, 0, 0]), 120.0, places=3)

        rgb = np.zeros((2, 2, 3), dtype=np.uint8)
        rgb[..., 2] = 255                       # 纯蓝
        self.assertAlmostEqual(float(rgb_to_hsv(rgb)[0, 0, 0]), 240.0, places=3)

    def test_gray_has_no_saturation(self):
        rgb = np.full((3, 3, 3), 128, dtype=np.uint8)
        hsv = rgb_to_hsv(rgb)
        self.assertTrue(np.allclose(hsv[..., 1], 0.0))

    def test_shape_validation(self):
        with self.assertRaises(ValueError):
            rgb_to_hsv(np.zeros((4, 4)))


class TestHistogram(unittest.TestCase):
    def test_normalized(self):
        rng = np.random.default_rng(0)
        frame = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
        for space in ("hsv", "rgb", "gray"):
            h = frame_hist(frame, space=space)
            self.assertAlmostEqual(float(h.sum()), 1.0, places=9)
            self.assertTrue(np.all(h >= 0))

    def test_identical_frames_zero_distance(self):
        rng = np.random.default_rng(1)
        frame = rng.integers(0, 256, (24, 24, 3), dtype=np.uint8)
        for space in ("hsv", "rgb", "gray"):
            a = frame_hist(frame, space=space)
            b = frame_hist(frame.copy(), space=space)
            self.assertEqual(l1(a, b), 0.0, space)

    def test_different_colors_large_distance(self):
        red = np.zeros((16, 16, 3), dtype=np.uint8)
        red[..., 0] = 220
        blue = np.zeros((16, 16, 3), dtype=np.uint8)
        blue[..., 2] = 220
        self.assertGreater(l1(frame_hist(red), frame_hist(blue)), 0.8)

    def test_hsv_hue_wraps_into_bins(self):
        bins = (18, 4, 4)
        rgb = np.zeros((8, 8, 3), dtype=np.uint8)
        rgb[..., 0] = 255                       # 纯红：H=0, S=1, V=1
        hist = hist_hsv(rgb_to_hsv(rgb), bins)
        # 索引 = (h * 4 + s) * 4 + v，饱和度和亮度都落在最后一个 bin
        self.assertAlmostEqual(float(hist[(0 * 4 + 3) * 4 + 3]), 1.0, places=6)

        rgb = np.zeros((8, 8, 3), dtype=np.uint8)
        rgb[..., 1] = 255                       # 纯绿：H=120 -> 第 6 个色相 bin
        hist = hist_hsv(rgb_to_hsv(rgb), bins)
        self.assertAlmostEqual(float(hist[(6 * 4 + 3) * 4 + 3]), 1.0, places=6)

    def test_gray_hist(self):
        g = np.zeros((10, 10), dtype=np.uint8)
        h = hist_gray(g, bins=16)
        self.assertAlmostEqual(float(h[0]), 1.0)
        self.assertEqual(h.size, 16)

    def test_uint8_rgb_vs_float(self):
        rng = np.random.default_rng(2)
        f = rng.integers(0, 256, (16, 16, 3)).astype(np.uint8)
        self.assertEqual(l1(hist_rgb(f), hist_rgb(f.astype(np.float64) / 255.0)), 0.0)

    def test_normalize_and_smooth(self):
        h = normalize(np.array([1.0, 3.0, 0.0]))
        self.assertAlmostEqual(float(h.sum()), 1.0)
        self.assertEqual(l1(smooth_hist(h), h) < 1.0, True)

    def test_empty_frame_is_uniform(self):
        h = hist_hsv(rgb_to_hsv(np.zeros((1, 1, 3), dtype=np.uint8)))
        self.assertAlmostEqual(float(h.sum()), 1.0)


if __name__ == "__main__":
    unittest.main()
