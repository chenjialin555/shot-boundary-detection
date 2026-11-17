import unittest

import numpy as np

from sbd.verify import (
    VerifyConfig,
    block_hist_distance,
    edge_change_ratio,
    ssim,
    to_gray,
    verify_pair,
)


class TestVerify(unittest.TestCase):
    def test_identical_frames(self):
        rng = np.random.default_rng(0)
        frame = rng.integers(0, 256, (48, 64, 3), dtype=np.uint8)
        self.assertAlmostEqual(ssim(frame, frame), 1.0, places=6)
        self.assertAlmostEqual(edge_change_ratio(frame, frame), 0.0, places=6)
        self.assertAlmostEqual(block_hist_distance(frame, frame), 0.0, places=9)

    def test_different_content(self):
        rng = np.random.default_rng(1)
        a = rng.integers(0, 256, (48, 64, 3), dtype=np.uint8)
        b = rng.integers(0, 256, (48, 64, 3), dtype=np.uint8)
        self.assertLess(ssim(a, b), 0.5)
        self.assertGreater(block_hist_distance(a, b), 0.3)
        ev = verify_pair(a, b)
        self.assertTrue(ev["is_cut"], ev)

    def test_same_scene_with_noise_is_not_a_cut(self):
        rng = np.random.default_rng(2)
        base = rng.integers(40, 200, (48, 64, 3), dtype=np.uint8)
        a = np.clip(base + rng.normal(0, 3, base.shape), 0, 255).astype(np.uint8)
        b = np.clip(base + rng.normal(0, 3, base.shape), 0, 255).astype(np.uint8)
        ev = verify_pair(a, b)
        self.assertFalse(ev["is_cut"], ev)

    def test_gray_conversion(self):
        rgb = np.zeros((2, 2, 3), dtype=np.uint8)
        rgb[..., 0] = 255
        self.assertAlmostEqual(float(to_gray(rgb)[0, 0]), 0.299 * 255, places=3)

    def test_shape_mismatch(self):
        a = np.zeros((10, 10, 3), dtype=np.uint8)
        b = np.zeros((10, 12, 3), dtype=np.uint8)
        self.assertEqual(ssim(a, b), 0.0)
        self.assertEqual(block_hist_distance(a, b), 0.0)

    def test_votes_and_config(self):
        rng = np.random.default_rng(3)
        a = rng.integers(0, 256, (40, 40, 3), dtype=np.uint8)
        b = rng.integers(0, 256, (40, 40, 3), dtype=np.uint8)
        strict = VerifyConfig(need=3, ssim_max=0.1, ecr_min=0.9, block_min=0.9)
        self.assertFalse(verify_pair(a, b, strict)["is_cut"])


if __name__ == "__main__":
    unittest.main()
