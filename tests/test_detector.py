import unittest

import numpy as np

from sbd.detector import (
    DetectorConfig,
    cuts_to_shots,
    detect,
    detect_gradual_transitions,
    detect_hard_cuts,
    robust_threshold,
)
from sbd.metrics import match_events
from sbd.synth import hist_scenario


TOL = 3


class TestDetector(unittest.TestCase):
    def test_hard_cuts(self):
        data = hist_scenario("cuts", seed=7)
        cuts = detect(data["hists"], DetectorConfig(), data["fps"])
        pairs, fp, fn = match_events([c.frame for c in cuts], data["cuts"], tol=TOL)
        self.assertEqual(len(pairs), len(data["cuts"]))
        self.assertEqual(fp, [])
        self.assertEqual(fn, [])

    def test_single_shot_no_false_cuts(self):
        """单镜头负样本：这是整套设计的核心回归测试。"""
        data = hist_scenario("single", seed=3)
        cuts = detect(data["hists"], DetectorConfig(), data["fps"])
        self.assertEqual(cuts, [], f"单镜头视频被误切：{[c.frame for c in cuts]}")

    def test_single_shot_without_absolute_floor_fails(self):
        """反证：去掉绝对下限、关掉持续性检验，纯相对阈值必然产生假切点。"""
        data = hist_scenario("single", seed=3)
        bad = DetectorConfig(t_abs=0.0, check_persistence=False, suppress_flash=False,
                             detect_gradual=False)
        cuts = detect(data["hists"], bad, data["fps"])
        self.assertGreater(len(cuts), 0, "这个用例本应暴露相对阈值的缺陷")

    def test_persistence_check_is_what_saves_us(self):
        """只去掉绝对下限、保留持续性检验，单镜头视频依然不应该被切。"""
        data = hist_scenario("single", seed=3)
        cfg = DetectorConfig(t_abs=0.0, check_persistence=True,
                             suppress_flash=False, detect_gradual=False)
        self.assertEqual(detect(data["hists"], cfg, data["fps"]), [])

    def test_flash_is_not_a_cut(self):
        data = hist_scenario("flash", seed=5)
        cuts = detect(data["hists"], DetectorConfig(), data["fps"])
        self.assertEqual(cuts, [], f"闪光被误判：{[c.frame for c in cuts]}")

    def test_flash_suppression_disabled_detects_it(self):
        """确保闪光用例本身是有区分度的：关掉抑制就应该检出。"""
        data = hist_scenario("flash", seed=5)
        cfg = DetectorConfig(suppress_flash=False, check_persistence=False,
                             detect_gradual=False)
        cuts = detect_hard_cuts(data["hists"], cfg, data["fps"])
        self.assertGreater(len(cuts), 0)

    def test_dissolve_is_detected_as_gradual(self):
        data = hist_scenario("dissolve", seed=11)
        cuts = detect(data["hists"], DetectorConfig(), data["fps"])
        self.assertEqual(len(cuts), 1)
        self.assertEqual(cuts[0].kind, "gradual")
        self.assertLessEqual(abs(cuts[0].frame - data["cuts"][0]), TOL)

    def test_gradual_detection_can_be_disabled(self):
        data = hist_scenario("dissolve", seed=11)
        cfg = DetectorConfig(detect_gradual=False)
        self.assertEqual(detect(data["hists"], cfg, data["fps"]), [])

    def test_min_shot_length(self):
        """两个间隔 5 帧的候选，只应保留一个。"""
        data = hist_scenario("cuts", lengths=(60, 60, 60), seed=2)
        cfg = DetectorConfig(min_shot_len_s=0.5)  # 25fps -> 12 帧
        cuts = detect_hard_cuts(data["hists"], cfg, 25.0)
        gaps = [b - a for a, b in zip([c.frame for c in cuts], [c.frame for c in cuts][1:])]
        self.assertTrue(all(g >= 12 for g in gaps))

    def test_empty_and_tiny_inputs(self):
        cfg = DetectorConfig()
        self.assertEqual(detect(np.zeros((0, 8)), cfg, 25.0), [])
        self.assertEqual(detect(np.zeros((2, 8)), cfg, 25.0), [])

    def test_cuts_to_shots(self):
        data = hist_scenario("cuts", seed=7)
        cuts = detect(data["hists"], DetectorConfig(), data["fps"])
        shots = cuts_to_shots(cuts, len(data["hists"]))
        self.assertEqual(shots[0][0], 0)
        self.assertEqual(shots[-1][1], len(data["hists"]))
        for (s1, e1), (s2, e2) in zip(shots, shots[1:]):
            self.assertEqual(e1, s2)
            self.assertGreater(e1, s1)


class TestRobustThreshold(unittest.TestCase):
    def test_spike_does_not_inflate_threshold(self):
        d = np.full(101, 0.02)
        d[50] = 0.9
        thr = robust_threshold(d, window=10, k=6.0)
        self.assertLess(thr[50], 0.1, "鲁棒阈值被切点自身抬高了")

    def test_constant_signal_threshold_is_constant(self):
        d = np.full(50, 0.05)
        thr = robust_threshold(d, window=5, k=3.0)
        self.assertTrue(np.allclose(thr, 0.05))

    def test_gradual_requires_quiet_afterwards(self):
        """持续运动（结束不静默）不应被判为渐变。"""
        rng = np.random.default_rng(0)
        n = 100
        base = rng.dirichlet(np.ones(32))
        hists = []
        for i in range(n):
            other = rng.dirichlet(np.ones(32))
            hists.append(0.9 * base + 0.1 * other)
            base = 0.98 * base + 0.02 * other   # 一直变，从不静默
        hists = np.vstack([h / h.sum() for h in hists])
        cuts = detect_gradual_transitions(hists, DetectorConfig(), 25.0)
        self.assertEqual(cuts, [])


if __name__ == "__main__":
    unittest.main()
