import json
import os
import tempfile
import unittest

import numpy as np

from sbd.metrics import (
    aggregate,
    evaluate,
    format_report,
    load_ground_truth,
    match_events,
    scan_dataset,
)


class TestMatch(unittest.TestCase):
    def test_tolerance(self):
        pairs, fp, fn = match_events([100, 300], [101, 305], tol=2)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(fp, [300])
        self.assertEqual(fn, [305])

    def test_one_to_one(self):
        pairs, fp, fn = match_events([10], [10, 11], tol=2)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(len(fn), 1)

    def test_greedy_prefers_nearest(self):
        pairs, fp, fn = match_events([12, 9], [10], tol=5)
        self.assertEqual(pairs, [(10, 9)])


class TestEvaluate(unittest.TestCase):
    def test_positive_clip(self):
        r = evaluate([100, 400], [100, 400], fps=25, n_frames=500, tol=2)
        self.assertEqual((r["tp"], r["fp"], r["fn"]), (2, 0, 0))
        self.assertEqual(r["f1"], 1.0)

    def test_negative_clip_reports_far_not_f1(self):
        r = evaluate([100], [], fps=25, n_frames=1500, tol=2)  # 60 秒
        self.assertIsNone(r["f1"])
        self.assertFalse(r["f1_defined"])
        self.assertEqual(r["fp"], 1)
        self.assertAlmostEqual(r["far_per_min"], 1.0, places=3)

    def test_missed_cut(self):
        r = evaluate([], [100], fps=25, n_frames=250, tol=2)
        self.assertIsNone(r["precision"])
        self.assertEqual(r["recall"], 0.0)
        self.assertIsNone(r["f1"])

    def test_duration_from_frames(self):
        r = evaluate([], [], fps=20, n_frames=1200)
        self.assertAlmostEqual(r["duration_min"], 1.0)


class TestAggregate(unittest.TestCase):
    def test_mixed_dataset(self):
        rs = [
            evaluate([100, 400], [100, 400], fps=25, n_frames=500, name="a"),
            evaluate([200], [201], fps=25, n_frames=500, name="b"),
            evaluate([50], [], fps=25, n_frames=1500, name="single"),
        ]
        agg = aggregate(rs)
        self.assertEqual(agg["n_positive"], 2)
        self.assertEqual(agg["n_negative"], 1)
        self.assertEqual(agg["negative_false_alarms"], 1)
        self.assertEqual(agg["clean_negative_rate"], 0.0)
        self.assertIsNotNone(agg["f1"])
        self.assertIsNotNone(agg["negative_far_per_min"])

    def test_all_negative(self):
        rs = [evaluate([], [], fps=25, n_frames=1500) for _ in range(3)]
        agg = aggregate(rs)
        self.assertIsNone(agg["f1"])
        self.assertEqual(agg["negative_false_alarms"], 0)
        self.assertEqual(agg["clean_negative_rate"], 1.0)

    def test_report_mentions_far(self):
        agg = aggregate([evaluate([10], [], fps=25, n_frames=1500)])
        text = format_report(agg)
        self.assertIn("FAR", text)


class TestGroundTruthIO(unittest.TestCase):
    def test_formats(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "a.json")
            for payload, expected in [
                ({"cuts": [10, 20]}, [10, 20]),
                ([10, 20], [10, 20]),
                ({"shots": [[0, 10], [10, 25]]}, [10]),
            ]:
                with open(p, "w", encoding="utf-8") as f:
                    json.dump(payload, f)
                self.assertEqual(load_ground_truth(p)["cuts"], expected)

    def test_scan_dataset_with_sidecar(self):
        with tempfile.TemporaryDirectory() as tmp:
            open(os.path.join(tmp, "clip.mp4"), "wb").close()
            with open(os.path.join(tmp, "clip.json"), "w", encoding="utf-8") as f:
                json.dump({"cuts": [7]}, f)
            entries = scan_dataset(tmp)
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0]["cuts"], [7])


if __name__ == "__main__":
    unittest.main()
