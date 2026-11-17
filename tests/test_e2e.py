import json
import os
import tempfile
import unittest

from sbd.detector import DetectorConfig, detect
from sbd.synth import rgb_scenario, write_png_sequence
from sbd.video import grab_frames, probe, read_hists

try:
    import PIL  # noqa: F401

    HAS_PIL = True
except Exception:  # pragma: no cover
    HAS_PIL = False


@unittest.skipUnless(HAS_PIL, "需要 Pillow 才能跑图片序列端到端测试")
class TestEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        data = rgb_scenario("cuts", lengths=(30, 25, 30), seed=4)
        cls.truth = data["cuts"]
        cls.dir = os.path.join(cls.tmp.name, "frames")
        write_png_sequence(cls.dir, data["frames"])

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_probe_image_sequence(self):
        info = probe(self.dir)
        self.assertEqual(info.backend, "images")
        self.assertEqual(info.n_frames, 85)
        self.assertEqual((info.width, info.height), (96, 72))

    def test_detect_from_png_sequence(self):
        cfg = DetectorConfig()
        hists, info = read_hists(self.dir, cfg, max_width=96)
        self.assertEqual(len(hists), 85)
        self.assertAlmostEqual(float(hists[0].sum()), 1.0)
        cuts = detect(hists, cfg, info.fps)
        frames = [c.frame for c in cuts]
        self.assertEqual(len(frames), len(self.truth), f"期望 {self.truth}，得到 {frames}")
        for f, t in zip(frames, self.truth):
            self.assertLessEqual(abs(f - t), 2)

    def test_grab_frames(self):
        frames = grab_frames(self.dir, [0, 30, 84], max_width=96)
        self.assertEqual(sorted(frames), [0, 30, 84])
        self.assertEqual(frames[30].shape, (72, 96, 3))


@unittest.skipUnless(HAS_PIL, "需要 Pillow")
class TestCli(unittest.TestCase):
    def test_detect_cli_writes_json(self):
        from sbd.cli import main

        with tempfile.TemporaryDirectory() as tmp:
            data = rgb_scenario("cuts", lengths=(30, 25, 30), seed=4)
            seq = os.path.join(tmp, "frames")
            write_png_sequence(seq, data["frames"])
            out = os.path.join(tmp, "cuts.json")
            rc = main(["detect", seq, "-o", out, "--min-shot", "0.2", "--max-width", "96"])
            self.assertEqual(rc, 0)
            with open(out, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.assertEqual(len(payload["cuts"]), 2)
            self.assertEqual(len(payload["shots"]), 3)
            self.assertEqual(payload["shots"][0][0], 0)
            self.assertEqual(payload["shots"][-1][1], len(data["frames"]))

    def test_calibrate_cli(self):
        from sbd.cli import main

        with tempfile.TemporaryDirectory() as tmp:
            ds = os.path.join(tmp, "ds")
            os.makedirs(ds)
            for name, kind, lengths in [("clip01", "cuts", (30, 25, 30)),
                                        ("single01", "single", (60,))]:
                data = rgb_scenario(kind, lengths=lengths, seed=4)
                seq = os.path.join(ds, name)
                write_png_sequence(seq, data["frames"])
                with open(os.path.join(ds, name + ".json"), "w", encoding="utf-8") as f:
                    json.dump({"cuts": data["cuts"], "fps": 25}, f)
            out = os.path.join(tmp, "calib.json")
            rc = main(["calibrate", ds, "--max-width", "96", "-o", out])
            self.assertEqual(rc, 0)
            with open(out, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.assertGreater(payload["suggested_t_abs"], 0)
            self.assertEqual(payload["n_clips"], 2)

    def test_demo_cli(self):
        from sbd.cli import main

        self.assertEqual(main(["demo"]), 0)


if __name__ == "__main__":
    unittest.main()
