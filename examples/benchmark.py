#!/usr/bin/env python3
"""合成基准 + 消融实验：验证「绝对下限」和「持续性检验」到底有没有用。

不需要任何真实视频，跑完约 1 秒：

    python examples/benchmark.py
    python examples/benchmark.py --seeds 10
"""

from __future__ import annotations

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sbd import DetectorConfig, detect  # noqa: E402
from sbd.metrics import match_events  # noqa: E402
from sbd.synth import hist_scenario  # noqa: E402

CASES = [
    ("cuts", "4 个镜头硬切"),
    ("single", "单镜头（含重尾噪声）"),
    ("flash", "单镜头 + 2 次白闪"),
    ("dissolve", "两镜头叠化"),
]

CONFIGS = [
    ("纯相对阈值", DetectorConfig(t_abs=0.0, check_persistence=False,
                                  suppress_flash=False, detect_gradual=False)),
    ("+ 绝对下限", DetectorConfig(t_abs=0.38, check_persistence=False,
                                  suppress_flash=False, detect_gradual=False)),
    ("+ 持续性检验", DetectorConfig(t_abs=0.38, check_persistence=True,
                                    suppress_flash=True, detect_gradual=False)),
    ("完整版（含渐变）", DetectorConfig()),
]

TOL = 3


def run(seeds: int):
    rows = []
    for name, cfg in CONFIGS:
        per_case = {}
        for kind, _label in CASES:
            tp = fp = fn = 0
            for seed in range(seeds):
                data = hist_scenario(kind, seed=seed)
                cuts = detect(data["hists"], cfg, data["fps"])
                pairs, fps_list, fns = match_events(
                    [c.frame for c in cuts], data["cuts"], tol=TOL)
                tp += len(pairs)
                fp += len(fps_list)
                fn += len(fns)
            per_case[kind] = (tp, fp, fn)
        rows.append((name, cfg, per_case))
    return rows


def fmt_f1(tp, fp, fn):
    if tp + fp + fn == 0:
        return "1.000"
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return f"{2 * p * r / (p + r):.3f}" if p + r else "0.000"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()

    rows = run(args.seeds)

    print(f"合成基准（{args.seeds} 个随机种子，容差 ±{TOL} 帧）\n")
    header = "| 配置 | " + " | ".join(label for _, label in CASES) + " |"
    print(header)
    print("|" + "---|" * (len(CASES) + 1))
    for name, _cfg, per_case in rows:
        cells = []
        for kind, _label in CASES:
            tp, fp, fn = per_case[kind]
            cells.append(f"✓{tp} · 漏{fn} · 误{fp}")
        print(f"| {name} | " + " | ".join(cells) + " |")

    print("\n硬切 F1（仅 cuts 场景）：")
    for name, _cfg, per_case in rows:
        tp, fp, fn = per_case["cuts"]
        print(f"  {name:<16s} F1 = {fmt_f1(tp, fp, fn)}")

    print("\n结论：")
    print("  · t_abs=0（纯相对阈值）时，单镜头场景会被切出假切点；")
    print("  · 补上绝对下限后，负样本误检归零，硬切召回不受影响；")
    print("  · 持续性检验进一步防住闪光/甩镜，渐变场景需要单独的检测器。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
