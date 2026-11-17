"""命令行入口：``python -m sbd <detect|eval|calibrate|demo> ...``"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np

from .changepoint import bic_penalty, pelt
from .detector import (
    Cut,
    DetectorConfig,
    cuts_to_shots,
    detect,
    detect_hard_cuts,
    min_shot_frames,
)
from .distance import diff_series
from .metrics import (
    aggregate,
    evaluate,
    format_report,
    load_ground_truth,
    scan_dataset,
)
from .synth import hist_scenario
from .video import grab_frames, probe, read_hists


# --------------------------------------------------------------------------- #
# 参数装配
# --------------------------------------------------------------------------- #

def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--space", default="hsv", choices=["hsv", "rgb", "gray"],
                   help="色彩空间，默认 hsv")
    p.add_argument("--bins", default="18,4,4", help="分 bin 数，如 18,4,4 或 8,8,8")
    p.add_argument("--metric", default="l1",
                   choices=["l1", "intersection", "l2", "chi2", "bhattacharyya", "cosine"])
    p.add_argument("--t-abs", type=float, default=None,
                   help="绝对下限；不填用默认 0.38（换 bins 后务必重标定）")
    p.add_argument("--min-shot", type=float, default=None, help="最短镜头（秒）")
    p.add_argument("--no-gradual", action="store_true", help="关闭渐变转场检测")
    p.add_argument("--max-width", type=int, default=320, help="解码时下采样宽度，默认 320")


def _config_from_args(args: argparse.Namespace) -> DetectorConfig:
    cfg = DetectorConfig()
    cfg.space = args.space
    cfg.bins = tuple(int(x) for x in str(args.bins).split(","))
    cfg.metric = args.metric
    if args.t_abs is not None:
        cfg.t_abs = float(args.t_abs)
    if args.min_shot is not None:
        cfg.min_shot_len_s = float(args.min_shot)
    if getattr(args, "no_gradual", False):
        cfg.detect_gradual = False
    return cfg


def _cuts_from_changepoint(hists, cfg: DetectorConfig, fps: float) -> List[Cut]:
    d = diff_series(hists, cfg.metric)
    breaks = pelt(hists, penalty=bic_penalty(hists), min_size=min_shot_frames(cfg, fps))
    return [
        Cut(frame=int(b), score=float(d[b]) if b < len(d) else 0.0, kind="hard")
        for b in breaks
    ]


# --------------------------------------------------------------------------- #
# detect
# --------------------------------------------------------------------------- #

def cmd_detect(args: argparse.Namespace) -> int:
    cfg = _config_from_args(args)
    if not os.path.exists(args.video):
        print(f"找不到输入：{args.video}", file=sys.stderr)
        return 2

    hists, info = read_hists(args.video, cfg, max_width=args.max_width)
    fps = info.fps or 25.0

    if args.mode == "changepoint":
        cuts = _cuts_from_changepoint(hists, cfg, fps)
    elif args.mode == "hybrid":
        cuts = detect(hists, cfg, fps)
        extra = _cuts_from_changepoint(hists, cfg, fps)
        length = min_shot_frames(cfg, fps)
        for e in extra:
            if all(abs(e.frame - c.frame) >= length for c in cuts):
                cuts.append(e)
        cuts.sort(key=lambda c: c.frame)
    else:
        cuts = detect(hists, cfg, fps)

    verification: List[Dict] = []
    if args.verify and cuts:
        frames = grab_frames(args.video, sorted({c.frame for c in cuts}
                                                | {c.frame - 1 for c in cuts}),
                             max_width=args.verify_width)
        kept: List[Cut] = []
        for c in cuts:
            prev, cur = frames.get(c.frame - 1), frames.get(c.frame)
            if prev is None or cur is None:
                kept.append(c)
                continue
            from .verify import verify_pair

            ev = verify_pair(prev, cur)
            verification.append({"frame": c.frame, **ev})
            if ev["is_cut"] or args.verify_soft:
                c.detail.update({"ssim": ev["ssim"], "ecr": ev["ecr"], "block": ev["block"]})
                kept.append(c)
        cuts = kept

    n_frames = info.n_frames or len(hists)
    shots = cuts_to_shots(cuts, n_frames)
    result = {
        "video": os.path.abspath(args.video),
        "fps": round(float(fps), 4),
        "n_frames": int(n_frames),
        "duration_s": round(n_frames / fps, 3) if fps else None,
        "mode": args.mode,
        "config": cfg.to_dict(),
        "cuts": [c.to_dict() for c in cuts],
        "shots": [[int(s), int(e)] for s, e in shots],
    }
    if verification:
        result["verification"] = verification

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    print(f"视频      : {result['video']}")
    print(f"帧率/帧数 : {fps:.3f} fps, {n_frames} 帧, {result['duration_s']} s")
    print(f"检测模式  : {args.mode}")
    if not cuts:
        print("切点      : 未检测到镜头切换（单镜头视频）")
    else:
        print(f"切点      : {len(cuts)} 个，切出 {len(shots)} 个镜头")
        for i, c in enumerate(cuts, 1):
            t = c.frame / fps
            tag = "硬切" if c.kind == "hard" else "渐变"
            extra = ""
            if "ssim" in c.detail:
                extra = f"  ssim={c.detail['ssim']:.3f} ecr={c.detail['ecr']:.3f}"
            print(f"  #{i:<3d} 帧 {c.frame:<7d} t={t:8.3f}s  {tag}  score={c.score:.3f}{extra}")
    if args.output:
        print(f"结果已写入: {args.output}")
    return 0


# --------------------------------------------------------------------------- #
# eval
# --------------------------------------------------------------------------- #

def cmd_eval(args: argparse.Namespace) -> int:
    cfg = _config_from_args(args)
    entries = scan_dataset(args.dataset)
    if not entries:
        print(f"数据集里没有找到「视频 + 同名 .json 标注」：{args.dataset}", file=sys.stderr)
        return 2

    results = []
    for e in entries:
        try:
            hists, info = read_hists(e["video"], cfg, max_width=args.max_width)
        except Exception as exc:  # noqa: BLE001
            print(f"  [跳过] {e['name']}: {exc}", file=sys.stderr)
            continue
        fps = e.get("fps") or info.fps or 25.0
        if args.mode == "changepoint":
            cuts = _cuts_from_changepoint(hists, cfg, fps)
        else:
            cuts = detect(hists, cfg, fps)
        r = evaluate(
            [c.frame for c in cuts], e["cuts"], fps=fps, n_frames=info.n_frames,
            tol=args.tol, name=e["name"], duration_s=e.get("duration_s"),
        )
        results.append(r)
        f1_text = "  n/a" if r["f1"] is None else f"{r['f1']:.3f}"
        print(f"  {e['name']:<28s} gt={r['n_gt']:<4d} pred={r['n_pred']:<4d} "
              f"tp={r['tp']:<4d} fp={r['fp']:<4d} fn={r['fn']:<4d} F1={f1_text}")

    agg = aggregate(results)
    print()
    print(format_report(agg))

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump({"config": cfg.to_dict(), "aggregate": agg, "clips": results},
                      f, ensure_ascii=False, indent=2)
        print(f"\n详细结果: {args.output}")
    return 0


# --------------------------------------------------------------------------- #
# calibrate
# --------------------------------------------------------------------------- #

def _auc(pos: np.ndarray, neg: np.ndarray) -> Optional[float]:
    if pos.size == 0 or neg.size == 0:
        return None
    allv = np.concatenate([neg, pos])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(len(allv), dtype=np.float64)
    ranks[order] = np.arange(1, len(allv) + 1)
    r_pos = ranks[neg.size:].sum()
    return float((r_pos - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


def cmd_calibrate(args: argparse.Namespace) -> int:
    cfg = _config_from_args(args)
    entries = scan_dataset(args.dataset)
    if not entries:
        print("没有找到可标定的片段（需要视频 + 同名 .json）", file=sys.stderr)
        return 2

    h0: List[float] = []
    h1: List[float] = []
    per_negative_max: List[float] = []

    for e in entries:
        hists, info = read_hists(e["video"], cfg, max_width=args.max_width)
        d = diff_series(hists, cfg.metric)
        gt = np.array(e["cuts"], dtype=int)
        mask = np.ones(len(d), dtype=bool)
        mask[0] = False
        for g in gt:
            lo, hi = max(0, g - args.exclude), min(len(d), g + args.exclude + 1)
            mask[lo:hi] = False
        h0.extend(d[mask].tolist())

        for g in gt:
            if 1 <= g < len(d):
                lo, hi = max(1, g - 1), min(len(d), g + 2)
                h1.append(float(d[lo:hi].max()))
        if len(gt) == 0 and len(d) > 1:
            per_negative_max.append(float(d[1:].max()))

    h0a = np.asarray(h0, dtype=np.float64)
    h1a = np.asarray(h1, dtype=np.float64)

    alpha = float(args.alpha)
    q_alpha = float(np.quantile(h0a, 1.0 - alpha)) if h0a.size else 0.0
    h1_min = float(np.quantile(h1a, 0.05)) if h1a.size else 1.0
    suggested = float(np.clip(0.5 * (q_alpha + h1_min), 0.05, 0.95)) if h1a.size else q_alpha

    report = {
        "metric": cfg.metric,
        "space": cfg.space,
        "bins": list(cfg.bins),
        "n_clips": len(entries),
        "n_cut_pairs": int(h1a.size),
        "n_same_shot_pairs": int(h0a.size),
        "h0_same_shot": {
            "p50": round(float(np.quantile(h0a, 0.5)), 4) if h0a.size else None,
            "p99": round(float(np.quantile(h0a, 0.99)), 4) if h0a.size else None,
            "p999": round(float(np.quantile(h0a, 0.999)), 4) if h0a.size else None,
            "max": round(float(h0a.max()), 4) if h0a.size else None,
        },
        "h1_cut": {
            "min": round(float(h1a.min()), 4) if h1a.size else None,
            "p05": round(h1_min, 4) if h1a.size else None,
            "p50": round(float(np.quantile(h1a, 0.5)), 4) if h1a.size else None,
            "max": round(float(h1a.max()), 4) if h1a.size else None,
        },
        "auc": round(_auc(h1a, h0a), 4) if _auc(h1a, h0a) is not None else None,
        "negative_clip_max": round(float(max(per_negative_max)), 4) if per_negative_max else None,
        "suggested_t_abs": round(suggested, 4),
        "per_frame_alpha": alpha,
    }

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\n已写入: {args.output}")

    if report["h1_cut"]["min"] is not None and report["h0_same_shot"]["p999"] is not None:
        if suggested > report["h1_cut"]["min"]:
            print("\n⚠ 同镜头噪声尾部已经盖住了最弱的切点：单纯靠直方图无解，"
                  "建议开启持续性检验 / 空间校验，或缩小 bins。")
    return 0


# --------------------------------------------------------------------------- #
# demo
# --------------------------------------------------------------------------- #

def cmd_demo(args: argparse.Namespace) -> int:
    cfg = DetectorConfig()
    cases = ["cuts", "single", "flash", "dissolve"]
    print(f"{'场景':<10}{'真值':<8}{'检测':<8}{'命中':<6}{'误检':<6}{'漏检':<6}详情")
    print("-" * 78)
    for kind in cases:
        data = hist_scenario(kind, seed=7)
        hists = data["hists"]
        cuts = detect(hists, cfg, data["fps"])
        from .metrics import match_events

        pairs, fps_list, fns = match_events([c.frame for c in cuts], data["cuts"], tol=3)
        print(f"{kind:<10}{len(data['cuts']):<8}{len(cuts):<8}"
              f"{len(pairs):<6}{len(fps_list):<6}{len(fns):<6}")
        for c in cuts:
            print(f"    └ 帧 {c.frame:>4d}  {c.kind:<8} score={c.score:.3f} {c.detail}")
    print("-" * 78)
    print("说明：single / flash 的真值是 0 个切点，理想输出就是「未检测到」。")
    return 0


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="sbd",
        description="基于直方图的视频分镜（镜头边界）检测：绝对下限 + 持续性检验 + 变点检测",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("detect", help="检测单个视频的切点")
    d.add_argument("video", help="视频文件，或图片序列目录")
    d.add_argument("-o", "--output", help="输出 JSON 路径")
    d.add_argument("--json", action="store_true", help="直接打印 JSON")
    d.add_argument("--mode", default="threshold", choices=["threshold", "changepoint", "hybrid"])
    d.add_argument("--verify", action="store_true", help="启用空间校验（需能解码视频）")
    d.add_argument("--verify-soft", action="store_true", help="空间校验只记录不剔除")
    d.add_argument("--verify-width", type=int, default=640, help="校验取帧宽度")
    _add_common(d)
    d.set_defaults(func=cmd_detect)

    e = sub.add_parser("eval", help="批量评估（视频 + 同名 .json 标注）")
    e.add_argument("dataset", help="数据集目录")
    e.add_argument("--tol", type=int, default=2, help="匹配容差（帧），默认 ±2")
    e.add_argument("-o", "--output", help="输出 JSON 路径")
    e.add_argument("--mode", default="threshold", choices=["threshold", "changepoint", "hybrid"])
    _add_common(e)
    e.set_defaults(func=cmd_eval)

    c = sub.add_parser("calibrate", help="标定绝对阈值 t_abs")
    c.add_argument("dataset", help="数据集目录（含单镜头负样本最好）")
    c.add_argument("--alpha", type=float, default=1e-4, help="每帧虚警率，默认 1e-4")
    c.add_argument("--exclude", type=int, default=5, help="切点前后排除的帧数")
    c.add_argument("-o", "--output", help="输出 JSON 路径")
    _add_common(c)
    c.set_defaults(func=cmd_calibrate)

    m = sub.add_parser("demo", help="跑内置合成场景（无需视频）")
    m.set_defaults(func=cmd_demo)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
