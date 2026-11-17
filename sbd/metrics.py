"""评估：精确率 / 召回率 / F1，以及**单镜头视频必须看的 FAR**。

坑：当一个视频的 ground truth 切点数为 0 时，
``precision = 0``、``recall`` 无定义、``F1`` 无定义。
这时候 F1 完全失去意义，必须换成

    FAR = false alarms / minute   （每分钟误检次数）

否则「单镜头视频误检」这个问题在指标上根本不可见。
"""

from __future__ import annotations

import json
import os
from typing import Dict, Iterable, List, Optional, Sequence

VIDEO_EXT = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".mpg", ".mpeg"}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


# --------------------------------------------------------------------------- #
# 标注读取
# --------------------------------------------------------------------------- #

def load_ground_truth(path: str) -> Dict:
    """读取标注文件。

    支持三种写法::

        {"cuts": [120, 480], "fps": 25, "duration_s": 40}   # 推荐
        [120, 480]                                          # 最简
        {"shots": [[0,120],[120,480]]}                      # 镜头区间
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return _normalize_gt(data)


def _normalize_gt(data) -> Dict:
    if isinstance(data, list):
        if data and isinstance(data[0], (list, tuple)):
            shots = [tuple(int(x) for x in s) for s in data]
            cuts = [s[0] for s in shots[1:]]
            return {"cuts": cuts, "fps": None, "duration_s": None}
        return {"cuts": sorted(int(x) for x in data), "fps": None, "duration_s": None}
    if not isinstance(data, dict):
        raise ValueError("标注格式不认识：需要 dict 或 list")

    fps = data.get("fps")
    duration = data.get("duration_s")
    if "cuts" in data:
        cuts = sorted(int(x) for x in data["cuts"])
    elif "shots" in data:
        shots = [tuple(int(x) for x in s) for s in data["shots"]]
        cuts = [s[0] for s in shots[1:]]
    else:
        cuts = []
    return {"cuts": cuts, "fps": float(fps) if fps else None,
            "duration_s": float(duration) if duration else None}


def scan_dataset(root: str) -> List[Dict]:
    """扫描数据集目录，返回 ``[{"name","video","cuts","fps","duration_s"}]``。

    标注约定（二选一）：

    - 每个视频旁边放一个同名 ``.json``（``clip01.mp4`` + ``clip01.json``）；
    - 目录下放一个 ``labels.json``：``{"clips": [{"video": "...", "cuts": [...], "fps": 25}]}``。
    """
    root = os.path.abspath(root)
    entries: List[Dict] = []

    index_path = os.path.join(root, "labels.json")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        for item in payload.get("clips", payload if isinstance(payload, list) else []):
            video = item["video"]
            if not os.path.isabs(video):
                video = os.path.join(root, video)
            gt = _normalize_gt(item)
            entries.append({"name": os.path.splitext(os.path.basename(video))[0],
                            "video": video, **gt})
        return entries

    for name in sorted(os.listdir(root)):
        full = os.path.join(root, name)
        stem, ext = os.path.splitext(name)
        is_video = ext.lower() in VIDEO_EXT
        is_frames = os.path.isdir(full) and any(
            os.path.splitext(n)[1].lower() in IMAGE_EXT for n in os.listdir(full)
        )
        if not (is_video or is_frames):
            continue
        label = os.path.join(root, stem + ".json")
        if not os.path.exists(label):
            continue
        gt = load_ground_truth(label)
        entries.append({"name": stem, "video": full, **gt})
    return entries


# --------------------------------------------------------------------------- #
# 指标
# --------------------------------------------------------------------------- #

def match_events(
    pred: Sequence[int], gt: Sequence[int], tol: int = 2
) -> tuple:
    """带容差的一对一匹配（贪心，优先匹配最近的预测）。

    返回 ``(pairs, false_positives, false_negatives)``。
    """
    pred = [int(p) for p in pred]
    gt = [int(g) for g in gt]
    used = set()
    pairs = []
    matched_gt = set()
    for g in gt:
        best_j, best_d = None, tol + 1
        for j, p in enumerate(pred):
            if j in used:
                continue
            delta = abs(p - g)
            if delta <= tol and delta < best_d:
                best_j, best_d = j, delta
        if best_j is not None:
            used.add(best_j)
            matched_gt.add(g)
            pairs.append((g, pred[best_j]))
    fps_list = [p for j, p in enumerate(pred) if j not in used]
    fns = [g for g in gt if g not in matched_gt]
    return pairs, fps_list, fns


def evaluate(
    pred: Iterable[int],
    gt: Sequence[int],
    fps: float = 25.0,
    n_frames: Optional[int] = None,
    tol: int = 2,
    name: str = "",
    duration_s: Optional[float] = None,
) -> Dict:
    """单个视频的评估结果。"""
    pred = [int(p) for p in pred]
    gt = [int(g) for g in gt]
    pairs, fp_list, fn_list = match_events(pred, gt, tol)

    if duration_s is None:
        duration_s = (n_frames / fps) if (n_frames and fps) else 0.0
    minutes = duration_s / 60.0 if duration_s else 0.0

    tp = len(pairs)
    fp = len(fp_list)
    fn = len(fn_list)
    precision = tp / len(pred) if pred else None
    recall = tp / len(gt) if gt else None
    if precision is None or recall is None:
        f1 = None
    elif precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {
        "name": name,
        "n_gt": len(gt),
        "n_pred": len(pred),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": None if precision is None else round(precision, 4),
        "recall": None if recall is None else round(recall, 4),
        "f1": None if f1 is None else round(f1, 4),
        "f1_defined": f1 is not None,
        "duration_min": round(minutes, 4),
        "far_per_min": round(fp / minutes, 4) if minutes > 0 else None,
        "pred": pred,
        "gt": gt,
        "fp_frames": fp_list,
        "fn_frames": fn_list,
    }


def aggregate(results: Sequence[Dict]) -> Dict:
    """汇总：正样本算 micro P/R/F1，负样本（单镜头）单独算 FAR。"""
    pos = [r for r in results if r["n_gt"] > 0]
    neg = [r for r in results if r["n_gt"] == 0]

    tp = sum(r["tp"] for r in pos)
    fp = sum(r["fp"] for r in pos)
    fn = sum(r["fn"] for r in pos)
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    f1 = None
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)

    pos_minutes = sum(r["duration_min"] for r in pos)
    neg_minutes = sum(r["duration_min"] for r in neg)
    neg_fp = sum(r["fp"] for r in neg)
    total_fp = sum(r["fp"] for r in results)
    total_minutes = sum(r["duration_min"] for r in results)

    macro_f1 = None
    f1s = [r["f1"] for r in pos if r["f1"] is not None]
    if f1s:
        macro_f1 = float(sum(f1s) / len(f1s))

    return {
        "n_clips": len(results),
        "n_positive": len(pos),
        "n_negative": len(neg),
        "tp": tp, "fp": fp, "fn": fn,
        "precision": None if precision is None else round(precision, 4),
        "recall": None if recall is None else round(recall, 4),
        "f1": None if f1 is None else round(f1, 4),
        "macro_f1_on_positive": None if macro_f1 is None else round(macro_f1, 4),
        "positive_minutes": round(pos_minutes, 3),
        "negative_clips": len(neg),
        "negative_false_alarms": neg_fp,
        "negative_minutes": round(neg_minutes, 3),
        "negative_far_per_min": round(neg_fp / neg_minutes, 4) if neg_minutes > 0 else None,
        "clean_negative_rate": round(
            sum(1 for r in neg if r["fp"] == 0) / len(neg), 4) if neg else None,
        "overall_far_per_min": round(total_fp / total_minutes, 4) if total_minutes > 0 else None,
    }


def format_report(agg: Dict) -> str:
    """人读的汇总报告。"""
    def fmt(v, nd=4):
        if v is None:
            return "  n/a "
        if isinstance(v, float):
            return f"{v:.{nd}f}"
        return str(v)

    lines = []
    lines.append("=" * 62)
    lines.append("镜头边界检测 · 评估汇总")
    lines.append("=" * 62)
    lines.append(f"片段数            : {agg['n_clips']}  (正样本 {agg['n_positive']} / 负样本 {agg['n_negative']})")
    lines.append(f"TP / FP / FN      : {agg['tp']} / {agg['fp']} / {agg['fn']}")
    lines.append(f"Precision         : {fmt(agg['precision'])}")
    lines.append(f"Recall            : {fmt(agg['recall'])}")
    lines.append(f"F1 (micro)        : {fmt(agg['f1'])}")
    lines.append(f"F1 (macro, 正样本): {fmt(agg['macro_f1_on_positive'])}")
    lines.append("-" * 62)
    lines.append("单镜头（负样本）专项 —— 这才是真实体验的瓶颈")
    lines.append(f"负样本片段数      : {agg['negative_clips']}")
    lines.append(f"误检总数          : {agg['negative_false_alarms']}")
    lines.append(f"FAR (次/分钟)     : {fmt(agg['negative_far_per_min'])}")
    lines.append(f"零误检片段占比    : {fmt(agg['clean_negative_rate'])}")
    lines.append(f"全数据集 FAR      : {fmt(agg['overall_far_per_min'])} 次/分钟")
    lines.append("=" * 62)
    if agg["n_negative"] and agg["negative_false_alarms"]:
        lines.append("⚠ 负样本上出现误检：优先加绝对下限 / 持续性检验，别急着调 F1。")
    return "\n".join(lines)
