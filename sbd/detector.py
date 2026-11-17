"""镜头边界检测器：绝对下限 + 鲁棒局部阈值 + 持续性检验 + 渐变检测。

设计原则（详细推导见 ``references/algorithm.md``）：

1. **相对阈值一定会产生假切点**，所以任何判决前面都必须先过绝对下限 ``t_abs``；
2. 局部统计用 **median + MAD**，不用 mean + std（后者会被切点自身的尖峰抬高，导致漏检）；
3. 真切点必须满足**持续性**：前后窗口的均值直方图差异显著大于窗口内部的波动，
   这一条同时干掉了闪光、甩镜、瞬态遮挡，是「单镜头视频误检」的主防线；
4. 渐变转场用「累积变化 + 单向性 + 结束后静默」三个条件，不用单帧差。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .distance import diff_series, resolve

_MAD_K = 1.4826  # 让 MAD 与标准差可比


@dataclass
class Cut:
    """一个镜头边界。

    ``frame`` 是**新镜头的第一帧**索引，也就是说边界落在 ``frame-1`` 与 ``frame`` 之间。
    """

    frame: int
    score: float
    kind: str = "hard"  # hard | gradual
    detail: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        out = {"frame": int(self.frame), "score": round(float(self.score), 4), "kind": self.kind}
        if self.detail:
            out["detail"] = {k: round(float(v), 4) for k, v in self.detail.items()}
        return out


@dataclass
class DetectorConfig:
    """全部可调参数。默认值在 25fps、HSV 18x4x4、L1 距离下标定。"""

    # --- 直方图 ---
    space: str = "hsv"
    bins: Tuple[int, ...] = (18, 4, 4)
    metric: str = "l1"

    # --- 硬切 ---
    t_abs: float = 0.38          # 绝对下限：归一化 L1 下硬切的物理下限
    k_mad: float = 6.0           # 局部鲁棒阈值系数
    window: int = 25             # 局部统计半窗（帧）
    local_max_gap: int = 3       # 峰值抑制半径（帧）
    min_shot_len_s: float = 0.5  # 最短镜头长度（秒）

    # --- 闪光抑制 ---
    suppress_flash: bool = True
    flash_ratio: float = 0.35    # dist(i-1, i+1) < ratio * d[i] 判为闪光

    # --- 持续性检验 ---
    check_persistence: bool = True
    persistence_k: int = 8
    t_persist: float = 0.28
    persist_snr: float = 2.0

    # --- 渐变转场 ---
    detect_gradual: bool = True
    gradual_floor: float = 0.008     # 单帧变化的下限（低于此视为静止）
    gradual_rel_floor: float = 1.3   # 同时要求高于全片中位帧差 1.3 倍
    gradual_min_frames_s: float = 0.25
    gradual_max_frames_s: float = 2.0
    gradual_total: float = 0.35      # 窗口内累积变化下限
    gradual_coherence: float = 0.60  # dist(首,尾) / Σd，衡量单向性
    gradual_post_quiet: float = 0.60 # 结束后变化率 / 期间变化率 的上限

    def to_dict(self) -> dict:
        d = asdict(self)
        d["bins"] = list(self.bins)
        return d

    @classmethod
    def from_dict(cls, data: Dict) -> "DetectorConfig":
        data = dict(data)
        if "bins" in data:
            data["bins"] = tuple(int(x) for x in data["bins"])
        return cls(**data)


# --------------------------------------------------------------------------- #
# 基础工具
# --------------------------------------------------------------------------- #

def robust_threshold(d: np.ndarray, window: int = 25, k: float = 6.0) -> np.ndarray:
    """逐帧的鲁棒局部阈值：``median + k * 1.4826 * MAD``。

    用 median/MAD 而不是 mean/std：切点本身是一个巨大的离群值，
    如果它在窗口内，mean/std 会被抬高，反而把真切点「藏」起来了。
    """
    d = np.asarray(d, dtype=np.float64)
    n = d.size
    if n == 0:
        return np.zeros(0)
    w = max(1, int(window))
    # 边缘用 NaN 填充：窗口被截断时只用有效样本估计，
    # 否则首尾几帧的阈值会被「边缘复制」带偏（会漏检片头切点）。
    pad = np.pad(d, (w, w), constant_values=np.nan)
    win = np.lib.stride_tricks.sliding_window_view(pad, 2 * w + 1)
    with np.errstate(all="ignore"):
        med = np.nanmedian(win, axis=1)
        mad = np.nanmedian(np.abs(win - med[:, None]), axis=1)
    med = np.nan_to_num(med, nan=0.0)
    mad = np.nan_to_num(mad, nan=0.0)
    return med + k * _MAD_K * mad


def min_shot_frames(cfg: DetectorConfig, fps: float) -> int:
    return max(1, int(round(cfg.min_shot_len_s * float(fps))))


def _enforce_min_length(cuts: List[Cut], cfg: DetectorConfig, fps: float, n: int) -> List[Cut]:
    """最短镜头约束：挨得太近的候选只保留第一个（更靠前的那个）。"""
    length = min_shot_frames(cfg, fps)
    out: List[Cut] = []
    last = 0
    for c in sorted(cuts, key=lambda c: c.frame):
        if c.frame <= 0 or c.frame >= n:
            continue
        if c.frame - last >= length:
            out.append(c)
            last = c.frame
    return out


def _is_flash(h: np.ndarray, d: np.ndarray, i: int, cfg: DetectorConfig) -> bool:
    """闪光判据：第 i 帧只是「中间被闪了一下」，前后两帧其实很像。"""
    if i - 1 < 0 or i + 1 >= len(h):
        return False
    metric = resolve(cfg.metric)
    across = metric(h[i - 1], h[i + 1])
    return across < cfg.flash_ratio * d[i]


def _persistent(h: np.ndarray, d: np.ndarray, i: int, cfg: DetectorConfig) -> bool:
    """持续性检验：切点两侧均值直方图差异必须远大于两侧内部波动。

    - 硬切：两侧均值差大，两侧内部各自稳定 → 通过
    - 闪光：两侧均值几乎相同 → 拒绝
    - 甩镜/剧烈运动：两侧差异虽大，但两侧内部波动同样大 → 拒绝
    """
    n = len(h)
    # 靠近首尾时缩短窗口，但至少要 3 帧，否则无法判断
    k = min(int(cfg.persistence_k), i, n - i)
    if k < 3:
        return True
    metric = resolve(cfg.metric)
    before = h[i - k:i].mean(axis=0)
    after = h[i:i + k].mean(axis=0)
    d_pers = metric(before, after)
    if d_pers < cfg.t_persist:
        return False
    inner_before = d[i - k + 1:i]
    inner_after = d[i + 1:i + k]
    noise = 0.0
    if inner_before.size:
        noise = max(noise, float(inner_before.mean()))
    if inner_after.size:
        noise = max(noise, float(inner_after.mean()))
    return d_pers >= cfg.persist_snr * noise


# --------------------------------------------------------------------------- #
# 硬切
# --------------------------------------------------------------------------- #

def detect_hard_cuts(hists: np.ndarray, cfg: DetectorConfig, fps: float) -> List[Cut]:
    """检测硬切（cut）。"""
    h = np.asarray(hists, dtype=np.float64)
    n = len(h)
    if n < 3:
        return []

    d = diff_series(h, cfg.metric)
    thr = np.maximum(cfg.t_abs, robust_threshold(d, cfg.window, cfg.k_mad))

    cand = np.flatnonzero(d >= thr)
    cand = cand[cand > 0]
    if cand.size == 0:
        return []

    # 峰值抑制：±gap 帧内只保留最大值，避免一个切点被检出多次
    gap = max(1, int(cfg.local_max_gap))
    picked: List[int] = []
    for i in cand:
        lo, hi = max(0, i - gap), min(n, i + gap + 1)
        if d[i] >= d[lo:hi].max() - 1e-12:
            picked.append(int(i))

    if cfg.suppress_flash:
        picked = [i for i in picked if not _is_flash(h, d, i, cfg)]
    if cfg.check_persistence:
        picked = [i for i in picked if _persistent(h, d, i, cfg)]

    cuts = [
        Cut(frame=i, score=float(d[i]), kind="hard",
            detail={"d": float(d[i]), "thr": float(thr[i])})
        for i in picked
    ]
    return _enforce_min_length(cuts, cfg, fps, n)


# --------------------------------------------------------------------------- #
# 渐变转场
# --------------------------------------------------------------------------- #

def detect_gradual_transitions(hists: np.ndarray, cfg: DetectorConfig, fps: float) -> List[Cut]:
    """检测渐变转场（dissolve / fade / wipe）。

    判据三连（缺一不可）：

    1. **累积变化够大**：窗口内 ``Σ d`` 超过 ``gradual_total``；
    2. **单向性够强**：``dist(窗口首帧, 窗口末帧) / Σ d`` 超过 ``gradual_coherence``。
       来回摇镜会反复穿越同样的颜色分布，Σd 很大但首尾差异不大，因此被排除；
    3. **结束后静默**：转场完成后变化率明显回落。
       持续摇镜不会「静下来」，因此被排除。
    """
    if not cfg.detect_gradual:
        return []
    h = np.asarray(hists, dtype=np.float64)
    n = len(h)
    if n < 6:
        return []

    metric = resolve(cfg.metric)
    d = diff_series(h, cfg.metric)
    floor = max(float(cfg.gradual_floor), 1e-6)
    if d.size > 1:
        floor = max(floor, float(cfg.gradual_rel_floor) * float(np.median(d[1:])))
    min_f = max(3, int(round(cfg.gradual_min_frames_s * float(fps))))
    max_f = max(min_f, int(round(cfg.gradual_max_frames_s * float(fps))))
    post_k = max(3, min(15, int(round(0.6 * fps))))

    cuts: List[Cut] = []
    i = 1
    while i < n:
        if d[i] <= floor:
            i += 1
            continue

        start = i
        total = 0.0
        peak = 0.0
        while i < n and d[i] > floor and (i - start) < max_f:
            total += d[i]
            peak = max(peak, float(d[i]))
            i += 1
        length = i - start

        if length < min_f:
            continue
        if peak >= cfg.t_abs:          # 中间有硬切，不是渐变
            continue
        if total < cfg.gradual_total:  # 总变化量不够
            continue

        end = min(i, n - 1)
        span = metric(h[start - 1], h[end])
        coherence = span / (total + 1e-12)
        if coherence < cfg.gradual_coherence:
            continue

        # 结束后必须静默（否则是在持续运动）
        post_lo, post_hi = end + 1, min(n, end + 1 + post_k)
        if post_hi - post_lo >= 3:
            post_rate = float(d[post_lo:post_hi].mean())
            during_rate = total / max(1, length)
            if post_rate > cfg.gradual_post_quiet * during_rate:
                continue

        cuts.append(
            Cut(
                frame=int(start),
                score=float(span),
                kind="gradual",
                detail={"length": float(length), "coherence": float(coherence),
                        "total": float(total)},
            )
        )

    return _enforce_min_length(cuts, cfg, fps, n)


# --------------------------------------------------------------------------- #
# 组合入口
# --------------------------------------------------------------------------- #

def detect(hists: np.ndarray, cfg: Optional[DetectorConfig] = None, fps: float = 25.0) -> List[Cut]:
    """完整流水线：硬切 + 渐变，合并去重，按时间排序。"""
    cfg = cfg or DetectorConfig()
    cuts = detect_hard_cuts(hists, cfg, fps)

    if cfg.detect_gradual and len(hists) > 6:
        gradual = detect_gradual_transitions(hists, cfg, fps)
        length = min_shot_frames(cfg, fps)
        for g in gradual:
            if all(abs(g.frame - c.frame) >= length for c in cuts):
                cuts.append(g)

    cuts.sort(key=lambda c: c.frame)
    return cuts


def cuts_to_shots(cuts: Sequence[Cut], n_frames: int) -> List[Tuple[int, int]]:
    """把切点列表转换成 ``[(start, end_exclusive), ...]`` 的镜头区间。"""
    bounds = [0] + [c.frame for c in cuts if 0 < c.frame < n_frames] + [n_frames]
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1) if bounds[i + 1] > bounds[i]]
