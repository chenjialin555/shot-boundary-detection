"""变点检测（change-point detection）：让「0 个切点」成为一个合法解。

阈值法的病根是**强行二值化**——它假定数据里必然存在切点，
于是单镜头视频也一定会被切出几段。

惩罚式变点检测把问题写成模型选择：

    min_分割  Σ_段 代价(段) + penalty * 切点数

单镜头视频上，任何切分带来的代价下降都不足以抵消惩罚，输出自然是空列表。
这就是「无阈值」路线，也是对单镜头视频最干净的解法。

实现：PELT（Killick et al., 2012），代价函数用 L2（段内平方误差和）。
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np


def _prefix(h: np.ndarray):
    """前缀和 / 前缀平方和，用于 O(1) 计算任意段的 L2 代价。"""
    n, dim = h.shape
    p = np.zeros((n + 1, dim), dtype=np.float64)
    q = np.zeros(n + 1, dtype=np.float64)
    p[1:] = np.cumsum(h, axis=0)
    q[1:] = np.cumsum((h ** 2).sum(axis=1))
    return p, q


def segment_cost(p: np.ndarray, q: np.ndarray, s: int, t: int) -> float:
    """段 ``[s, t)`` 的 L2 代价：``Σ ||h_i - mean||²``。"""
    n = t - s
    if n <= 0:
        return 0.0
    total = q[t] - q[s]
    mean_shift = ((p[t] - p[s]) ** 2).sum() / n
    return float(max(total - mean_shift, 0.0))


def estimate_sigma2(hists: np.ndarray, trim: float = 0.9) -> float:
    """用相邻帧差的稳健统计估计「段内噪声方差」，供 BIC 惩罚使用。

    相邻帧差 ``E||h_i - h_{i-1}||² = 2 * dim * sigma²``，取分位数（默认 90%）
    而不是最大值，避免切点把估计值抬高。
    """
    h = np.asarray(hists, dtype=np.float64)
    if len(h) < 2:
        return 1e-6
    sq = ((h[1:] - h[:-1]) ** 2).sum(axis=1)
    dim = h.shape[1]
    robust = float(np.quantile(sq, trim)) / (2.0 * dim)
    return max(robust, 1e-9)


def bic_penalty(hists: np.ndarray, sigma2: Optional[float] = None) -> float:
    """BIC 惩罚项：``dim * sigma² * log(n)``。

    这个量纲正好和 L2 代价（平方误差和）对齐，所以可以直接相加。
    """
    h = np.asarray(hists, dtype=np.float64)
    n, dim = h.shape
    s2 = estimate_sigma2(h) if sigma2 is None else float(sigma2)
    return dim * s2 * np.log(max(n, 2))


def _costs(p: np.ndarray, q: np.ndarray, starts: np.ndarray, t: int) -> np.ndarray:
    """向量化计算多个候选起点到 t 的段代价。"""
    durs = (t - starts).astype(np.float64)
    total = q[t] - q[starts]
    shift = ((p[t] - p[starts]) ** 2).sum(axis=1) / durs
    return np.maximum(total - shift, 0.0)


def pelt(
    hists: np.ndarray,
    penalty: Optional[float] = None,
    min_size: int = 5,
    jump: Optional[int] = None,
) -> List[int]:
    """PELT 变点检测。

    参数
    ----
    hists : (n_frames, n_bins) 的直方图矩阵
    penalty : 惩罚值；``None`` 时用 :func:`bic_penalty`
    min_size : 每段最少帧数
    jump : 每隔 jump 帧才考虑一个候选点（长视频加速用）。
           ``None`` 时自动：≤5000 帧精确计算，更长则按比例跳步，
           最后再把变点吸附回局部最大差分处，精度基本无损。

    返回
    ----
    变点位置列表（新镜头首帧索引），不含 0 和 n。
    """
    h = np.asarray(hists, dtype=np.float64)
    n = len(h)
    min_size = max(1, int(min_size))
    if jump is None:
        jump = 1 if n <= 5000 else int(min(10, max(2, n // 5000)))
    jump = max(1, int(jump))
    if n < 2 * min_size:
        return []

    p, q = _prefix(h)
    if penalty is None:
        penalty = bic_penalty(h)
    penalty = float(penalty)

    # F[t] = 前 t 帧的最优代价（含惩罚）；F[0] = -penalty 使第一段不付惩罚
    F = np.full(n + 1, np.inf, dtype=np.float64)
    F[0] = -penalty
    back = np.zeros(n + 1, dtype=np.int64)
    candidates: List[int] = [0]

    for t in range(1, n + 1):
        if jump > 1 and t % jump and t != n:
            continue

        Rs = np.asarray(candidates, dtype=np.int64)
        valid = (t - Rs) >= min_size
        if valid.any():
            starts = Rs[valid]
            costs = _costs(p, q, starts, t)
            vals = F[starts] + costs + penalty
            j = int(np.argmin(vals))
            F[t] = float(vals[j])
            back[t] = int(starts[j])

            # PELT 剪枝：无法改进当前最优的前驱，未来也不可能改进
            short = (t - Rs) < min_size
            keep_mask = short | ((F[Rs] + _costs(p, q, Rs, t)) <= F[t])
            candidates = [int(s) for s, ok in zip(Rs, keep_mask) if ok]

        # 当前帧本身成为「未来某段的起点」
        candidates.append(t)

    if not np.isfinite(F[n]):
        return []

    breaks: List[int] = []
    t = n
    while t > 0 and back[t] > 0:
        t = int(back[t])
        if 0 < t < n:
            breaks.append(t)
    breaks.reverse()

    # jump>1 时，把变点吸附到局部最大差分处，恢复帧级精度
    if jump > 1 and breaks:
        d = 0.5 * np.abs(np.diff(h, axis=0)).sum(axis=1)  # d[k] 对应边界 k+1
        refined: List[int] = []
        for b in breaks:
            lo = max(1, b - jump)
            hi = min(n, b + jump + 1)
            seg = d[lo - 1:hi - 1]
            refined.append(int(lo + int(np.argmax(seg))) if seg.size else b)
        breaks = sorted(set(refined))

    return breaks


def segment_shots(
    hists: np.ndarray,
    min_size: int = 5,
    penalty: Optional[float] = None,
    jump: Optional[int] = None,
) -> List[int]:
    """``pelt`` 的语义化封装：返回镜头边界帧索引。"""
    return pelt(hists, penalty=penalty, min_size=min_size, jump=jump)
