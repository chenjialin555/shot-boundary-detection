"""空间校验：给「直方图粗筛」加一道高精度的第二关。

直方图是纯颜色统计，没有空间信息——构图完全不同但配色接近的两个镜头，
直方图距离可能很小。这三种特征互补性最好：

- **SSIM**：结构相似度，切点处断崖式下降；
- **ECR**（Edge Change Ratio）：新边缘大量出现 + 旧边缘大量消失；
- **分块直方图**：把画面切成网格，逐块算距离取最大/均值，运动是局部平移，切镜是全画面重排。

默认「三取二」通过，阈值偏宽松——目标是干掉明显的假切，不是替代人工。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Tuple

import numpy as np

from .distance import l1
from .hist import hist_rgb


@dataclass
class VerifyConfig:
    ssim_max: float = 0.55   # SSIM 低于此值 → 一票
    ecr_min: float = 0.25    # 边缘变化率高于此值 → 一票
    block_min: float = 0.30  # 分块直方图距离高于此值 → 一票
    need: int = 2            # 至少几票通过
    grid: int = 3            # 分块网格边长
    ecr_thresh: float = 25.0 # 边缘强度阈值（0-255 梯度）

    def to_dict(self) -> dict:
        return asdict(self)


def to_gray(frame: np.ndarray) -> np.ndarray:
    a = np.asarray(frame, dtype=np.float64)
    if a.ndim == 3:
        return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
    return a


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """全局 SSIM（不分窗）。够快，也足够做「是不是同一画面」的判别。"""
    x = to_gray(a)
    y = to_gray(b)
    if x.shape != y.shape:
        return 0.0
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    mx, my = x.mean(), y.mean()
    vx, vy = x.var(), y.var()
    cov = ((x - mx) * (y - my)).mean()
    num = (2 * mx * my + c1) * (2 * cov + c2)
    den = (mx ** 2 + my ** 2 + c1) * (vx + vy + c2)
    return float(num / den) if den > 0 else 1.0


def _sobel_mag(g: np.ndarray) -> np.ndarray:
    gx = np.zeros_like(g)
    gy = np.zeros_like(g)
    gx[:, 1:-1] = (g[:, 2:] - g[:, :-2]) * 0.5
    gy[1:-1, :] = (g[2:, :] - g[:-2, :]) * 0.5
    return np.hypot(gx, gy)


def _dilate(mask: np.ndarray, r: int = 1) -> np.ndarray:
    if r <= 0:
        return mask
    p = np.pad(mask, r, mode="constant", constant_values=False)
    out = np.zeros_like(mask)
    h, w = mask.shape
    for dy in range(2 * r + 1):
        for dx in range(2 * r + 1):
            out |= p[dy:dy + h, dx:dx + w]
    return out


def edge_change_ratio(prev: np.ndarray, cur: np.ndarray, thresh: float = 25.0) -> float:
    """边缘变化率：新旧边缘「互不重合」的比例，取两个方向的较大值。"""
    e_prev = _sobel_mag(to_gray(prev)) > thresh
    e_cur = _sobel_mag(to_gray(cur)) > thresh
    if not e_prev.any() and not e_cur.any():
        return 0.0
    near_prev = _dilate(e_prev, 1)
    near_cur = _dilate(e_cur, 1)
    new = (e_cur & ~near_prev).sum() / max(e_cur.sum(), 1)
    gone = (e_prev & ~near_cur).sum() / max(e_prev.sum(), 1)
    return float(max(new, gone))


def block_hist_distance(
    prev: np.ndarray,
    cur: np.ndarray,
    grid: int = 3,
    bins: Tuple[int, int, int] = (8, 8, 8),
    top_ratio: float = 0.5,
) -> float:
    """分块直方图距离：逐块算 L1，取最大的 ``top_ratio`` 块求均值。

    只看「变化最剧烈的那几块」，比全图平均更能抓住「整块画面被换掉」。
    """
    a = np.asarray(prev)
    b = np.asarray(cur)
    if a.shape != b.shape or a.ndim != 3:
        return 0.0
    h, w = a.shape[:2]
    g = max(1, int(grid))
    ys = np.linspace(0, h, g + 1).astype(int)
    xs = np.linspace(0, w, g + 1).astype(int)
    dists = []
    for i in range(g):
        for j in range(g):
            pa = a[ys[i]:ys[i + 1], xs[j]:xs[j + 1]]
            pb = b[ys[i]:ys[i + 1], xs[j]:xs[j + 1]]
            if pa.size == 0 or pb.size == 0:
                continue
            dists.append(l1(hist_rgb(pa, bins), hist_rgb(pb, bins)))
    if not dists:
        return 0.0
    dists.sort(reverse=True)
    k = max(1, int(round(len(dists) * top_ratio)))
    return float(np.mean(dists[:k]))


def verify_pair(prev: np.ndarray, cur: np.ndarray, cfg: VerifyConfig | None = None) -> Dict:
    """对一对相邻帧做空间校验，返回证据字典与判决。"""
    cfg = cfg or VerifyConfig()
    s = ssim(prev, cur)
    e = edge_change_ratio(prev, cur, cfg.ecr_thresh)
    b = block_hist_distance(prev, cur, grid=cfg.grid)

    votes = int(s < cfg.ssim_max) + int(e > cfg.ecr_min) + int(b > cfg.block_min)
    return {
        "ssim": round(float(s), 4),
        "ecr": round(float(e), 4),
        "block": round(float(b), 4),
        "votes": votes,
        "is_cut": votes >= cfg.need,
    }
