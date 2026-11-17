"""直方图构建：色彩空间转换 → 量化 → L1 归一化。

整个模块只依赖 numpy，不依赖 OpenCV。这样：
- 算法核心可以在任何环境跑；
- 视频解码由 :mod:`sbd.video` 用 cv2 / ffmpeg / Pillow 三选一，互不绑定。
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

DEFAULT_BINS: Tuple[int, ...] = (18, 4, 4)


def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """uint8/float RGB ``(H, W, 3)`` → float32 HSV。

    - H ∈ ``[0, 360)``（度）
    - S, V ∈ ``[0, 1]``

    和 OpenCV 的 ``COLOR_BGR2HSV``（H ∈ [0,180)）数值范围不同，
    但 bin 划分是等价的。
    """
    arr = np.asarray(rgb)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"rgb 需要 (H, W, 3)，收到 {arr.shape}")
    if arr.dtype == np.uint8:
        arr = arr.astype(np.float32) / 255.0
    else:
        arr = arr.astype(np.float32)

    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    mx = arr.max(axis=-1)
    mn = arr.min(axis=-1)
    diff = mx - mn

    h = np.zeros_like(mx)
    nz = diff > 1e-8
    safe = np.maximum(diff, 1e-8)

    rmask = nz & (mx == r)
    gmask = nz & (mx == g) & ~rmask
    bmask = nz & ~rmask & ~gmask

    h[rmask] = (60.0 * ((g[rmask] - b[rmask]) / safe[rmask])) % 360.0
    h[gmask] = 60.0 * ((b[gmask] - r[gmask]) / safe[gmask]) + 120.0
    h[bmask] = 60.0 * ((r[bmask] - g[bmask]) / safe[bmask]) + 240.0

    s = np.where(mx > 1e-8, diff / np.maximum(mx, 1e-8), 0.0)
    return np.stack([h, s, mx], axis=-1).astype(np.float32)


def _quantize(values: np.ndarray, nbins: int, span: float) -> np.ndarray:
    idx = (values / span * nbins).astype(np.int32, copy=False)
    return np.clip(idx, 0, nbins - 1)


def hist_hsv(
    hsv: np.ndarray,
    bins: Sequence[int] = DEFAULT_BINS,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """对 HSV 图像做 3D 直方图，返回**已归一化**的一维向量。"""
    bins = tuple(int(b) for b in bins)
    if len(bins) != 3:
        raise ValueError("HSV 直方图需要 3 个 bin 数，例如 (18, 4, 4)")
    hb, sb, vb = bins
    h = _quantize(hsv[..., 0], hb, 360.0)
    s = _quantize(hsv[..., 1], sb, 1.0)
    v = _quantize(hsv[..., 2], vb, 1.0)
    flat = (h * sb + s) * vb + v
    if mask is not None:
        flat = flat[np.asarray(mask, dtype=bool)]
    counts = np.bincount(flat.ravel(), minlength=hb * sb * vb).astype(np.float64)
    total = counts.sum()
    if total <= 0:
        return np.full(hb * sb * vb, 1.0 / (hb * sb * vb), dtype=np.float64)
    return counts / total


def hist_rgb(rgb: np.ndarray, bins: Sequence[int] = (8, 8, 8)) -> np.ndarray:
    """对 RGB 图像做 3D 直方图（不转色彩空间，速度最快）。"""
    bins = tuple(int(b) for b in bins)
    if len(bins) != 3:
        raise ValueError("RGB 直方图需要 3 个 bin 数，例如 (8, 8, 8)")
    arr = np.asarray(rgb)
    if arr.dtype != np.uint8:
        arr = np.clip(arr * 255.0 if arr.max() <= 1.0 else arr, 0, 255).astype(np.uint8)
    r = _quantize(arr[..., 0].astype(np.float32), bins[0], 256.0)
    g = _quantize(arr[..., 1].astype(np.float32), bins[1], 256.0)
    b = _quantize(arr[..., 2].astype(np.float32), bins[2], 256.0)
    flat = (r * bins[1] + g) * bins[2] + b
    counts = np.bincount(flat.ravel(), minlength=bins[0] * bins[1] * bins[2]).astype(np.float64)
    total = counts.sum()
    if total <= 0:
        return np.full(counts.size, 1.0 / counts.size, dtype=np.float64)
    return counts / total


def hist_gray(gray: np.ndarray, bins: int = 64) -> np.ndarray:
    """灰度直方图（最快，但无法区分「同亮度不同色调」）。"""
    g = np.asarray(gray)
    if g.ndim == 3:
        g = (0.299 * g[..., 0] + 0.587 * g[..., 1] + 0.114 * g[..., 2])
    g = np.clip(g, 0, 255)
    idx = _quantize(g.astype(np.float32), bins, 256.0)
    counts = np.bincount(idx.ravel(), minlength=bins).astype(np.float64)
    total = counts.sum()
    if total <= 0:
        return np.full(bins, 1.0 / bins, dtype=np.float64)
    return counts / total


def frame_hist(
    frame: np.ndarray,
    space: str = "hsv",
    bins: Sequence[int] = DEFAULT_BINS,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """单帧 → 归一化直方图。``frame`` 为 RGB（H,W,3）或灰度（H,W）。"""
    arr = np.asarray(frame)
    if space == "hsv":
        if arr.ndim == 2:
            arr = np.repeat(arr[..., None], 3, axis=-1)
        return hist_hsv(rgb_to_hsv(arr), bins=bins, mask=mask)
    if space == "rgb":
        return hist_rgb(arr, bins=bins)
    if space == "gray":
        nb = int(bins[0]) if np.isscalar(bins) or isinstance(bins, (int, np.integer)) else 64
        return hist_gray(arr, bins=nb)
    raise ValueError(f"未知色彩空间 {space!r}，可选：hsv / rgb / gray")


def hist_sequence(
    frames: Iterable[np.ndarray],
    space: str = "hsv",
    bins: Sequence[int] = DEFAULT_BINS,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """帧序列 → ``(n_frames, n_bins)`` 的直方图矩阵。"""
    rows = [frame_hist(f, space=space, bins=bins, mask=mask) for f in frames]
    if not rows:
        return np.zeros((0, 0), dtype=np.float64)
    return np.vstack(rows)


def normalize(h: np.ndarray) -> np.ndarray:
    """把任意非负向量归一到和为 1。"""
    h = np.asarray(h, dtype=np.float64)
    s = h.sum()
    if s <= 0:
        return np.full(h.size, 1.0 / h.size)
    return h / s


def smooth_hist(h: np.ndarray, passes: int = 1) -> np.ndarray:
    """对直方图做一次 3 点均值平滑，压制量化噪声造成的尖峰。"""
    out = np.asarray(h, dtype=np.float64)
    if out.size < 3:
        return out
    for _ in range(passes):
        out = np.convolve(out, np.array([0.25, 0.5, 0.25]), mode="same")
        out = normalize(out)
    return out
