"""直方图距离度量。

约定：
- 所有函数接收两个**同形状**的一维数组（归一化直方图，各 bin 之和为 1）。
- 除 :func:`l2` / :func:`chi2` 外，其余度量都归一到 ``[0, 1]``，
  这样绝对阈值 ``t_abs`` 在不同度量之间才有相近的物理含义。
"""

from __future__ import annotations

from typing import Callable

import numpy as np

_EPS = 1e-12


def l1(h1: np.ndarray, h2: np.ndarray) -> float:
    """曼哈顿距离的一半：``0.5 * Σ|h1 - h2|``，取值 ``[0, 1]``。

    工程上最常用：对噪声不敏感、计算便宜、和「直方图交集」等价，
    但比交集少一次 ``min`` 运算。
    """
    a = np.asarray(h1, dtype=np.float64)
    b = np.asarray(h2, dtype=np.float64)
    return float(0.5 * np.abs(a - b).sum())


def intersection(h1: np.ndarray, h2: np.ndarray) -> float:
    """直方图交集距离：``1 - Σ min(h1, h2)``，取值 ``[0, 1]``。"""
    a = np.asarray(h1, dtype=np.float64)
    b = np.asarray(h2, dtype=np.float64)
    return float(1.0 - np.minimum(a, b).sum())


def l2(h1: np.ndarray, h2: np.ndarray) -> float:
    """欧氏距离（未归一化）。对单个 bin 的突变很敏感。"""
    a = np.asarray(h1, dtype=np.float64)
    b = np.asarray(h2, dtype=np.float64)
    return float(np.sqrt(((a - b) ** 2).sum()))


def chi2(h1: np.ndarray, h2: np.ndarray) -> float:
    """卡方距离 ``Σ (h1-h2)^2 / (h1+h2)``（未归一化，稀疏 bin 更稳健）。"""
    a = np.asarray(h1, dtype=np.float64)
    b = np.asarray(h2, dtype=np.float64)
    return float((((a - b) ** 2) / (a + b + _EPS)).sum())


def bhattacharyya(h1: np.ndarray, h2: np.ndarray) -> float:
    """Bhattacharyya 距离 ``1 - Σ sqrt(h1*h2)``，取值 ``[0, 1]``。

    对「整体分布形状」的变化敏感，对单个 bin 的抖动不敏感。
    """
    a = np.asarray(h1, dtype=np.float64)
    b = np.asarray(h2, dtype=np.float64)
    return float(1.0 - np.sqrt(np.maximum(a, 0) * np.maximum(b, 0)).sum())


def cosine(h1: np.ndarray, h2: np.ndarray) -> float:
    """余弦距离 ``1 - <h1,h2> / (|h1||h2|)``，取值 ``[0, 1]``。"""
    a = np.asarray(h1, dtype=np.float64)
    b = np.asarray(h2, dtype=np.float64)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < _EPS or nb < _EPS:
        return 0.0
    # 浮点误差可能让同一向量算出 -2e-16，夹到 0
    return float(min(1.0, max(0.0, 1.0 - float(a @ b) / (na * nb))))


METRICS: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "l1": l1,
    "intersection": intersection,
    "l2": l2,
    "chi2": chi2,
    "bhattacharyya": bhattacharyya,
    "cosine": cosine,
}


def resolve(name: str) -> Callable[[np.ndarray, np.ndarray], float]:
    """把度量名字解析成函数；名字非法时抛出带提示的 ValueError。"""
    try:
        return METRICS[name]
    except KeyError:
        raise ValueError(
            f"未知的距离度量 {name!r}，可选：{', '.join(sorted(METRICS))}"
        ) from None


def diff_series(hists: np.ndarray, metric: str = "l1") -> np.ndarray:
    """逐帧差分序列：``d[i] = dist(h[i-1], h[i])``，``d[0] = 0``。

    对常用度量做了向量化，避免 n 次 Python 循环。
    """
    h = np.asarray(hists, dtype=np.float64)
    n = len(h)
    d = np.zeros(n, dtype=np.float64)
    if n < 2:
        return d
    a, b = h[:-1], h[1:]
    if metric == "l1":
        d[1:] = 0.5 * np.abs(a - b).sum(axis=1)
    elif metric == "intersection":
        d[1:] = 1.0 - np.minimum(a, b).sum(axis=1)
    elif metric == "l2":
        d[1:] = np.sqrt(((a - b) ** 2).sum(axis=1))
    elif metric == "chi2":
        d[1:] = (((a - b) ** 2) / (a + b + _EPS)).sum(axis=1)
    elif metric == "bhattacharyya":
        d[1:] = 1.0 - np.sqrt(np.maximum(a, 0) * np.maximum(b, 0)).sum(axis=1)
    elif metric == "cosine":
        na = np.linalg.norm(a, axis=1)
        nb = np.linalg.norm(b, axis=1)
        denom = np.maximum(na * nb, _EPS)
        d[1:] = 1.0 - (a * b).sum(axis=1) / denom
    else:
        fn = resolve(metric)
        d[1:] = [fn(a[i], b[i]) for i in range(n - 1)]
    return d
