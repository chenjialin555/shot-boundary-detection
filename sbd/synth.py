"""合成测试数据：不需要任何真实视频，就能验证/演示整套算法。

提供两类数据：

- :func:`hist_scenario` —— 直接生成直方图序列 + 真值切点，用于单元测试和参数实验；
- :func:`rgb_scenario`  —— 生成 RGB 帧序列，可写成 PNG 序列跑端到端流程。

场景覆盖分镜检测最典型的四种情况：

===========  ==========================================  ==============
场景          含义                                          真值切点数
===========  ==========================================  ==============
``cuts``     4 个镜头硬切                                     4
``single``   单镜头长视频，含重尾噪声（甩镜/遮挡）              0
``flash``    单镜头 + 两次白闪                                 0
``dissolve`` 两个静态镜头之间的叠化                            1
===========  ==========================================  ==============

直方图生成模型：每帧 = 镜头基准分布 + 一定比例的「别的分布」。
这个比例近似等于相邻帧的 L1 距离，所以 ``noise=0.05`` 就表示
「帧间距离约 0.05 的镜内运动」，量级和真实视频对得上。
"""

from __future__ import annotations

import os
from typing import Dict, List, Sequence, Tuple

import numpy as np

DIM = 288  # 与 HSV (18, 4, 4) 的 bin 数一致


def _normalize(h: np.ndarray) -> np.ndarray:
    h = np.clip(h, 0, None)
    s = h.sum()
    return h / s if s > 0 else np.full(h.size, 1.0 / h.size)


def _rand_hist(rng: np.random.Generator, dim: int, alpha: float = 1.0) -> np.ndarray:
    """随机「内容分布」。alpha 越小越稀疏，方向性越强。"""
    return rng.dirichlet(np.full(dim, float(alpha)))


def _mix(base: np.ndarray, other: np.ndarray, amount: float) -> np.ndarray:
    return _normalize((1.0 - amount) * base + amount * other)


def _shot(
    rng: np.random.Generator,
    dim: int,
    n: int,
    base: np.ndarray,
    noise: float,
    spike_prob: float = 0.0,
    spike_amount: float = 0.25,
) -> List[np.ndarray]:
    """一个镜头内的帧序列：每帧混入 ``noise`` 比例的随机分布。"""
    out = []
    for _ in range(n):
        other = _rand_hist(rng, dim)
        h = _mix(base, other, noise)
        if spike_prob > 0 and rng.random() < spike_prob:
            # 重尾：模拟甩镜、快速遮挡这类偶发剧烈运动
            h = _mix(base, _rand_hist(rng, dim), spike_amount)
        out.append(h)
    return out


def hist_scenario(
    kind: str = "cuts",
    lengths: Sequence[int] = (120, 90, 150, 110, 130),
    seed: int = 0,
    dim: int = DIM,
    noise: float | None = None,
) -> Dict:
    """生成直方图序列。

    返回 ``{"hists": (n, dim), "cuts": [int], "name": str, "fps": 25}``。
    """
    rng = np.random.default_rng(seed)
    kind = kind.lower()

    if kind == "single":
        n = int(lengths[0]) if len(lengths) == 1 else 600
        base = _rand_hist(rng, dim, alpha=2.0)
        hists = _shot(rng, dim, n, base, noise=0.05 if noise is None else noise,
                      spike_prob=0.012, spike_amount=0.25)
        return {"hists": np.vstack(hists), "cuts": [], "name": "single", "fps": 25}

    if kind == "flash":
        n = int(lengths[0]) if len(lengths) == 1 else 300
        base = _rand_hist(rng, dim, alpha=0.4)
        hists = _shot(rng, dim, n, base, noise=0.01 if noise is None else noise)
        for pos in (n // 3, 2 * n // 3):
            if 0 < pos < n - 1:
                hists[pos] = np.full(dim, 1.0 / dim)  # 白闪
        return {"hists": np.vstack(hists), "cuts": [], "name": "flash", "fps": 25}

    if kind == "dissolve":
        a_len, trans, b_len = 60, 20, 70
        a = _rand_hist(rng, dim, alpha=0.5)
        b = _rand_hist(rng, dim, alpha=0.5)
        base_noise = 0.005 if noise is None else noise
        hists = _shot(rng, dim, a_len, a, noise=base_noise)
        for k in range(1, trans + 1):
            alpha = k / (trans + 1)
            hists.append(_mix(_normalize((1 - alpha) * a + alpha * b),
                              _rand_hist(rng, dim), base_noise))
        hists += _shot(rng, dim, b_len, b, noise=base_noise)
        return {"hists": np.vstack(hists), "cuts": [a_len], "name": "dissolve", "fps": 25}

    if kind != "cuts":
        raise ValueError(f"未知场景 {kind!r}，可选：cuts / single / flash / dissolve")

    base_noise = 0.03 if noise is None else noise
    hists: List[np.ndarray] = []
    cuts: List[int] = []
    for idx, length in enumerate(lengths):
        base = _rand_hist(rng, dim, alpha=0.35)
        if idx > 0:
            cuts.append(len(hists))
        hists.extend(_shot(rng, dim, int(length), base, noise=base_noise))
    return {"hists": np.vstack(hists), "cuts": cuts, "name": "cuts", "fps": 25}


def rgb_scenario(
    kind: str = "cuts",
    lengths: Sequence[int] = (40, 30, 40),
    size: Tuple[int, int] = (72, 96),
    seed: int = 0,
) -> Dict:
    """生成 RGB 帧序列（用于端到端测试 / 抽帧预览）。

    每个镜头固定一组色块 + 小幅抖动，保证「镜内帧差小、切点帧差大」，
    和真实视频的统计特性一致。

    返回 ``{"frames": [uint8 (H,W,3)], "cuts": [int], "name": str, "fps": 25}``。
    """
    rng = np.random.default_rng(seed)
    hgt, wid = size

    def new_patches(color: np.ndarray, n: int = 4):
        out = []
        for _ in range(n):
            hh = int(rng.integers(8, max(hgt // 2, 9)))
            ww = int(rng.integers(8, max(wid // 2, 9)))
            y0 = int(rng.integers(0, max(hgt - hh, 1)))
            x0 = int(rng.integers(0, max(wid - ww, 1)))
            out.append((y0, x0, hh, ww, np.clip(color * 0.3 + rng.uniform(0, 255, 3), 0, 255)))
        return out

    yy, xx = np.mgrid[0:hgt, 0:wid]
    grad = (0.70 + 0.60 * (xx / max(wid - 1, 1)))[..., None]

    def make_frame(color: np.ndarray, patches) -> np.ndarray:
        frame = np.clip(color[None, None, :] * grad, 0, 255).astype(np.float64)
        frame = frame + rng.normal(0, 4, size=frame.shape)
        for (y0, x0, hh, ww, pcol) in patches:
            dy, dx = rng.integers(-1, 2, 2)
            y = int(np.clip(y0 + dy, 0, hgt - 1))
            x = int(np.clip(x0 + dx, 0, wid - 1))
            frame[y:min(y + hh, hgt), x:min(x + ww, wid)] = np.clip(
                pcol + rng.normal(0, 4, 3), 0, 255)
        return np.clip(frame, 0, 255).astype(np.uint8)

    colors = [
        np.array([200.0, 60.0, 60.0]),
        np.array([40.0, 60.0, 190.0]),
        np.array([230.0, 210.0, 80.0]),
        np.array([50.0, 170.0, 90.0]),
    ]

    if kind == "single":
        color = colors[0]
        patches = new_patches(color)
        return {"frames": [make_frame(color, patches) for _ in range(int(sum(lengths)))],
                "cuts": [], "name": "single", "fps": 25}

    frames: List[np.ndarray] = []
    cuts: List[int] = []
    for i, length in enumerate(lengths):
        if i > 0:
            cuts.append(len(frames))
        color = colors[i % len(colors)]
        patches = new_patches(color)
        for _ in range(int(length)):
            frames.append(make_frame(color, patches))
    return {"frames": frames, "cuts": cuts, "name": "cuts", "fps": 25}


def write_png_sequence(directory: str, frames: Sequence[np.ndarray], prefix: str = "frame") -> int:
    """把帧序列写成 ``frame_000000.png`` …，返回写出的帧数。"""
    from PIL import Image

    os.makedirs(directory, exist_ok=True)
    for i, frame in enumerate(frames):
        Image.fromarray(frame).save(os.path.join(directory, f"{prefix}_{i:06d}.png"))
    return len(frames)
