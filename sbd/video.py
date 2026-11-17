"""视频读取：三种后端任选，互相不绑定。

优先级：``cv2`` > ``ffmpeg`` > 图片序列（Pillow）。

- 有 ``opencv-python`` 就用它（最省事）；
- 没有就调系统 ``ffmpeg`` 管道（零 Python 依赖，只需要装了 ffmpeg）；
- 都没有时，图片序列（``frames/0001.png`` …）也能跑，方便做实验和测试。

所有后端统一输出 **RGB uint8** 的 ``numpy`` 数组，行序为 ``(H, W, 3)``。
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from typing import Dict, Iterator, List, Optional, Sequence

import numpy as np

from .hist import frame_hist

_IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass
class VideoInfo:
    path: str
    fps: float = 25.0
    width: int = 0
    height: int = 0
    n_frames: Optional[int] = None
    backend: str = ""

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "fps": float(self.fps),
            "width": int(self.width),
            "height": int(self.height),
            "n_frames": None if self.n_frames is None else int(self.n_frames),
            "backend": self.backend,
        }


# --------------------------------------------------------------------------- #
# 后端探测
# --------------------------------------------------------------------------- #

def has_cv2() -> bool:
    try:
        import cv2  # noqa: F401
        return True
    except Exception:
        return False


def has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def _is_image_dir(path: str) -> bool:
    return os.path.isdir(path)


def _list_images(path: str) -> List[str]:
    def key(name: str):
        return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]

    names = [n for n in os.listdir(path) if os.path.splitext(n)[1].lower() in _IMAGE_EXT]
    return [os.path.join(path, n) for n in sorted(names, key=key)]


def probe(path: str) -> VideoInfo:
    """读取视频基本信息（fps / 宽高 / 帧数）。"""
    if _is_image_dir(path):
        files = _list_images(path)
        if not files:
            raise ValueError(f"目录里没有图片帧：{path}")
        from PIL import Image

        with Image.open(files[0]) as im:
            w, h = im.size
        return VideoInfo(path=path, fps=25.0, width=w, height=h,
                         n_frames=len(files), backend="images")

    if has_cv2():
        import cv2

        cap = cv2.VideoCapture(path)
        if cap.isOpened():
            info = VideoInfo(
                path=path,
                fps=float(cap.get(cv2.CAP_PROP_FPS) or 25.0),
                width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                n_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None,
                backend="cv2",
            )
            cap.release()
            return info
        cap.release()

    if has_ffmpeg():
        cmd = [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,avg_frame_rate,nb_frames",
            "-of", "json", path,
        ]
        out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
        st = (json.loads(out).get("streams") or [{}])[0]
        rate = st.get("avg_frame_rate") or "25/1"
        try:
            fps = float(Fraction(rate))
        except Exception:
            fps = 25.0
        nf = st.get("nb_frames")
        return VideoInfo(
            path=path,
            fps=fps if fps > 0 else 25.0,
            width=int(st.get("width") or 0),
            height=int(st.get("height") or 0),
            n_frames=int(nf) if nf and str(nf).isdigit() else None,
            backend="ffmpeg",
        )

    raise RuntimeError(
        "无法解码视频：既没有 opencv-python，也没有 ffmpeg。\n"
        "  pip install opencv-python     # 方案一\n"
        "  brew install ffmpeg           # 方案二"
    )


# --------------------------------------------------------------------------- #
# 逐帧迭代
# --------------------------------------------------------------------------- #

def _target_size(w: int, h: int, max_width: Optional[int]):
    if not max_width or w <= max_width:
        return w, h
    nw = int(max_width)
    nh = max(2, int(round(h * nw / max(w, 1) / 2.0)) * 2)
    return nw, nh


def iter_frames(path: str, max_width: Optional[int] = 320) -> Iterator[np.ndarray]:
    """逐帧产出 RGB uint8 图像；``max_width`` 用于下采样加速（直方图对此不敏感）。"""
    info = probe(path)

    if info.backend == "images":
        from PIL import Image

        w, h = _target_size(info.width, info.height, max_width)
        for f in _list_images(path):
            with Image.open(f) as im:
                im = im.convert("RGB")
                if (im.width, im.height) != (w, h):
                    im = im.resize((w, h), Image.BILINEAR)
                yield np.asarray(im, dtype=np.uint8)

    elif info.backend == "cv2":
        import cv2

        w, h = _target_size(info.width, info.height, max_width)
        cap = cv2.VideoCapture(path)
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame.shape[1] != w or frame.shape[0] != h:
                    frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
                yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        finally:
            cap.release()

    elif info.backend == "ffmpeg":
        w, h = _target_size(info.width, info.height, max_width)
        cmd = [
            "ffmpeg", "-v", "error", "-i", path,
            "-vf", f"scale={w}:{h}",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        frame_bytes = w * h * 3
        try:
            while True:
                buf = proc.stdout.read(frame_bytes)
                if not buf or len(buf) < frame_bytes:
                    break
                yield np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3).copy()
        finally:
            proc.stdout.close()
            proc.wait()
    else:  # pragma: no cover
        raise RuntimeError(f"未知后端 {info.backend}")


def read_hists(
    path: str,
    cfg,
    max_width: Optional[int] = 320,
) -> tuple:
    """一次性扫完视频，返回 ``(hists, info)``。

    只保留直方图（每帧几百个 float），不缓存图像，内存占用与帧数线性但极小。
    """
    info = probe(path)
    rows: List[np.ndarray] = []
    for frame in iter_frames(path, max_width=max_width):
        rows.append(frame_hist(frame, space=cfg.space, bins=cfg.bins))
    if not rows:
        raise ValueError(f"没有读到任何帧：{path}")
    info.n_frames = len(rows)
    return np.vstack(rows), info


# --------------------------------------------------------------------------- #
# 定点取帧（用于空间校验 / 抽帧预览）
# --------------------------------------------------------------------------- #

def grab_frames(path: str, indexes: Sequence[int], max_width: Optional[int] = 320) -> Dict[int, np.ndarray]:
    """取出指定帧号的 RGB 图像，返回 ``{index: frame}``。"""
    info = probe(path)
    want = sorted({int(i) for i in indexes if 0 <= int(i) < (info.n_frames or 10 ** 9)})
    out: Dict[int, np.ndarray] = {}
    if not want:
        return out

    if info.backend == "images":
        from PIL import Image

        files = _list_images(path)
        w, h = _target_size(info.width, info.height, max_width)
        for i in want:
            if i >= len(files):
                continue
            with Image.open(files[i]) as im:
                im = im.convert("RGB").resize((w, h), Image.BILINEAR)
                out[i] = np.asarray(im, dtype=np.uint8)
        return out

    if info.backend == "cv2":
        import cv2

        cap = cv2.VideoCapture(path)
        try:
            for i in want:
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ok, frame = cap.read()
                if ok:
                    out[i] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        finally:
            cap.release()
        return out

    # ffmpeg：逐帧精确定位（-ss 放在 -i 之后保证准确）
    w, h = _target_size(info.width, info.height, max_width)
    for i in want:
        t = i / max(info.fps, 1e-6)
        cmd = [
            "ffmpeg", "-v", "error", "-i", path, "-ss", f"{t:.6f}",
            "-frames:v", "1", "-vf", f"scale={w}:{h}",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
        ]
        res = subprocess.run(cmd, capture_output=True)
        if len(res.stdout) >= w * h * 3:
            out[i] = np.frombuffer(res.stdout[: w * h * 3], dtype=np.uint8).reshape(h, w, 3)
    return out
