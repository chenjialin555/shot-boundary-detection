"""sbd —— 基于直方图的视频分镜（镜头边界）检测。

最小用法::

    from sbd import DetectorConfig, detect_video, detect

    cuts = detect_video("input.mp4")            # 一行搞定
    for c in cuts:
        print(c.frame, c.score, c.kind)

分步用法::

    hists, info = read_hists("input.mp4", DetectorConfig())
    cuts = detect(hists, DetectorConfig(), info.fps)

设计要点（详见 references/algorithm.md）：

- **绝对下限 t_abs** 是必需的，否则单镜头视频必然被切碎；
- 局部阈值用 **median + MAD**，不要用 mean + std；
- 真切点必须过**持续性检验**，闪光/甩镜在这里被拦掉；
- 渐变转场用**累积变化 + 单向性 + 结束静默**，不用单帧差。
"""

from .changepoint import bic_penalty, pelt, segment_cost, segment_shots
from .detector import (
    Cut,
    DetectorConfig,
    cuts_to_shots,
    detect,
    detect_gradual_transitions,
    detect_hard_cuts,
    min_shot_frames,
)
from .distance import (
    METRICS,
    bhattacharyya,
    chi2,
    cosine,
    diff_series,
    intersection,
    l1,
    l2,
    resolve,
)
from .hist import frame_hist, hist_gray, hist_hsv, hist_rgb, normalize, rgb_to_hsv
from .metrics import aggregate, evaluate, format_report, match_events
from .verify import VerifyConfig, edge_change_ratio, ssim, verify_pair
from .video import VideoInfo, grab_frames, iter_frames, probe, read_hists

__version__ = "0.1.0"


def detect_video(
    path: str,
    cfg: DetectorConfig | None = None,
    max_width: int = 320,
    mode: str = "threshold",
):
    """端到端：读视频 → 出切点。

    ``mode``：``threshold``（默认，快）/ ``changepoint``（无阈值）/
    ``hybrid``（两者并集）。
    """
    cfg = cfg or DetectorConfig()
    hists, info = read_hists(path, cfg, max_width=max_width)
    fps = info.fps or 25.0

    if mode == "changepoint":
        breaks = segment_shots(hists, min_size=min_shot_frames(cfg, fps))
        d = diff_series(hists, cfg.metric)
        cuts = [Cut(frame=int(b), score=float(d[b]) if b < len(d) else 0.0) for b in breaks]
    elif mode == "hybrid":
        cuts = detect(hists, cfg, fps)
        for b in segment_shots(hists, min_size=min_shot_frames(cfg, fps)):
            if all(abs(b - c.frame) >= min_shot_frames(cfg, fps) for c in cuts):
                cuts.append(Cut(frame=int(b), score=0.0))
        cuts.sort(key=lambda c: c.frame)
    else:
        cuts = detect(hists, cfg, fps)

    return cuts


__all__ = [
    "Cut",
    "DetectorConfig",
    "METRICS",
    "VerifyConfig",
    "VideoInfo",
    "__version__",
    "aggregate",
    "bhattacharyya",
    "bic_penalty",
    "chi2",
    "cosine",
    "cuts_to_shots",
    "detect",
    "detect_gradual_transitions",
    "detect_hard_cuts",
    "detect_video",
    "diff_series",
    "edge_change_ratio",
    "evaluate",
    "format_report",
    "frame_hist",
    "grab_frames",
    "hist_gray",
    "hist_hsv",
    "hist_rgb",
    "intersection",
    "iter_frames",
    "l1",
    "l2",
    "match_events",
    "min_shot_frames",
    "normalize",
    "pelt",
    "probe",
    "read_hists",
    "resolve",
    "rgb_to_hsv",
    "segment_cost",
    "segment_shots",
    "ssim",
    "verify_pair",
]
