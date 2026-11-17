---
name: shot-boundary-detection
description: 视频分镜 / 镜头边界（shot boundary, cut detection）检测。当用户要用帧差或直方图差分做分镜、找切点、数一个视频有几个镜头、调检测阈值，或者抱怨「单镜头视频被切成了好几段」「闪光被当成切镜」时使用。包含可直接运行的 Python 实现（绝对阈值 + 持续性检验 + PELT 变点检测 + 空间校验）、阈值标定流程，以及 F1 之外必须看的 FAR 评估方法。
---

# 基于直方图的视频分镜检测

## 何时用这个 skill

- 要做「视频分镜 / 镜头切分 / 找切点 / shot boundary detection」
- 已经用相邻帧直方图差分，但效果不稳定：误检多、漏检多、阈值难调
- **单镜头视频被切碎**、闪光/摇镜被当成切镜
- 需要评估这类算法的准确率，但发现 F1 说不清问题

## 30 秒结论

直方图差分本身没错，**错的是「只做相对阈值」**。真正能跑的实现必须是三件套：

1. **绝对下限 `t_abs`** —— 相对阈值（`μ+kσ`、`median+k·MAD`）是尺度无关的，
   任何视频都一定会有帧落在尾部，所以单镜头视频**必然**被切出几段。
2. **持续性检验** —— 真切点前后内容「持续不同」；闪光、甩镜、瞬态遮挡不是。
   判据：`dist(前K帧均值, 后K帧均值) ≥ max(t_persist, 2×窗口内平均帧差)`。
3. **渐变转场单独处理** —— 叠化/淡入淡出的单帧差很小，必须用
   「窗口累积变化 + 单向性 + 结束后静默」三个条件，不能用单帧差。

再加两层可选项：**变点检测（PELT + BIC 惩罚）** 让「0 个切点」成为合法解；
**空间校验（SSIM / ECR / 分块直方图）** 做高精度复核。

## 工作流

1. **先跑基线**：`python -m sbd detect input.mp4`，看默认参数下的切点。
2. **看 `d` 曲线**：正样本应该「双峰分离」，负样本应该是「带毛刺的平地」。
   分不开就说明单纯直方图不够，别急着调阈值，先加持续性/空间校验。
3. **标定 `t_abs`**：用 `python -m sbd calibrate dataset/`，
   数据集里**必须包含单镜头负样本**，否则标出来的阈值一定是错的。
4. **评估**：`python -m sbd eval dataset/`。正样本看 F1，负样本看 **FAR（次/分钟）**。
   一个视频 0 个真值切点时 F1 无定义，用 F1 会完全看不见误检。
5. **长视频**：>5000 帧时 PELT 自动跳步；阈值法本身是 O(n)，没有压力。
6. **上线前**：至少 20~30 段单镜头素材（手持、甩镜、变焦、黑场淡入淡出、长镜头跟拍），
   目标是**每个视频 ≤ 0.1 次误检**。

## 最小可用代码

```python
from sbd import DetectorConfig, detect_video, read_hists, detect

cuts = detect_video("input.mp4")          # 一行：读视频 → 出切点
for c in cuts:
    print(c.frame, c.kind, c.score)       # frame = 新镜头首帧索引

# 分步（需要自己控制直方图时）
hists, info = read_hists("input.mp4", DetectorConfig())
cuts = detect(hists, DetectorConfig(), info.fps)
```

命令行（无第三方依赖，只需 numpy；解码用 cv2 或 ffmpeg 或图片序列）：

```bash
python -m sbd detect input.mp4 -o cuts.json      # 单视频
python -m sbd eval dataset/                      # 批量评估
python -m sbd calibrate dataset/                 # 标定 t_abs
python -m sbd demo                               # 内置合成场景，无需视频
```

## 关键参数（默认值按 25fps / HSV 18×4×4 / L1 标定）

| 参数 | 默认 | 作用 | 什么时候动它 |
|---|---|---|---|
| `t_abs` | 0.38 | 绝对下限 | **换 bins 数必须重标定**；误检多就调高 |
| `k_mad` | 6.0 | 局部鲁棒阈值系数 | 运动剧烈就调高（8~10） |
| `min_shot_len_s` | 0.5 | 最短镜头 | 短视频/快剪就调低（0.2~0.3） |
| `persistence_k` | 8 | 持续性窗口 | 渐变多就调大 |
| `t_persist` | 0.28 | 持续性下限 | 与 `t_abs` 同向调整 |
| `gradual_total` | 0.35 | 渐变累积变化下限 | 漏渐变就调低，误检渐变就调高 |
| `bins` | (18,4,4) | 量化粒度 | 漏检相似镜头→增大；噪声大→减小 |

**调参顺序**：先 `t_abs`（决定误检），再 `k_mad`，最后 `min_shot_len_s`。
不要一上来调 `k`——相对阈值怎么调都救不了单镜头。

## 反模式（不要做）

- ❌ 只用 `μ+kσ` 或 `median+k·MAD` 而不加绝对下限 → 单镜头必被切碎
- ❌ 用 `mean/std` 做局部统计 → 切点尖峰把阈值抬高，导致漏检
- ❌ 用 F1 评估全部素材 → 负样本上的误检完全不可见
- ❌ 拿单帧最大差判断渐变转场 → 叠化每帧只变 0.03，根本过不了阈值
- ❌ 换 `bins` 后沿用旧 `t_abs` → 噪声底变了，阈值必须重标
- ❌ 「宁错勿漏」 → 这个场景恰好相反：切漏了用户能补一刀，切错了很刺眼，**要偏向保守**

## 参考文件

- `references/algorithm.md` —— 完整推导：直方图、六种距离、阈值策略、渐变、变点、复杂度
- `references/tuning.md` —— 阈值标定 SOP、单镜头问题、FAR 评估协议、参数敏感度
- `references/checklist.md` —— 失败模式排查表（现象 → 原因 → 处理）
- `sbd/` —— 实现；`tests/` —— 65 个单元测试；`examples/benchmark.py` —— 一键复现消融实验

## 输出约定

切点 `frame` 是**新镜头的第一帧**，边界落在 `frame-1` 与 `frame` 之间；
JSON 里同时给出 `cuts` 和 `shots: [[start, end_exclusive], ...]`，
后者可以直接喂给剪辑软件或前端时间轴。
