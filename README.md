# Shot Boundary Detection (直方图分镜)

[![python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![license](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

用**相邻帧直方图差分**做视频分镜（镜头边界 / shot boundary detection）。

很多人第一版都会写成「算每帧与上一帧的颜色直方图差异，超过阈值就是切镜」，
然后发现：阈值怎么调都不对，**单镜头视频也被切成了好几段**。
这个仓库是那个算法的**正确版本**，包含：

- ✅ 绝对下限 + 鲁棒局部阈值（median/MAD，不是 mean/std）
- ✅ 持续性检验（干掉闪光、甩镜、瞬态遮挡）
- ✅ 渐变转场检测（累积变化 + 单向性 + 结束静默）
- ✅ PELT 变点检测（惩罚项让「0 个切点」成为合法解）
- ✅ 空间校验（SSIM / ECR / 分块直方图）
- ✅ 正确的评估口径（F1 + **FAR**，单镜头视频不能只看 F1）
- ✅ 65 个单元测试 + 无需真实视频的合成基准

## 为什么「只做相对阈值」一定会坏

设 `d[i]` 是第 `i` 帧与上一帧的直方图距离。相对阈值（`μ+kσ`）是**尺度无关**的：
把 `d` 标准化以后，总有一部分帧落在尾部。10 分钟 30fps 有 18000 帧，
即使尾部概率只有 `1e-4`，也会冒出近 2 个「异常点」——于是单镜头视频被切成几段。

而单镜头视频里没有「切点样本」，双峰分布退化成单峰，
你**没有任何依据**判断阈值该定在哪。这就是根本矛盾。

解法：给阈值加**绝对下限**，再加**持续性检验**。
详见 [`references/algorithm.md`](references/algorithm.md)。

## 安装

```bash
pip install numpy            # 唯一必需依赖
pip install opencv-python    # 解码视频（或用系统 ffmpeg，二选一）
```

```bash
git clone <your-repo-url>
cd shot-boundary-detection
pip install -e .
```

## 快速开始

```python
from sbd import DetectorConfig, detect_video

cuts = detect_video("input.mp4")
for c in cuts:
    print(f"帧 {c.frame}  类型={c.kind}  得分={c.score:.3f}")
```

```bash
python -m sbd detect input.mp4 -o cuts.json
```

输出：

```json
{
  "video": "/abs/path/input.mp4",
  "fps": 25.0,
  "n_frames": 1500,
  "cuts": [
    {"frame": 312, "score": 0.71, "kind": "hard"},
    {"frame": 900, "score": 0.44, "kind": "gradual", "detail": {"length": 21, "coherence": 0.98}}
  ],
  "shots": [[0, 312], [312, 900], [900, 1500]]
}
```

## 命令行

```bash
python -m sbd detect   <video|frames_dir> [-o out.json] [--mode threshold|changepoint|hybrid] [--verify]
python -m sbd eval     <dataset_dir> [--tol 2] [-o report.json]
python -m sbd calibrate <dataset_dir> [--alpha 1e-4]
python -m sbd demo
```

`detect` 支持视频文件、图片序列目录；解码后端自动选择 `cv2` → `ffmpeg` → `Pillow`。

数据集约定：每个视频旁边放一个同名 `.json`（`clip01.mp4` + `clip01.json`），
内容 `{"cuts": [120, 480], "fps": 25}`；单镜头视频写 `{"cuts": []}`。

## 三种模式

| 模式 | 原理 | 适用 |
|---|---|---|
| `threshold`（默认） | 绝对下限 + 鲁棒局部阈值 + 持续性检验 | 快，通用 |
| `changepoint` | PELT 分段，BIC 惩罚 | 单镜头/长镜头多，追求零误检 |
| `hybrid` | 两者并集 | 召回优先，事后人工确认 |

## 合成基准

`python -m sbd demo` 不需要任何视频文件即可验证算法行为：

| 场景 | 真值切点 | 检测 | 命中 | 误检 | 漏检 |
|---|---|---|---|---|---|
| 4 个镜头硬切 | 4 | 4 | 4 | 0 | 0 |
| 单镜头（含重尾噪声） | 0 | 0 | – | 0 | 0 |
| 单镜头 + 2 次白闪 | 0 | 0 | – | 0 | 0 |
| 两镜头叠化 | 1 | 1（gradual） | 1 | 0 | 0 |

消融实验（`python examples/benchmark.py --seeds 10`，数字为 10 次运行合计）：

| 配置 | 硬切 命中/漏检 | 单镜头 误检 | 白闪 误检 | 叠化 命中/误检 |
|---|---|---|---|---|
| 纯相对阈值（`t_abs=0`，无持续性） | 40 / 0 | **66** | 20 | 6 / 14 |
| ＋绝对下限 | 40 / 0 | **0** | 20 | 0 / 0 |
| ＋持续性检验 | 40 / 0 | 0 | **0** | 0 / 0 |
| 完整版（含渐变） | 40 / 0 | 0 | 0 | **10 / 0** |

10 段单镜头视频合计 66 次误检 —— 平均每个视频 6.6 次。
这就是「只做相对阈值」在真实素材上的表现。

## 评估：为什么必须看 FAR

单镜头视频的 ground truth 切点数为 0，此时 `precision=0`、`recall` 无定义、**F1 无定义**。
用 F1 汇总会得出荒谬结论。所以本项目同时输出：

- 正样本：`Precision / Recall / F1`
- 负样本：`FAR = 误检次数 / 分钟`、`零误检片段占比`
- 全数据集：`overall FAR`

目标：**每个单镜头视频 ≤ 0.1 次误检**（即 10 个视频里最多 1 个出现 1 次误检）。

## 项目结构

```
sbd/
  hist.py         直方图：RGB→HSV、量化、归一化（纯 numpy）
  distance.py     6 种距离：L1 / 交集 / L2 / χ² / Bhattacharyya / 余弦
  detector.py     主检测器：绝对下限 + median/MAD + 持续性 + 渐变
  changepoint.py  PELT 变点检测 + BIC 惩罚（含跳步加速）
  verify.py       空间校验：SSIM / ECR / 分块直方图
  video.py        解码：cv2 / ffmpeg / 图片序列，统一定点取帧
  metrics.py      评估：带容差匹配、F1、FAR、数据集汇总
  synth.py        合成数据：硬切/单镜头/闪光/叠化 + RGB 帧
  cli.py          detect / eval / calibrate / demo
tests/            65 个单元测试（含单镜头零误检回归）
examples/         合成基准 + 消融实验
references/       算法推导、调参 SOP、排查清单
```

## 已知局限

- 直方图**丢失空间信息**：构图不同但配色接近的两个镜头会漏检 → 开 `--verify` 加空间校验。
- **渐变转场**在剧烈运动的镜头里很难只靠直方图检出 → 需要空间特征，本项目对此保守处理（宁可漏）。
- **一镜到底但有大范围光照变化**（开灯、日落）可能误检 → 用 HSV 的 H/S 为主、降低 V 权重。
- 首尾黑场/淡入淡出会被单独处理，不算切点。

## License

MIT，见 [LICENSE](LICENSE)。
