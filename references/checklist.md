# 排查清单

## 症状 → 原因 → 处理

| 症状 | 最可能的原因 | 处理 |
|---|---|---|
| 一镜到底的视频被切成 2~5 段 | 只用了相对阈值，没有绝对下限 | `t_abs` 设为标定值（默认 0.38） |
| 每个镜头开头都多切一刀 | 峰值抑制没做，一个切点被检多次 | `local_max_gap = 4` |
| 闪光、爆炸被当成切镜 | 没做持续性/闪光检验 | 开 `check_persistence`、`suppress_flash` |
| 甩镜、快速摇摄被当成切镜 | 持续性检验的 SNR 太松 | `persist_snr` 提到 2.5~3.0 |
| 淡入淡出、叠化完全漏检 | 渐变检测没开，或 `gradual_total` 太高 | 开 `detect_gradual`，`gradual_total` 降到 0.25 |
| 整段运动镜头被当成渐变 | 缺“结束后静默”判据 | 确认 `gradual_post_quiet` 生效 |
| 两个配色相近的镜头漏检 | 直方图无空间信息 | 开 `--verify`（SSIM/ECR/分块） |
| 换了 `bins` 后效果变差 | 噪声底变了，`t_abs` 没重标 | 重跑 `calibrate` |
| 首尾几帧附近误检 | 窗口边缘填充把阈值带偏 | 确认用的是 NaN 填充 + 有效样本统计 |
| 片头黑场被当成切点 | 黑场未排除 | 检测平均亮度，排除近黑段 |
| 长视频跑得很慢 | PELT 精确模式是 $O(n \cdot |R|)$ | 用默认自动 `jump`，或改用 `threshold` 模式 |

## 必做的 4 个诊断

### 1. 画 `d` 的时间序列

```python
import numpy as np
from sbd import DetectorConfig, read_hists, diff_series

cfg = DetectorConfig()
hists, info = read_hists("input.mp4", cfg)
d = diff_series(hists, cfg.metric)

print("median", np.median(d[1:]), "p99", np.quantile(d[1:], 0.99), "max", d.max())
```

- 正样本应有清晰的双峰：绝大多数点 < 0.1，切点 > 0.4；
- 负样本应该是「带毛刺的平地」，**最大值不应该接近 0.38**；
- 如果负样本的 max 已经 > 0.3，说明这段素材本身有剧烈运动，
  阈值必须靠标定而不是拍脑袋。

### 2. 打印候选与拒绝原因

```python
from sbd.detector import detect_hard_cuts, DetectorConfig

cfg = DetectorConfig(suppress_flash=False, check_persistence=False)
raw = detect_hard_cuts(hists, cfg, info.fps)   # 只看阈值筛出来的
full = detect_hard_cuts(hists, DetectorConfig(), info.fps)  # 完整
print("阈值候选:", [c.frame for c in raw])
print("最终保留:", [c.frame for c in full])
```

差集就是被持续性/闪光检验干掉的——如果真切点也在差集里，说明 `persist_snr` 太严。

### 3. 单镜头负样本回归

用 `tests/test_detector.py::test_single_shot_no_false_cuts` 的构造方式，
拿**你自己的素材**跑一遍，确认输出是空列表。这是最重要的一条。

### 4. 消融对照

```bash
python examples/benchmark.py --seeds 10
```

确认三件事：加绝对下限后负样本误检归零、加持续性后闪光归零、
完整版能检出渐变。如果本地化之后某一项不成立，说明参数被改坏了。

## 上线前检查

- [ ] 数据集包含 ≥ 20 段单镜头负样本（手持、甩镜、变焦、光照变化、长镜头跟拍）
- [ ] 报告里有 FAR（次/分钟）和零误检片段占比，不只有 F1
- [ ] `t_abs` 是用 `calibrate` 在你自己的 bin 配置下标定的
- [ ] 正样本上硬切与渐变**分开**统计
- [ ] 极端情况：纯黑视频、纯白视频、单色视频 → 不应崩溃、不应报切点
- [ ] 1 小时长视频跑一遍，确认耗时和内存可接受
- [ ] 输出 `shots` 区间连续、无重叠、覆盖 `[0, n_frames)`
