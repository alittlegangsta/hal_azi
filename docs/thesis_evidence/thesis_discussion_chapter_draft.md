# Thesis Discussion Chapter Draft

Generated: 2026-07-08. This draft freezes the discussion framing after EXP-008 and EXP-007 depth-heldout evidence.

## 7.1 方位失配与标签设计

XSI 声波数据和 CAST 胶结图像之间存在方位失配，Relative Bearing 等方位信息在旧实验中没有形成可靠的点对点监督基础。因此，本文不采用直接方位像素级匹配作为主线，而是将 CAST Zc 转化为弱监督标签。1D percentage label 通过方位平均得到深度剖面，降低了方位对齐要求；FFT severity label 进一步保留方位分布的频域结构，同时丢弃相位以获得旋转不变性。

从最终实验看，FFT severity label 更适合作为方法创新主线。它既保留了空间/频域结构，又与方位不可靠这一问题定义直接相连。1D percentage label 适合作为 fallback，因为它更直观，但 depth-heldout 结果没有比 EXP-008 更稳。

## 7.2 随机划分结果的证据边界

旧项目中 EXP-006、EXP-007、EXP-008 等结果大多来自随机或 shuffle 后的验证划分。由于样本沿连续深度采集，相邻深度样本在声波响应和 CAST 标签上可能高度相似，随机划分会带来相邻深度泄漏风险。因此，旧随机划分指标不能写成最终泛化性能。

这些旧结果仍有价值：EXP-006 的高 validation AUC 和 accuracy 可以说明 CWT 中包含可学习信号；旧 EXP-007/EXP-008 的训练曲线、散点图和 Grad-CAM 结果可以说明方法探索过程；失败路线可以放在讨论或附录中展示技术取舍。但最终性能表应以 depth-heldout 实验为准。

## 7.3 Depth-Heldout 结果的意义

EXP-008 和 EXP-007 的补充实验使用同一口井 `array_03` 的连续深度留出段。这个设置比随机验证更严格，因为测试集来自未参与训练的连续深度区间。但它仍然不是多井泛化，不能证明模型在不同井、不同仪器或不同地质环境下稳定。

EXP-008 的 depth-heldout 结果显示：模型对 FFT severity 标签有一定可学习性，测试 R2 为 `0.106634`，Pearson 为 `0.351950`，Spearman 为 `0.480519`。但是模型没有在 MAE 上超过 zero baseline。EXP-007 的结果更弱，R2 为 `-0.011278`，同样没有在 MAE 上超过 zero baseline。这说明当前标签分布稀疏，简单零预测在绝对误差指标上具有天然优势。

## 7.4 误差来源

EXP-008 的主要误差来自低频 FFT 系数和高严重度样本低估。低频系数对应整体严重度和大尺度方位结构，因此对模型来说更难但也更关键。高频系数绝对误差较低，不能简单解释为模型对高频结构更强，因为高频目标幅值本身更小。

EXP-007 的误差集中在 1D profile 的 shallow/mid index 区间，而 deep 区间误差接近零可能反映标签稀疏或接近零。对于这一路线，不能把低误差区间写成模型理解能力更强，而应写成标签分布导致的指标特性。

## 7.5 失败路线的讨论价值

旧项目中的 SE-ResNet 方位匹配、dual-channel metadata fusion、eccentricity pre-correction、GAN/two-channel binary label、sample weights + asymmetric loss 等路线不适合作为主线。它们的价值在于说明为什么最终选择弱监督标签和 CWT + EfficientNet 路线：

- 直接方位匹配路线受 XSI/CAST 方位不可靠影响。
- 偏心预校正和元数据融合没有形成稳定收益。
- GAN/two-channel 路线证据不足，且训练/验证结论不适合主线。
- 样本权重与非对称损失没有解决高严重度预测困难。

这些内容宜放在讨论或附录，不宜占用结果章主体。

## 7.6 论文最终限制

本文结果应明确以下限制：

1. 最终 depth-heldout 证据只来自 `array_03` 单井。
2. 没有完成多井留出验证。
3. 旧随机划分结果只能作为 exploratory evidence。
4. EXP-008 和 EXP-007 均未在 MAE 上优于 zero baseline。
5. 高严重度样本和低频 FFT 系数仍是主要误差来源。
6. EXP-007 severity-group 细节需要在远程小文件复制后进一步核验。
7. Grad-CAM 证据主要是定性解释，不是因果或物理验证。

## 7.7 后续工作

最小后续工作不是继续训练更多模型，而是完善论文图件和证据引用。若需要进一步提升研究完整性，优先级应为：多井或跨井 depth-heldout 验证、严重度分组统计、低频 FFT 系数定向误差分析、以及批量 Grad-CAM 统计。大规模重训、复杂新弱标签、STC/APES 或多目标人工审核不适合作为当前毕业论文的短期任务。

<!-- FIGURE_REDRAW_PACK_START -->
## Discussion Chapter Figure Placement Update (2026-07-08)

Use these redraw-pack figures in the discussion chapter:

| section | figure | discussion role |
| --- | --- | --- |
| 7.1 方位失配与标签设计 | `fig_01_xsi_cast_azimuth_mismatch_schematic.png`; `fig_03_percentage_label_construction.png`; `fig_04_fft_severity_label_construction.png` | Explain why direct azimuth supervision was avoided and why FFT magnitude is the mainline label. |
| 7.3 Depth-heldout 结果意义 | `fig_08_exp008_baseline_comparison.png`; `fig_11_exp007_baseline_comparison.png` | Compare metric-qualified improvements and zero-baseline MAE limitations. |
| 7.4 误差来源 | `fig_09_exp008_per_fft_coefficient_error.png`; `fig_12_limitation_high_severity_underestimation.png` | Discuss low-frequency FFT error and high-severity underestimation. |

Do not insert a generated EXP-007 prediction scatter until a per-sample EXP-007 prediction summary is copied or otherwise available as a small archived evidence file.
<!-- FIGURE_REDRAW_PACK_END -->
