# Reusable Problem Definition And Significance

Generated: 2026-07-09.

## Reusable Problem Definition

本文可以将核心问题定义为：

> 在垂直井固井质量评价中，阵列声波测井 XSI 与高分辨率 CAST 胶结成像在深度、分辨率和方位参考上存在差异。尤其是在声波仪器与超声仪器相对方位角不可靠的条件下，难以直接构建点对点方位监督标签。本文关注如何利用 CAST 构造对方位失配更鲁棒的弱监督标签，并从 XSI 声波 CWT 时频特征中学习水泥窜槽相关结构。

## Reusable Key Problems

| problem | reusable wording | final-evidence-safe caveat |
| --- | --- | --- |
| 方位失配 | 垂直井中 XSI 与 CAST 的相对方位角不稳定，Relative Bearing 在部分条件下难以支撑可靠点对点监督。 | 不要写 Relative Bearing 永远无用；写“旧实验未能支持可靠点对点方位监督”。 |
| 标签构造 | CAST Zc 图像需要转换为适合声波模型学习的弱监督标签。 | 最终采用 EXP-008 FFT severity magnitude 主线，EXP-007 1D percentage fallback。 |
| 声波信号复杂 | XSI 全波列含多模式波和噪声，传统低维指标难以覆盖全部信息。 | 不要声称模型已解决所有噪声和复杂工况。 |
| 可解释性 | 深度模型需要通过 Grad-CAM 等方法检查关注区域，提高结果可审查性。 | Grad-CAM 只能作为定性解释，不是物理因果证明。 |

## Reusable Research Significance

### Engineering Significance

- 若能从覆盖范围更广、成本更低的声波测井中学习 CAST 相关胶结信息，可为固井质量评价提供辅助工具。
- 方位失配是多源测井监督学习的重要障碍。通过 FFT magnitude 等旋转不变标签降低对绝对方位的依赖，具有实际问题针对性。
- 通过显式区分随机划分探索性结果和 depth-heldout 结果，论文能够更严谨地评估模型在连续深度留出段上的可学习性。

### Scientific Significance

- 将非平稳声波全波列转化为 CWT 时频图，有助于从时间-频率联合域分析窜槽相关敏感特征。
- FFT magnitude 标签将环向 CAST 信息转化为频域幅值结构，为方位不可靠条件下的弱监督学习提供一种信号处理思路。
- Grad-CAM 可用于定性检查模型对 CWT 时频区域的关注，为后续声学机理分析提供线索。

## Safe Objective Wording

- 构建基于 CWT 时频特征和 CAST 弱监督标签的水泥窜槽识别/严重度预测方法。
- 研究方位失配条件下的 1D percentage 标签与 FFT magnitude 标签构造路线。
- 验证 CWT + EfficientNet 在单井 depth-heldout 设置下对 FFT severity 标签的可学习性。
- 分析模型误差结构和可解释性结果，明确方法限制。

## Avoid These Proposal-Stage Objective Words

- “精准反演”
- “像素级对齐”
- “超声级洞察力”
- “验证物理因果”
- “实现低成本全井段精细评价”

Use only if heavily softened and tied to future work or motivation.
