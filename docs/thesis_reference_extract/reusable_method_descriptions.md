# Reusable Method Descriptions

Generated: 2026-07-09. Method wording is adapted from the reference materials but constrained by final evidence.

## CWT Feature Construction

Reusable wording:

> 针对声波全波列信号的非平稳和瞬态特征，本文采用连续小波变换（CWT）将一维时间序列转换为二维时频图谱。每个样本由多个接收器/方位通道构成，形成尺寸为 `150 x 400 x 8` 的 CWT 张量，用作 EfficientNet 回归模型输入。

Evidence alignment:

- Supported by proposal materials and final figure/data pack.
- Use as method description, not result claim.

## 1D Percentage Label

Reusable wording:

> 1D percentage 标签首先根据 CAST Zc 阈值 `Zc < 2.5` 构造窜槽掩膜，然后沿方位方向求平均，得到随深度变化的一维窜槽百分比剖面。该标签避免了直接方位点对点匹配，适合作为简单且直观的 fallback 标签路线。

Final thesis role:

- EXP-007 fallback / limitation comparison.
- Do not write as final mainline.
- Must state sparse-label and zero-baseline MAE limitation.

## FFT Severity Magnitude Label

Reusable wording:

> FFT severity 标签首先将 CAST Zc 转换为严重度图 `severity=max(0, 2.5-Zc)`，再沿方位维进行快速傅里叶变换，取幅度谱作为监督标签。由于环向旋转对应方位序列的循环平移，而傅里叶幅度对平移不敏感，该标签能够在丢弃相位的同时保留一定环向结构频率信息，从而降低对绝对方位匹配的依赖。

Final thesis role:

- EXP-008 method innovation mainline.
- Safe claim: supports single-well depth-heldout learnability.
- Unsafe claim: fully solves azimuth mismatch or preserves complete azimuth information.

## EfficientNetV2B0 Regression Model

Reusable wording:

> 本文采用 EfficientNetV2B0 作为 CWT 图像特征提取主干。由于 CWT 输入包含 8 个通道，模型前端使用 `1 x 1` 卷积进行通道适配，然后通过 EfficientNetV2B0、全局平均池化、Dropout 和全连接回归头输出对应标签。

Final-evidence note:

- Architecture descriptions should cite code/evidence.
- Do not claim deployment efficiency unless measured.

## Huber Loss And Robust Regression

Reusable wording:

> 回归训练采用 Huber loss，以降低异常标签或局部高误差样本对训练过程的影响。

Use note:

- This is method description only. It does not prove high-severity samples are solved.

## Artifact Masking

Reusable wording:

> 参考材料提出对 CWT 前 0.3 ms 的边缘伪影进行屏蔽，以减少模型学习非因果变换边缘的风险。最终论文可将其作为方法设计动机或已有代码路线说明。

Evidence caveat:

- Only state artifact masking where code/evidence confirms it for the specific run.
- Do not imply all experiments used identical masking unless verified.

## Grad-CAM Interpretability

Reusable wording:

> Grad-CAM 用于将模型关注区域映射回 CWT 时频图，从定性角度检查模型是否关注与窜槽相关的时间-频率区域。

Safe phrasing:

- “定性显示模型关注某些高频/早到波区域。”
- “为敏感时频区域分析提供线索。”

Unsafe phrasing:

- “证明模型学到物理规律。”
- “验证物理因果关系。”

## Failure Route Wording

Reusable wording from final project report:

- Dual-input metadata fusion attempted to use eccentricity metadata together with CWT, but did not learn effective features.
- Eccentricity pre-correction attempted to clean contaminated waveforms using metadata-derived correction, but final performance was poor.

Thesis placement:

- Discussion/appendix only.
- Use as method-selection rationale, not main result.
