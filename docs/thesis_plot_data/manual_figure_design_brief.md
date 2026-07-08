# Manual Figure Design Brief

Generated: 2026-07-08. This directory is for manual redrawing in Origin/PPT/AI/Figma. It intentionally does not contain final thesis-style generated figures.

## Global Rules

- Distinguish `random split exploratory` from `single-well depth-heldout`.
- Do not claim multi-well generalization or industrial deployment.
- Do not use values not present in the CSV files in this directory or the cited evidence paths.
- For method schematics, use them as conceptual diagrams only; do not add performance numbers.

## 1. XSI/CAST 方位失配示意图

图题：XSI 与 CAST 方位失配问题示意图

必须包含：
- 左侧 XSI 声波接收器/波形或 CWT 输入。
- 右侧 CAST Zc 方位图。
- 深度方向可对齐，但方位零点/旋转角未知。
- 标注“直接点对点方位监督不可靠”。
- 引出两条弱标签路线：1D percentage 和 FFT magnitude。

推荐布局：
- 左右对照布局：XSI 在左，CAST 在右，中间画深度对齐箭头和方位旋转偏移箭头。
- 底部放“weak label strategy”小结。

推荐颜色逻辑：
- XSI/CWT 用蓝色。
- CAST/Zc 用红色或橙色。
- 不确定方位偏移用灰色虚线或红色旋转箭头。

禁止表达：
- 不要写 Relative Bearing 永远无用。
- 不要写方位问题已经完全解决。
- 不要放任何模型性能指标。

## 2. 数据处理流程图

图题：XSI-CWT 与 CAST 弱标签数据构建流程

必须包含：
- XSI waveform -> CWT `150 x 400 x 8`。
- CAST Zc -> severity `max(0, 2.5-Zc)` -> FFT magnitude label。
- CAST Zc -> binary mask `Zc < 2.5` -> azimuth mean -> 1D percentage label。
- Explicit train/val/test TFRecord split。
- EfficientNetV2B0 regression。

推荐布局：
- 上支路为 XSI 特征，下支路为 CAST 标签，中间在 TFRecord split 处汇合。
- EXP-008 主线用实线或强调色，EXP-007 fallback 用次级色。

禁止表达：
- 不要写 validation_split。
- 不要暗示 EXP-006 是最终性能主线。

## 3. Percentage Label 构造图

图题：一维窜槽百分比标签构造流程

必须包含：
- CAST Zc depth-window slice。
- 阈值 `Zc < 2.5` 的二值掩膜。
- 沿方位维求平均，得到每个深度位置的窜槽百分比。
- 输出 70 点 profile。

推荐布局：
- 四步横向流程：Zc slice -> mask -> azimuth mean -> 1D profile。

禁止表达：
- 不要写该标签保留完整方位信息；它丢弃了方位结构。
- 不要把 EXP-007 写成比 EXP-008 更强。

## 4. FFT Severity Label 构造图

图题：FFT severity magnitude 标签构造流程

必须包含：
- CAST Zc。
- severity transform: `severity = max(0, 2.5 - Zc)`。
- 沿方位维做 FFT。
- 取 magnitude/log magnitude，phase discarded。
- 输出 70 x 30 FFT label map。

推荐布局：
- 横向流程，最后用频谱条形或热图表示低频到高频系数。
- 明确标注“phase discarded for rotation invariance”。

禁止表达：
- 不要写 FFT magnitude 保留完整方位相位信息。
- 不要写旋转不变标签已完全解决所有方位问题。

## 5. CWT-EfficientNet 架构图

图题：CWT-EfficientNet 回归模型结构示意

必须包含：
- CWT input `150 x 400 x 8`。
- 1x1 Conv channel adapter `8 -> 3`。
- EfficientNetV2B0 backbone。
- Global Average Pooling。
- Dropout。
- Dense regression head。
- 两种输出：EXP-008 `70 x 30` FFT label；EXP-007 `70` percentage profile。
- 训练输入为 explicit train/val/test TFRecord，不使用 validation_split。

推荐布局：
- 主干横向网络结构，输出处分叉到 EXP-008 和 EXP-007。

禁止表达：
- 不要加入未核验参数量、FLOPs 或部署速度。
- 不要把架构图作为性能证明。
