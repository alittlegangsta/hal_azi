# Reusable Background Points

Generated: 2026-07-09. These are writing points extracted from the reference materials and filtered against the final evidence freeze.

## Cementing And Channeling Background

- 固井质量评价是油气井全生命周期管理中的关键环节。水泥环窜槽会造成层间流体连通、产量下降和井筒完整性风险。
- 套管井固井质量评价通常依赖声波测井和超声成像测井。二者提供的信息互补，但数据形态、分辨率和方位属性不同。
- 传统声波固井评价常依赖首波幅度、特定时窗能量或 CBL/VDL 等低维特征，难以充分利用全波列中的散射、干涉和时频结构信息。

## Sonic Logging Versus CAST

- 声波测井具有覆盖范围广、成本相对低、适合全井段普查等优点。
- 声波全波列信号成分复杂，包含多种模式波、反射/散射成分和井眼环境干扰，因此直接人工判读或简单特征提取存在局限。
- CAST/超声成像测井具有较高空间分辨率，能够提供环向水泥胶结图像，可作为构造弱监督标签的重要参考。
- CAST 的成本、作业条件和覆盖范围限制，使得“用 CAST 监督声波模型、再用声波进行更广范围评价”的思路具有工程动机。

## Why Deep Learning Is Useful

- 深度学习模型适合从高维时频图中学习非线性映射，可用于挖掘传统低维指标没有显式表达的窜槽相关特征。
- EfficientNetV2B0 的优势可以表述为“高效图像特征提取主干”，但不要在没有专门实验的情况下夸大参数效率或部署性能。
- 可解释性分析可以帮助检查模型是否关注合理的时频区域，使模型结果更容易被测井解释人员审查。

## Safe Literature-Framing Points

- 井孔声学理论、声波测井定量解释、小波时频分析、深度残差/高效卷积网络、Grad-CAM 可解释性方法构成本文的基础。
- 参考材料列出的文献方向可用于文献综述骨架：Biot 多孔弹性理论、套管井声场模拟、定量声波测井、小波变换、ResNet/EfficientNet、Grad-CAM。
- 写作时应避免把文献综述直接写成“本文已达到工程应用”，而应落到“本文探索 CWT-EfficientNet 与弱监督标签构造的可行性”。

## Recommended Background Paragraph Logic

1. 固井质量与窜槽风险引出问题重要性。
2. 声波测井和 CAST 的互补性引出“用高分辨 CAST 监督低成本声波”的数据基础。
3. 方位失配使直接像素/方位点对点监督不可行，引出弱标签路线。
4. 声波非平稳全波列适合 CWT 表征，引出 CWT + EfficientNet。
5. FFT magnitude 丢弃相位，适合构造旋转不变标签，引出 EXP-008 主线。

## Mandatory Caveat

Background may discuss engineering motivation, but final thesis result claims must remain scoped to `array_03` single-well depth-heldout evidence.
