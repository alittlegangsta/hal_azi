# Thesis Results Chapter Draft

Generated: 2026-07-08. This is a writing draft, not a new experiment report. All performance claims must cite the evidence files listed in `final_thesis_metrics_table.csv`.

## 5.1 实验设置与结果证据分级

本章将实验结果分为两类。第一类是旧项目中保存的随机划分实验结果，主要用于说明方法探索过程和特征可学习性；由于样本来自连续深度井段，随机划分可能导致相邻深度样本泄漏，因此这些结果不作为最终泛化性能。第二类是补充完成的 single-well depth-heldout 实验，其中训练集、验证集和测试集按照深度连续区间显式划分，并在 split audit 中确认不同集合的 record index 互斥、深度范围不重叠且边界设置了约 5 ft 的缓冲。

最终结果以 EXP-008 为方法主线，以 EXP-007 为 fallback 对照，以 EXP-006 为二分类 baseline 背景。EXP-008 与 EXP-007 均只覆盖 `array_03` 单井连续深度留出段，不代表多井泛化或工业部署性能。

## 5.2 EXP-006: CWT 二分类 baseline

EXP-006 使用 8 通道 CWT 时频图预测二分类窜槽标签。旧实验在随机验证设置下取得 validation AUC `0.95361`，validation accuracy `0.885366`。该结果说明 CWT 特征与窜槽存在性之间具有可学习关系，也支持后续使用 CWT 图像作为回归模型输入。

该结果的写作边界很明确：只能表述为随机划分下的可学习性 baseline。由于 split forensic 显示旧训练管线使用 shuffle 后的 take/skip 或类似随机验证方式，不能把 EXP-006 写成 depth-heldout 泛化结果，也不建议在最短论文路径中再补 EXP-006 depth-heldout 训练。

## 5.3 EXP-008: FFT severity label 主线结果

EXP-008 使用 CWT 输入和 EfficientNetV2B0 回归模型，目标标签为基于 CAST Zc 的 severity transform 及方位维 FFT magnitude/log 系数。该标签路线的核心动机是避免直接点对点方位监督，将方位信息转化为旋转不变的频域幅值表征。

在 single-well depth-heldout split 上，EXP-008 train_v001 的测试结果为：MAE `0.079025`，RMSE `0.277019`，R2 `0.106634`，Pearson `0.351950`，Spearman `0.480519`，测试样本数 `423`。验证集 MAE 为 `0.042817`，RMSE 为 `0.187167`，R2 为 `0.010882`。EarlyStopping 在第 12 轮停止并恢复第 2 轮权重，训练曲线显示较早出现过拟合。

与简单基线相比，EXP-008 的结论是 metric-qualified 的。模型优于 train-mean 和 train-median baseline 的 MAE/RMSE/R2，也优于 zero baseline 的 RMSE 和 R2；但模型 MAE `0.079025` 高于 zero baseline MAE `0.069751`。因此正文应写为：EXP-008 在单井 depth-heldout 条件下显示 FFT severity 标签具有一定可学习性，但稀疏标签使零预测在 MAE 上很强，模型不能被描述为在所有指标上优于简单基线。

误差结构进一步显示，低频 FFT 系数的绝对误差高于高频系数。low-frequency `k=0-5` MAE 为 `0.128785`，high-frequency `k=15-29` MAE 为 `0.050184`。结合高严重度样本低估现象，论文应将低频/高严重度预测不足作为主要限制之一。

## 5.4 EXP-007: 1D percentage label fallback

EXP-007 使用同样的 CWT + EfficientNetV2B0 建模框架，但目标标签改为沿方位平均的一维窜槽百分比剖面。该路线的优点是标签含义直观，能够作为 EXP-008 的 fallback 对照；缺点是标签更稀疏，zero baseline 在 MAE 上较强。

在 single-well depth-heldout split 上，EXP-007 train_v002 是最佳完成 run。测试结果为：MAE `0.811672`，RMSE `2.832056`，R2 `-0.011278`，Pearson `0.198048`，Spearman `0.413335`，测试样本数 `423`。它优于 train-mean 和 train-median baseline 的 MAE/RMSE/R2，也优于 zero baseline 的 RMSE/R2；但它没有优于 zero baseline 的 MAE，模型 MAE `0.811672` 高于 zero baseline MAE `0.738339`。

因此，EXP-007 不适合作为比 EXP-008 更强的主结果。它更适合作为 fallback/limitation comparison：一维百分比标签确实可以训练，但在单井深度留出测试中表现仍受稀疏标签和早期过拟合限制。profile index 误差显示 shallow `0-9` 区间 MAE `2.790083`、mid `10-34` 区间 MAE `1.151847`、deep `35-69` 区间 MAE `0.003429`；deep 区间低误差更可能反映标签接近零，而不是模型在该区间天然更强。

## 5.5 最终对比与主线选择

EXP-008 保留为论文方法创新主线，原因有三点。第一，FFT magnitude label 直接回应 XSI 与 CAST 方位失配问题，比 1D percentage label 更贴合研究问题。第二，EXP-008 在 depth-heldout 测试集上取得正 R2，而 EXP-007 的 R2 略为负。第三，EXP-008 的误差结构可以自然引出频域标签、低频系数和高严重度样本的讨论。

EXP-007 应放在 EXP-008 之后作为 fallback 对照，说明更直观的一维标签同样可训练但并未解决稀疏标签下的 zero-baseline MAE 问题。EXP-006 放在前面作为 CWT 可学习性 baseline，不参与最终回归主线性能排序。

本章最终应避免“模型已达到工程应用精度”或“泛化性能充分验证”等表述。更安全的结论是：在 `array_03` 单井连续深度留出条件下，CWT + EfficientNet 能从 XSI 时频特征中学习到与 CAST 窜槽标签相关的结构，FFT severity 标签路线较 1D percentage fallback 更适合作为本文主线，但仍存在稀疏标签、低频系数误差和高严重度低估等限制。
