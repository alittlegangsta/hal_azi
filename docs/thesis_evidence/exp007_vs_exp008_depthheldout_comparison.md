# EXP-007 vs EXP-008 Depth-Heldout Comparison

Generated: 2026-07-07. Both experiments are single-well `array_03` depth-heldout experiments and should not be described as multi-well generalization.

| experiment | best run | target label | test MAE | test RMSE | test R2 | Pearson | Spearman | thesis role |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| EXP-008 | train_v001 | FFT severity magnitude/log coefficients | 0.079025 | 0.277019 | 0.106634 | 0.351950 | 0.480519 | method mainline |
| EXP-007 | train_v002 | 1D channeling percentage profile | 0.811672 | 2.832056 | -0.011278 | 0.198048 | 0.413335 | fallback / limitation comparison |

## Answered Questions

1. EXP-007 是否比 zero/train-mean baseline 更稳：
   Mixed. It beats zero by RMSE/R2 but not MAE; it beats train-mean/train-median by MAE/RMSE/R2.

2. EXP-007 是否比 EXP-008 更适合作为实证 fallback：
   It is useful as fallback evidence, but not stronger than EXP-008. EXP-008 remains the main method result because it has positive R2 and addresses azimuth-invariant FFT labeling.

3. EXP-008 是否仍作为方法创新主线：
   Yes. Use EXP-008 as the method innovation mainline, with metric-qualified baseline caveats.

4. 论文主线如何组织：
   Present EXP-008 as the main CWT + EfficientNet + FFT severity-label route. Present EXP-007 as a simpler 1D percentage-label fallback that confirms sparse-label difficulty under depth holdout.

5. 是否需要再训练 EXP-006：
   Not recommended for the shortest thesis path. EXP-006 may be useful only if the thesis needs a binary-detection safety net, but it would not fix the regression-label sparsity limitation.

6. 是否可以停止实验进入论文写作：
   Yes, if the result chapter is framed as single-well method feasibility plus limitations rather than strong deployment/generalization performance.
