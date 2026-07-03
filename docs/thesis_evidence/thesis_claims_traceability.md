# Thesis Claims Traceability

Every claim below is tied to at least one result path, code path, git branch/commit, memo path, or explicit inference marker.

| claim_id | claim | evidence | status |
| --- | --- | --- | --- |
| C-001 | CWT 时频图与窜槽存在性之间存在可学习关系。 | EXP-006; temp_result/test_relativity/result.txt.txt; memo AUC=0.95361 | supported |
| C-002 | 1D 窜槽百分比标签规避方位失配并得到较稳定敏感区域。 | EXP-007; origin/1D+percentage_Label code; memo section 6 | supported, metric table still needed |
| C-003 | FFT 幅值/丢弃相位是处理方位旋转不变性的合理标签路线。 | memo PPT section; create_tfrecords FFT code | method rationale supported, performance needs verification |
| C-004 | log transform improves Grad-CAM localization but does not solve prediction quality. | EXP-002; memo section 1; temp_result/log_label plots | supported by memo/result artifacts |
| C-005 | frequency-weighted FFT loss did not solve low-coefficient collapse. | EXP-003; memo section 2; origin/frequency-weighted_loss train.py | supported by memo/code |
| C-006 | GAN/two-channel image-generation routes collapsed. | EXP-004; EXP-005; memo sections 3-4; result.txt losses | supported for GAN, two-channel implementation mapping needs verification |
| C-007 | Dual-channel and pre-correction failed/poor. | result directories 双通道学习 and 预校正; user task framing | explicitly_marked_inference_needs_verification |
| C-008 | Sensitive CWT region is concentrated in high-frequency bands roughly 22-30 kHz and 0.5-1.3 ms depending on route. | memo sections 1/2/6; temp_result/test_relativity/result.txt.txt; Grad-CAM result files | supported qualitatively; batch statistics recommended |
