# Risk Register

| risk_id | risk | impact | mitigation |
| --- | --- | --- | --- |
| R-001 | 把 memo 定性结论写成未证实的定量结论 | high | 所有结论引用 memo/result/code/git；没有指标则写 unknown/needs_verification。 |
| R-002 | train/validation 深度泄漏 | high | 补 depth-blocked split 说明或承认为限制。 |
| R-003 | 结果目录与代码分支映射错误 | medium | 用 Git commit、result modified_time、服务器目录核对。 |
| R-004 | FFT 主线性能不足 | medium | 把 1D percentage EfficientNet 作为强证据主结果，FFT 作为角度不匹配方法探索/局限。 |
| R-005 | 为了补证据触发大规模重训 | medium | 当前阶段只做整理和轻量验证；训练需单独审批。 |
