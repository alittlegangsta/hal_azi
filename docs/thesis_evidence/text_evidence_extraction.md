# Text Evidence Extraction

Readable text evidence was extracted from results text files and repository code/config/docs.

## Summary

- Result text files: `5`
- Repo text/code files: `27`
- Memo extraction status: `utf-8-sig`
- Memo claims: `7`
- Keyword snippets: `249`

## Memo Experiment Claims

| claim_id | section | method | model | target_label | evaluation_metrics | result_conclusion | failure_or_limitation | possible_result_dir |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| memo-01 | Baseline | FFT; CWT; Grad-CAM | Attention U-Net / A2INet | FFT magnitude label | unknown | ## Results: 目前**grad-cam**热力图效果很差，热力点分布分散，很难看出集中特征，且很多热力点分布并不合理 Predction**结果很差，只能预测"整体平均"的结构，基本上只对FFT第0个系数学习到了。且分布是**垂直条带 | 目前**grad-cam**热力图效果很差，热力点分布分散，很难看出集中特征，且很多热力点分布并不合理 Predction**结果很差，只能预测"整体平均"的结构，基本上只对FFT第0个系数学习到了。且分布是**垂直条带 | temp_result/baseline |
| memo-02 | 1. 对数变换标签 | FFT; Grad-CAM; log label | unknown | FFT magnitude label | unknown | ## Results: 目前**grad-cam**热力图效果变好，热力点分布集中，目前注意力主要集中在**0.5-0.7ms，25-30kHz处**。 | unknown | temp_result/log_label |
| memo-03 | 2. FFT系数高频损失惩罚 | FFT; Grad-CAM; SSIM/PSNR; GAN/GaN | GAN | FFT magnitude label | ssim; psnr | ## Results: 目前**grad-cam**热力图没啥大变化，但是更加集中了，目前注意力主要集中在**0.75ms-0.85ms和25-28kHz处 | Predction**结果依旧很差，即使增加了FFT系数高频的惩罚，模型依旧只学习拟合FFT低系数，下一步引入GaN。 | temp_result/frequency-weighted_loss |
| memo-04 | 3. GaN+“窜槽严重性”变换 | FFT; log label; severity; GAN/GaN | GAN | FFT magnitude label; severity transform max(0, 2.5 - Zc); CAST Zc slice | unknown | ## Results: 模型崩塌 | 模型崩塌 | temp_result/GaN+2Dlabel |
| memo-05 | 4. GaN+"双通道二元标签"+焦点损失函数+过拟合测试 | CWT; dual-channel; severity; GAN/GaN | Attention U-Net / A2INet | binary channeling label/mask; severity transform max(0, 2.5 - Zc); CAST Zc slice | loss | ## Results: 模型崩塌 | 在这种情况下，训练在完整数据集上的失败，就说明问题在于**泛化能力**，即模型难以从2846个样本中归纳出普适的规律，这可能需要更强大的模型或更多的训练技巧。 模型崩塌 | temp_result/GaN+2Dlabel |
| memo-06 | 5. CNN+分类任务（验证CWT与标签之间是否存在对应关系） | CWT; AUC | CNN classifier | binary channeling label/mask; CAST Zc slice | auc; accuracy; val_auc; val_accuracy; loss; val_loss; accuracy≈85%; AUC=0.95361 | ## Results: 最关键的指标是验证集上的AUC（`val_auc`），它达到了**0.95361**的峰值。AUC是衡量二元分类模型性能的黄金标准，0.5代表随机猜测，1.0代表完美分类。0.95以上是一个**非常出色**的结果，它无可辩驳地证明了：**您的模型已经成功地学会了如何从CWT时频图中，区分出“有窜槽”和“无窜槽”的样本**。 训练准确率（`accuracy`）和验证准确率（`val_accuracy`）也都达到了约**85%**，这同样是一个非常积极的信号。 损失函数（`loss`和`val_loss`）在持续下降，而性能指标（`auc`和`accuracy`）在持续上升，这是一个非常健康的学习曲线。 训练在第72个epoch时**提前停止（early stopping）**，这也是一个好现象。这意味着模型在第57个epoch时在验证集上达到了最佳性能（`val_auc: 0.95361`），并在那之后自动保存了最佳模型。这有效地防止了模型在后续训练中发生过拟合，确保了我们得到的是泛化能力最强的模型。 目前的项目是一个判断有无窜槽的模型，但这并不是我最终想要的，目前这一阶段达成了验证”CWT时频图与水泥窜槽的存在与否之间，确实存在着强烈的、可以被机器学习模型稳定捕捉到的内在联系“这一结论。我的想法是，超声所测量的Zc声阻抗值和声波信号的数据的对应关系既然在方位上已经不匹配了，但是超声在每个深度点所测量的Zc值，取声波的收发路径为深度范围，该深度点的声波信号和超声在这一深度范围的数据是匹配的。那么，超声所反映的在深度方向上的窜槽信息，如结构窜槽分布（集中式缺陷或者分布式），窜槽严重程度等等，可以和声波的信号很好的对应起来。所以我希望的是这个模型最终能够通过声波信号最终尽可能预测中深度方向上的胶结情况（结构分布，严重程度等）。所以我一开始的想法也是做图像生成，因为图像携带的信息就包括了结构分布等信息 | unknown | temp_result/test_relativity |
| memo-07 | 6. 一维百分比窜槽比例标签 | FFT; CWT; EfficientNetV2B0; eccentricity/pre-correction; 1D percentage | EfficientNetV2B0 | 1D channeling percentage profile; FFT magnitude label; CAST Zc slice | unknown | ## Results: 不足之处：模型在定量预测窜槽的严重程度时，表现出一种与严重程度相关的、持续增强的低估倾向。真实窜槽越严重，模型的预测结果就越偏离真实值（越保守）。 FFT：阐述原理，数据处理，展示标签，展示输入输出。讲清楚为什么要这么做？FFT把角度对齐变成相移，取绝对值，**时域的时移相当于频率域相移，于是只取幅值，忽略相位信息，可以使数据匹配**。解释为什么效果不好：展示效果，角度上只有8个接收器，但我们想要区分的情况集中在低频，频率分辨率不足以区分这两种情况，一般来说，一个深度处不会出现多个不连续的窜槽。 | 对超声数据在方位上进行FFT所得来的标签并不理想，模型根本学不到东西 不足之处：模型在定量预测窜槽的严重程度时，表现出一种与严重程度相关的、持续增强的低估倾向。真实窜槽越严重，模型的预测结果就越偏离真实值（越保守）。 FFT：阐述原理，数据处理，展示标签，展示输入输出。讲清楚为什么要这么做？FFT把角度对齐变成相移，取绝对值，**时域的时移相当于频率域相移，于是只取幅值，忽略相位信息，可以使数据匹配**。解释为什么效果不好：展示效果，角度上只有8个接收器，但我们想要区分的情况集中在低频，频率分辨率不足以区分这两种情况，一般来说，一个深度处不会出现多个不连续的窜槽。 | temp_result/1D+percentage_Label; FFT_EfficientNet; FFT_EfficientNet_1 |

## Text Files With Keywords

| source_type | relative_path | read_status | keywords | line_count |
| --- | --- | --- | --- | --- |
| results | temp_result/1D+percentage_Label/result.txt | utf-8-sig | severity | 16 |
| results | temp_result/test_relativity/result.txt.txt | utf-8-sig | AUC | 11 |
| results | temp_result/改进memo.md | utf-8-sig | FFT; CWT; Grad-CAM; EfficientNetV2B0; dual-channel; eccentricity/pre-correction; log label; severity; 1D percentage; AUC; SSIM/PSNR; GAN/GaN | 155 |
| repo | config.py | utf-8-sig | FFT; CWT; log label | 89 |
| repo | main.py | utf-8-sig | CWT | 108 |
| repo | src/cwt_transformation/main_transform_translation.py | utf-8-sig | CWT | 99 |
| repo | src/data_processing/create_tfrecords.py | utf-8-sig | FFT; CWT; log label | 139 |
| repo | src/data_processing/main_preprocess.py | utf-8-sig | log label | 191 |
| repo | src/interpretation/debug_data_visualization.py | utf-8-sig | FFT; CWT; Grad-CAM | 65 |
| repo | src/interpretation/generate_analysis_candidates.py | utf-8-sig | CWT; Grad-CAM | 145 |
| repo | src/interpretation/grad_cam.py | utf-8-sig | Grad-CAM | 50 |
| repo | src/interpretation/run_analysis.py | utf-8-sig | FFT; CWT; Grad-CAM; log label | 161 |
| repo | src/interpretation/visualize_model.py | utf-8-sig | SE-ResNet | 77 |
| repo | src/modeling/dataset.py | utf-8-sig | FFT | 93 |
| repo | src/modeling/model.py | utf-8-sig | FFT; Grad-CAM | 107 |
| repo | src/modeling/train.py | utf-8-sig | FFT; log label | 131 |
| repo | src/utils/plotting.py | utf-8-sig | CWT; Grad-CAM; log label; severity | 59 |
| repo | src/visualization/visualize_data_pipeline.py | utf-8-sig | FFT; CWT; log label; 1D percentage | 122 |
| repo | src/visualization/visualize_training_history.py | utf-8-sig | log label | 61 |

## Representative Snippets

| source_type | relative_path | line | keywords | snippet |
| --- | --- | --- | --- | --- |
| results | temp_result/test_relativity/result.txt.txt | 5 | AUC | 验证集上的AUC（val_auc），达到了0.95361的峰值 |
| results | temp_result/test_relativity/result.txt.txt | 7 |  | 训练准确率（accuracy）和验证准确率（val_accuracy）也都达到了约85% |
| results | temp_result/test_relativity/result.txt.txt | 9 | AUC | 损失函数（loss和val_loss）在持续下降，而性能指标（auc和accuracy）在持续上升，这是一个非常健康的学习曲线。 |
| results | temp_result/test_relativity/result.txt.txt | 11 | AUC | 训练在第72个epoch时提前停止（early stopping），这也是一个好现象。这意味着模型在第57个epoch时在验证集上达到了最佳性能（val_auc: 0.95361），并在那之后自动保存了最佳模型。这有效地防止了模型在后续训练中发生过拟合，确保了得到的是泛化能力最强的模型。 |
| results | temp_result/改进memo.md | 1 |  | # Baseline |
| results | temp_result/改进memo.md | 3 | FFT; CWT; Grad-CAM | ## 超声方位FFT+8通道CWT+UNet+Grad-CAM: |
| results | temp_result/改进memo.md | 5 | FFT | * 目前是采用混合损失函数，但是并未增加FFT系数高频的惩罚力度 |
| results | temp_result/改进memo.md | 9 | Grad-CAM | * 目前**grad-cam**热力图效果很差，热力点分布分散，很难看出集中特征，且很多热力点分布并不合理 |
| results | temp_result/改进memo.md | 11 | FFT | * **Predction**结果很差，只能预测"整体平均"的结构，基本上只对FFT第0个系数学习到了。且分布是**垂直条带** |
| results | temp_result/改进memo.md | 15 | log label | # 1. 对数变换标签 |
| results | temp_result/改进memo.md | 17 | FFT | * 同样未增加FFT系数高频的惩罚力度 |
| results | temp_result/改进memo.md | 19 | FFT; log label | * 对FFT系数进行对数变换，在生成标签时，不要直接使用FFT的幅度 `magnitude`，而是使用 `log(1 + magnitude)`。 |
| results | temp_result/改进memo.md | 23 | Grad-CAM | * 目前**grad-cam**热力图效果变好，热力点分布集中，目前注意力主要集中在**0.5-0.7ms，25-30kHz处**。 |
| results | temp_result/改进memo.md | 25 | FFT | * **Predction**结果目前不太好，都是**水平条带**分布。下一步预计**增加FFT系数高频的惩罚力度** |
| results | temp_result/改进memo.md | 29 | FFT | # 2. FFT系数高频损失惩罚 |
| results | temp_result/改进memo.md | 31 | FFT | * 增加了FFT系数高频的惩罚 |
| results | temp_result/改进memo.md | 33 | SSIM/PSNR | * 改进了run_analysis.py代码，增加了新指标，如SSIM,PSNR |
| results | temp_result/改进memo.md | 37 | Grad-CAM | * 目前**grad-cam**热力图没啥大变化，但是更加集中了，目前注意力主要集中在**0.75ms-0.85ms和25-28kHz处** |
| results | temp_result/改进memo.md | 39 | FFT; GAN/GaN | * **Predction**结果依旧很差，即使增加了FFT系数高频的惩罚，模型依旧只学习拟合FFT低系数，下一步引入GaN。 |
| results | temp_result/改进memo.md | 43 | severity; GAN/GaN | # 3. GaN+“窜槽严重性”变换 |
| results | temp_result/改进memo.md | 45 | severity | - **创建“窜槽严重性图”**：我们将原始的 `zc_slice` 变换为一个新的矩阵。对于矩阵中的每一个像素点，其新值为 `max(0, 2.5 - zc_slice)`。 |
| results | temp_result/改进memo.md | 47 | severity | - **如果 Zc = 1.0 (严重窜槽)**，新值为 `2.5 - 1.0 = 1.5` (一个较大的正数)。 |
| results | temp_result/改进memo.md | 49 | severity | - **如果 Zc = 2.4 (临界窜槽)**，新值为 `2.5 - 2.4 = 0.1` (一个小的正数)。 |
| results | temp_result/改进memo.md | 51 | severity | - **如果 Zc = 6.0 (胶结良好)**，新值为 `2.5 - 6.0 = -3.5`，经过 `max(0, ...)` 处理后，最终值为 **0**。 |
| results | temp_result/改进memo.md | 53 | FFT; log label; severity | - **对“严重性图”进行FFT**：接下来，我们对这个全新的、只包含窜槽信息的“严重性图”进行FFT、取幅度、对数变换等后续操作。 |
| results | temp_result/改进memo.md | 57 |  | * 模型崩塌 |
| results | temp_result/改进memo.md | 61 | dual-channel; GAN/GaN | # 4. GaN+"双通道二元标签"+焦点损失函数+过拟合测试 |
| results | temp_result/改进memo.md | 63 | dual-channel; severity | * **双通道二元标签 (Two-Channel Binary Label)**：我们不再生成单一的“严重性图”，而是为每个样本创建两个独立的、非0即1的“掩码”图： |
| results | temp_result/改进memo.md | 65 |  | * **窜槽掩码 (Channeling Mask)**：一个矩阵，其中`Zc < 2.5`的位置为1，其他位置为0。 |
| results | temp_result/改进memo.md | 67 |  | * **良好掩码 (Good Bonding Mask)**：一个矩阵，其中`Zc >= 2.5`的位置为1，其他位置为0。 |
| results | temp_result/改进memo.md | 68 | dual-channel | 我们将这两个掩码**堆叠**起来，形成一个双通道的标签。这样，我们就把原始的Zc值，转换成了一个清晰的、非黑即白的分类问题。 |
| results | temp_result/改进memo.md | 70 |  | * 我们将强制让模型只在一个**极小的、只有1个样本**的数据集上进行训练。 |
| results | temp_result/改进memo.md | 72 |  | - **如果模型能够成功地在这个单一样本上过拟合**（即Generator Loss显著下降，最终能够完美地生成这个样本对应的标签图像），那么这就**证明**： |
| results | temp_result/改进memo.md | 74 |  | 1. 您的模型架构（U-Net）有足够的能力来完成这个任务。 |
| results | temp_result/改进memo.md | 76 | GAN/GaN | 2. 您的训练代码（包括GAN的博弈、梯度更新等）是**正确无误**的。 |
| results | temp_result/改进memo.md | 78 |  | 3. 输入和标签之间**确实存在可以被学会的对应关系**！ |
| results | temp_result/改进memo.md | 79 |  | 在这种情况下，训练在完整数据集上的失败，就说明问题在于**泛化能力**，即模型难以从2846个样本中归纳出普适的规律，这可能需要更强大的模型或更多的训练技巧。 |
| results | temp_result/改进memo.md | 81 |  | - **如果模型连这一个样本都学不会**（Generator Loss依然无法下降），那么这就**强烈地暗示**： |
| results | temp_result/改进memo.md | 83 |  | 1. 问题可能出在模型架构本身（例如梯度无法有效传播）。 |
| results | temp_result/改进memo.md | 85 | CWT | 2. 或者，正如您所担心的，**输入和标签之间的关系确实极其微弱，甚至在CWT这个特征空间中是不存在的**。 |
| results | temp_result/改进memo.md | 89 |  | * 模型崩塌 |
| results | temp_result/改进memo.md | 93 | CWT | # 5. CNN+分类任务（验证CWT与标签之间是否存在对应关系） |
| results | temp_result/改进memo.md | 95 |  | * 如果样本对应的 Zc 切片中有超过 1% 的像素低于阈值 2.5，则该样本被标记为 1（存在通道效应）。该阈值是一个合理的起始值，之后可以根据需要进行调整。 |
| results | temp_result/改进memo.md | 99 |  | - **模型确实学会了！** |
| repo | config.py | 31 | log label | LOG_DIR = os.path.join(OUTPUT_DIR, 'logs') |
| repo | config.py | 47 | FFT | FFT_COEFFICIENTS = 30 |
| repo | config.py | 51 | CWT | # --- CWT和模型参数 (CWT & Model Parameters) --- |
| repo | config.py | 53 | CWT | # --- CWT变换参数 --- |
| repo | config.py | 55 | CWT | CWT_WAVELET = 'cmor1.5-1.0' |
| repo | config.py | 58 | CWT | CWT_CHUNK_SIZE = 2048 |
| repo | config.py | 60 | CWT | # *** 逻辑修正：根据目标频率范围 (1-30kHz) 计算CWT尺度 *** |
| repo | config.py | 69 | CWT | CENTRAL_FREQ = pywt.central_frequency(CWT_WAVELET) |
| repo | config.py | 73 | log label | # 使用对数间隔生成尺度数组，以在低频获得更好分辨率，符合原始方案建议 |
| repo | config.py | 74 | CWT | CWT_SCALES = np.geomspace(MIN_SCALE, MAX_SCALE, N_SCALES) |
| repo | config.py | 76 | CWT | # 预计算CWT尺度对应的频率轴 (单位: kHz) |
| repo | config.py | 77 | CWT | FREQUENCIES_HZ = pywt.scale2frequency(CWT_WAVELET, CWT_SCALES) / SAMPLING_PERIOD |
| repo | config.py | 78 | CWT | CWT_FREQUENCIES_KHZ = FREQUENCIES_HZ / 1000 |
| repo | config.py | 81 |  | # --- 模型参数 --- |
| repo | config.py | 85 |  | # --- 训练参数 (针对A100优化) --- |
| repo | main.py | 23 | CWT | cwt_h5_path = os.path.join(PROCESSED_DATA_DIR, f'array_{str(ARRAY_ID).zfill(2)}', 'cwt_images.h5') |
| repo | main.py | 31 | CWT | return os.path.exists(cwt_h5_path) |
| repo | main.py | 85 | CWT | 'transform': 'src/cwt_transformation/main_transform_translation.py', |
| repo | src/cwt_transformation/main_transform_translation.py | 15 | CWT | ARRAY_ID, PROCESSED_DATA_DIR, CWT_SCALES, CWT_WAVELET, |
| repo | src/cwt_transformation/main_transform_translation.py | 16 | CWT | INPUT_SHAPE, DEBUG_MODE, DEBUG_SONIC_DEPTH_POINTS, CWT_CHUNK_SIZE |
| repo | src/cwt_transformation/main_transform_translation.py | 19 | CWT | def batch_cwt_transformer(waveforms, scales, wavelet): |
| repo | src/cwt_transformation/main_transform_translation.py | 21 | CWT | 对一个批次的8通道波形并行执行CWT。 |
| repo | src/cwt_transformation/main_transform_translation.py | 23 | CWT | 输出 cwt_images: (batch_size, n_scales, time_steps, 8) |
| repo | src/cwt_transformation/main_transform_translation.py | 29 | CWT | cwt_images = np.zeros((batch_size, n_scales, time_steps, n_channels), dtype=np.float32) |
| repo | src/cwt_transformation/main_transform_translation.py | 35 | CWT | coeffs, _ = pywt.cwt(waveforms[i, j, :], scales, wavelet) |
| repo | src/cwt_transformation/main_transform_translation.py | 39 | CWT | cwt_images[i, :, :, :] = np.stack(all_coeffs, axis=-1) |
| repo | src/cwt_transformation/main_transform_translation.py | 41 | CWT | return cwt_images |
| repo | src/cwt_transformation/main_transform_translation.py | 43 | CWT | def transform_waveforms_to_cwt_images(): |
| repo | src/cwt_transformation/main_transform_translation.py | 45 | CWT | 主函数：读取处理后的波形，分块执行CWT，并将结果保存到HDF5文件。 |
| repo | src/cwt_transformation/main_transform_translation.py | 47 | CWT | print("--- Starting AVIP Phase 2a: Chunked CWT Transformation ---") |
| repo | src/cwt_transformation/main_transform_translation.py | 52 | CWT | output_h5_path = os.path.join(array_dir, 'cwt_images.h5') |
| repo | src/cwt_transformation/main_transform_translation.py | 67 | CWT | num_chunks = math.ceil(num_samples / CWT_CHUNK_SIZE) |
| repo | src/cwt_transformation/main_transform_translation.py | 68 | CWT | print(f"Total samples: {num_samples}, Chunk size: {CWT_CHUNK_SIZE}, Number of chunks: {num_chunks}") |
| repo | src/cwt_transformation/main_transform_translation.py | 72 | CWT | # 创建一个可调整大小的数据集来存储所有CWT图像 |
| repo | src/cwt_transformation/main_transform_translation.py | 74 | CWT | 'cwt_images', |
| repo | src/cwt_transformation/main_transform_translation.py | 83 | CWT | start_idx = i * CWT_CHUNK_SIZE |
| repo | src/cwt_transformation/main_transform_translation.py | 84 | CWT | end_idx = min((i + 1) * CWT_CHUNK_SIZE, num_samples) |
| repo | src/cwt_transformation/main_transform_translation.py | 89 | CWT | # 对当前块执行CWT变换 |
| repo | src/cwt_transformation/main_transform_translation.py | 90 | CWT | cwt_images_chunk = batch_cwt_transformer(waveforms_chunk, CWT_SCALES, CWT_WAVELET) |
| repo | src/cwt_transformation/main_transform_translation.py | 93 | CWT | dset[start_idx:end_idx] = cwt_images_chunk |
| repo | src/cwt_transformation/main_transform_translation.py | 95 | CWT | print("\n--- Chunked CWT Transformation Complete ---") |
| repo | src/cwt_transformation/main_transform_translation.py | 96 | CWT | print(f"All CWT images saved to: {output_h5_path}") |
| repo | src/cwt_transformation/main_transform_translation.py | 99 | CWT | transform_waveforms_to_cwt_images() |
| repo | src/data_processing/create_tfrecords.py | 17 | FFT | ARRAY_ID, PROCESSED_DATA_DIR, GROUND_TRUTH_DB_PATH, FFT_COEFFICIENTS, |
| repo | src/data_processing/create_tfrecords.py | 18 | CWT | MAX_PATH_DEPTH_POINTS, DEBUG_MODE, DEBUG_SONIC_DEPTH_POINTS, CWT_CHUNK_SIZE |
| repo | src/data_processing/create_tfrecords.py | 27 | FFT | def process_zc_slice_to_label(zc_slice, n_fft_coeffs, max_len): |
| repo | src/data_processing/create_tfrecords.py | 29 | FFT | 将Zc切片处理成固定大小的FFT幅度图像标签。 |
| repo | src/data_processing/create_tfrecords.py | 31 | FFT | # 1. 沿方位角轴（180个点）执行FFT |
| repo | src/data_processing/create_tfrecords.py | 32 | FFT | fft_coeffs = np.fft.fft(zc_slice, axis=1) |
| repo | src/data_processing/create_tfrecords.py | 35 | FFT | fft_magnitudes_raw = np.abs(fft_coeffs) |
| repo | src/data_processing/create_tfrecords.py | 40 | log label | # 使用对数变换 log(1 + x) 来压缩数值范围，提升高频（结构）信号的权重。 |
| repo | src/data_processing/create_tfrecords.py | 42 | FFT; log label | fft_magnitudes = np.log1p(fft_magnitudes_raw).astype(np.float32) |
| repo | src/data_processing/create_tfrecords.py | 48 | FFT | label_image = fft_magnitudes[:, :n_fft_coeffs] |
| repo | src/data_processing/create_tfrecords.py | 52 | FFT | padded_label = np.zeros((max_len, n_fft_coeffs), dtype=np.float32) |
| repo | src/data_processing/create_tfrecords.py | 64 | FFT; CWT | def create_tfrecord_example(cwt_image, fft_label): |
| repo | src/data_processing/create_tfrecords.py | 69 | CWT | 'feature': _bytes_feature(tf.io.serialize_tensor(cwt_image)), |
| repo | src/data_processing/create_tfrecords.py | 70 | FFT | 'label': _bytes_feature(tf.io.serialize_tensor(fft_label)), |
| repo | src/data_processing/create_tfrecords.py | 76 | CWT | 主函数：读取CWT图像和Zc真值，创建用于图像翻译的TFRecord文件。 |
| repo | src/data_processing/create_tfrecords.py | 78 | log label | print("--- Starting AVIP Phase 2b: TFRecord Generation (with Log Scaling) ---") |
| repo | src/data_processing/create_tfrecords.py | 82 | CWT | cwt_h5_path = os.path.join(array_dir, 'cwt_images.h5') |
| repo | src/data_processing/create_tfrecords.py | 90 | CWT | print(f"Loading CWT images from: {cwt_h5_path}") |
| repo | src/data_processing/create_tfrecords.py | 105 | CWT | with h5py.File(cwt_h5_path, 'r') as cwt_hf, \ |
| repo | src/data_processing/create_tfrecords.py | 109 | CWT | cwt_dset = cwt_hf['cwt_images'] |
| repo | src/data_processing/create_tfrecords.py | 117 | CWT | # --- 1. 读取预先计算好的CWT图像 --- |
| repo | src/data_processing/create_tfrecords.py | 118 | CWT | cwt_image = cwt_dset[i] |
| repo | src/data_processing/create_tfrecords.py | 120 |  | # --- 2. 读取Zc真值切片并生成标签 --- |
| repo | src/data_processing/create_tfrecords.py | 123 |  | zc_slice = path_data_group[sonic_depth_key][:] |
| repo | src/data_processing/create_tfrecords.py | 125 | FFT | fft_label = process_zc_slice_to_label( |
| repo | src/data_processing/create_tfrecords.py | 126 | FFT | zc_slice, FFT_COEFFICIENTS, MAX_PATH_DEPTH_POINTS |
| repo | src/data_processing/create_tfrecords.py | 130 | FFT; CWT | example = create_tfrecord_example(cwt_image, fft_label) |
| repo | src/data_processing/main_preprocess.py | 32 | log label | b, a = butter(order, normal_cutoff, btype='high', analog=False) |
| repo | src/data_processing/main_preprocess.py | 46 |  | zc_data = cast_data['Zc'] |
| repo | src/data_processing/main_preprocess.py | 64 |  | return df_cast, zc_data, df_sonic, waveforms_dict |
| repo | src/data_processing/main_preprocess.py | 80 |  | def interpolate_to_grid(df, zc_data, unified_depth): |
| repo | src/data_processing/main_preprocess.py | 82 |  | 将Zc数据插值到统一的深度网格上。 |
| repo | src/data_processing/main_preprocess.py | 84 |  | print("Interpolating Zc data to the unified grid...") |

Full extraction is in `text_evidence_extraction.json`; memo claims are in `memo_experiment_claims.csv`.
