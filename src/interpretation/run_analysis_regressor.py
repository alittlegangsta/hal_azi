# 文件路径: src/interpretation/run_analysis_regressor.py
# (已修正：加载并使用索引文件来过滤数据)

import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import h5py
from tqdm import tqdm
from collections import defaultdict
import random
import pandas as pd
import seaborn as sns

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.file_io import create_dir, load_pickle
from config import (
    ARRAY_ID, PROCESSED_DATA_DIR, MODEL_DIR, RESULTS_DIR,
    INPUT_SHAPE, TIME_STEPS, CWT_FREQUENCIES_KHZ, BATCH_SIZE,
    TASK_TYPE, FFT_COEFFICIENTS, MAX_PATH_DEPTH_POINTS
)

def make_gradcam_heatmap_for_regression(img_array, model, last_conv_layer_name):
    try:
        last_conv_layer = model.get_layer(last_conv_layer_name)
    except ValueError:
        print(f"警告: 层 '{last_conv_layer_name}' 未找到。正在尝试自动查找...")
        for layer in reversed(model.layers):
            if isinstance(layer, (tf.keras.layers.Conv2D)):
                last_conv_layer_name = layer.name
                last_conv_layer = layer
                print(f"自动找到最后一个卷积层: '{last_conv_layer_name}'")
                break
        if not last_conv_layer: raise ValueError("错误：无法在模型中自动找到任何卷积层。")
    grad_model = tf.keras.models.Model(model.inputs, [model.output, last_conv_layer.output])
    with tf.GradientTape() as tape:
        final_preds, last_conv_layer_output = grad_model(img_array)
        class_channel = tf.reduce_sum(final_preds)
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy(), last_conv_layer_name

def _calculate_mean_severity(profile):
    if profile.ndim == 2:
        dc_log_mag_profile = profile[:, 0]
        dc_mag_profile = np.exp(dc_log_mag_profile) - 1
        valid_indices = np.where(dc_mag_profile > 0.01)[0]
        if len(valid_indices) == 0: return 0.0
        return np.mean(dc_mag_profile[valid_indices])
    else:
        valid_indices = np.where(profile > 0.1)[0]
        if len(valid_indices) == 0: return 0.0
        return np.mean(profile[valid_indices])

def classify_by_severity_profile(profile):
    mean_severity_value = _calculate_mean_severity(profile)
    if mean_severity_value == 0.0: return "Negligible"
    if profile.ndim == 2:
        if mean_severity_value < 1.8: return "Negligible"
        elif mean_severity_value < 9.0: return "Low Severity"
        elif mean_severity_value < 27.0: return "Medium Severity"
        else: return "High Severity"
    else:
        if mean_severity_value < 1.0: return "Negligible"
        elif mean_severity_value < 5.0: return "Low Severity"
        elif mean_severity_value < 15.0: return "Medium Severity"
        else: return "High Severity"

def plot_overall_performance_scatter(all_labels, all_predictions, output_path):
    print(f"\n--- 正在生成 {TASK_TYPE} 任务的总体性能散点图 ---")
    true_values = []
    pred_values = []
    categories = []
    for i in tqdm(range(len(all_labels)), desc="Calculating severity for scatter plot"):
        true_label = all_labels[i]
        pred_label = all_predictions[i]
        true_severity = _calculate_mean_severity(true_label)
        pred_severity = _calculate_mean_severity(pred_label)
        if true_severity > 0.01:
            true_values.append(true_severity)
            pred_values.append(pred_severity)
            categories.append(classify_by_severity_profile(true_label))
    df = pd.DataFrame({
        'Ground Truth Severity': true_values, 'Predicted Severity': pred_values, 'Category': categories
    })
    plt.figure(figsize=(10, 10))
    sns.scatterplot(data=df, x='Ground Truth Severity', y='Predicted Severity', 
                    hue='Category', hue_order=['Negligible', 'Low Severity', 'Medium Severity', 'High Severity'],
                    palette='viridis', alpha=0.5, s=15)
    max_val = max(max(true_values, default=0), max(pred_values, default=0)) + 1
    plt.plot([0, max_val], [0, max_val], 'r--', label='Ideal Prediction')
    plt.title(f'Overall Prediction Performance ({TASK_TYPE})', fontsize=16)
    if TASK_TYPE == 'fft_regression':
        plt.xlabel('Ground Truth Mean Integrated Severity', fontsize=12)
        plt.ylabel('Predicted Mean Integrated Severity', fontsize=12)
    else:
        plt.xlabel('Ground Truth Mean Channeling Percentage (%)', fontsize=12)
        plt.ylabel('Predicted Mean Channeling Percentage (%)', fontsize=12)
    plt.legend(title='Severity Category'); plt.grid(True, linestyle='--', alpha=0.6)
    plt.axis('equal'); plt.xlim(0, max_val); plt.ylim(0, max_val)
    plt.savefig(output_path, dpi=150); plt.close()
    print(f"总体性能散点图已保存至: {output_path}")

def generate_waveform_and_gradcam_plots(model, all_cwts_original, all_cwts_masked, all_waveforms, all_labels, output_dir, num_samples_per_category=8):
    print(f"\n--- 正在为每个类别生成 {num_samples_per_category} 个波形与Grad-CAM对比图 ---")
    indices_by_category = defaultdict(list)
    for i, label in enumerate(all_labels):
        category = classify_by_severity_profile(label)
        indices_by_category[category].append(i)
    last_conv_layer_name = 'top_conv'
    is_first_plot = True
    time_axis_ms = np.arange(TIME_STEPS) * 0.01
    cwt_plot_extent = [time_axis_ms[0], time_axis_ms[-1], CWT_FREQUENCIES_KHZ[-1], CWT_FREQUENCIES_KHZ[0]]
    categories_order = ["Negligible", "Low Severity", "Medium Severity", "High Severity"]
    for category in categories_order:
        indices = indices_by_category[category]
        if not indices:
            print(f"\n类别 '{category}' 中没有样本，跳过。")
            continue
        print(f"\n处理类别: {category} (共 {len(indices)} 个样本)")
        num_to_pick = min(len(indices), num_samples_per_category)
        indices_to_plot = random.sample(indices, num_to_pick)
        for i in tqdm(indices_to_plot, desc=f"生成 {category} 组合图"):
            waveform = all_waveforms[i]
            cwt_image_original = all_cwts_original[i]
            cwt_image_masked = all_cwts_masked[i]
            img_array_expanded = np.expand_dims(cwt_image_masked, axis=0)
            heatmap, used_layer_name = make_gradcam_heatmap_for_regression(img_array_expanded, model, last_conv_layer_name)
            if is_first_plot:
                last_conv_layer_name = used_layer_name
                print(f"Grad-CAM将作用于最终确定的层: '{last_conv_layer_name}'")
                is_first_plot = False
            heatmap_resized = tf.image.resize(np.expand_dims(heatmap, axis=-1), [INPUT_SHAPE[0], INPUT_SHAPE[1]]).numpy()
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 12), gridspec_kw={'height_ratios': [1, 2]})
            fig.suptitle(f'Analysis for Sample {i} (Category: {category}) - Task: {TASK_TYPE}', fontsize=18)
            ax1.set_title('High-pass Filtered Waveforms (8 Channels)', fontsize=14)
            for channel in range(waveform.shape[0]):
                ax1.plot(time_axis_ms, waveform[channel, :], alpha=0.7)
            ax1.set_xlabel('Time (ms)'); ax1.set_ylabel('Amplitude')
            ax1.grid(True, linestyle='--', alpha=0.6); ax1.set_xlim(time_axis_ms[0], time_axis_ms[-1])
            ax2.set_title('Grad-CAM on CWT Spectrogram (Heatmap from Masked Input)', fontsize=14)
            ax2.imshow(cwt_image_original[:, :, 0], aspect='auto', cmap='gray', extent=cwt_plot_extent)
            ax2.imshow(heatmap_resized, cmap='jet', alpha=0.5, extent=cwt_plot_extent, aspect='auto')
            ax2.set_xlabel('Time (ms)'); ax2.set_ylabel('Frequency (kHz)')
            plot_dir = os.path.join(output_dir, f'combo_plots_{category.replace(" ", "_")}')
            create_dir(plot_dir)
            plot_path = os.path.join(plot_dir, f'combo_sample_{i}.png')
            plt.tight_layout(rect=[0, 0, 1, 0.96]); plt.savefig(plot_path, dpi=120); plt.close(fig)

def plot_depth_log_comparison(all_labels, all_predictions, sonic_depths, output_path):
    print(f"\n--- 正在生成 {TASK_TYPE} 任务的测井曲线对比图 ---")
    true_values = []
    pred_values = []
    for i in range(len(all_labels)):
        true_values.append(_calculate_mean_severity(all_labels[i]))
        pred_values.append(_calculate_mean_severity(all_predictions[i]))

    plt.figure(figsize=(10, 15))
    plt.plot(true_values, sonic_depths, label='Ground Truth Severity', color='blue', linewidth=1)
    plt.plot(pred_values, sonic_depths, label='Predicted Severity', color='red', alpha=0.7, linewidth=1)
    plt.title(f'Depth Log Comparison ({TASK_TYPE})', fontsize=16)
    plt.xlabel('Mean Severity', fontsize=12); plt.ylabel('Absolute Sonic Depth (ft)', fontsize=12)
    plt.gca().invert_yaxis()
    plt.legend(); plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(output_path, dpi=200); plt.close()
    print(f"测井曲线对比图已保存至: {output_path}")

def create_profile_FFT_spectrum_image(all_labels, all_predictions, sonic_depths, output_path):
    print(f"\n--- 正在生成 FFT 频谱成像对比图 ---")
    if TASK_TYPE != 'fft_regression':
        print(f"  - 跳过：此图表仅适用于 'fft_regression' 任务。")
        return
    truth_stack = np.stack(all_labels, axis=0)
    predictions_stack = all_predictions
    truth_image = np.mean(truth_stack, axis=1)
    predictions_image = np.mean(predictions_stack, axis=1)
    error_image = predictions_image - truth_image
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 15), sharey=True)
    fig.suptitle('Virtual FFT Spectrum Log Comparison (Mean across 70 path points)', fontsize=18)
    plot_extent = [0, FFT_COEFFICIENTS - 1, sonic_depths[-1], sonic_depths[0]]
    vmax = np.percentile(truth_image, 98)
    im1 = ax1.imshow(truth_image, aspect='auto', cmap='jet', extent=plot_extent, vmin=0, vmax=vmax)
    ax1.set_title('Ground Truth (Mean FFT Spectrum)', fontsize=14)
    ax1.set_xlabel('FFT Coefficient Index (0-29)'); ax1.set_ylabel('Absolute Sonic Depth (ft)')
    fig.colorbar(im1, ax=ax1, label='Log(1 + Integrated Severity)')
    im2 = ax2.imshow(predictions_image, aspect='auto', cmap='jet', extent=plot_extent, vmin=0, vmax=vmax)
    ax2.set_title('Predicted (Mean FFT Spectrum)', fontsize=14)
    ax2.set_xlabel('FFT Coefficient Index (0-29)')
    fig.colorbar(im2, ax=ax2, label='Log(1 + Integrated Severity)')
    error_limit = vmax / 3.0
    im3 = ax3.imshow(error_image, aspect='auto', cmap='coolwarm', extent=plot_extent, vmin=-error_limit, vmax=error_limit)
    ax3.set_title('Prediction Error (Pred - True)', fontsize=14)
    ax3.set_xlabel('FFT Coefficient Index (0-29)')
    fig.colorbar(im3, ax=ax3, label='Error')
    plt.tight_layout(rect=[0, 0, 1, 0.95]); plt.savefig(output_path, dpi=200); plt.close()
    print(f"FFT频谱成像对比图已保存至: {output_path}")

def run_analysis_regressor():
    array_dir = os.path.join(PROCESSED_DATA_DIR, f'array_{str(ARRAY_ID).zfill(2)}')

    if TASK_TYPE == 'fft_regression':
        print("--- 开始为【FFT】回归模型进行全面的分析 ---")
        model_path = os.path.join(MODEL_DIR, 'best_fft_regressor_model.h5')
        tfrecord_path = os.path.join(array_dir, 'tfrecords', 'fft_regression_data.tfrecord')
        output_plot_dir = os.path.join(RESULTS_DIR, 'fft_model_analysis_plots')
        label_shape = [MAX_PATH_DEPTH_POINTS, FFT_COEFFICIENTS]
    else:
        print("--- 开始为【原始】回归模型进行全面的分析 ---")
        model_path = os.path.join(MODEL_DIR, 'best_advanced_regressor_model.h5')
        tfrecord_path = os.path.join(array_dir, 'tfrecords', 'profile_regression_data.tfrecord')
        output_plot_dir = os.path.join(RESULTS_DIR, 'original_model_analysis_plots')
        label_shape = [MAX_PATH_DEPTH_POINTS]

    if not os.path.exists(model_path):
        print(f"错误：找不到模型文件 '{model_path}'。请先运行训练。")
        return
    
    model = tf.keras.models.load_model(model_path)
    print("模型加载成功。")
    create_dir(output_plot_dir)
    print(f"所有分析图将被保存至: {output_plot_dir}")

    print("正在加载所有样本数据...")
    
    def _parse_label_only(example_proto):
        feature_description = {'label': tf.io.FixedLenFeature([], tf.string)}
        parsed = tf.io.parse_single_example(example_proto, feature_description)
        label_tensor = tf.io.parse_tensor(parsed['label'], out_type=tf.float32)
        return tf.reshape(label_tensor, label_shape)
    
    raw_dataset = tf.data.TFRecordDataset(tfrecord_path)
    all_labels = [t.numpy() for t in raw_dataset.map(_parse_label_only)]
    
    cwt_h5_path = os.path.join(array_dir, 'cwt_images.h5')
    with h5py.File(cwt_h5_path, 'r') as cwt_file:
        all_cwts_original = [cwt_file['cwt_images'][i] for i in range(len(all_labels))]

    waveforms_pkl_path = os.path.join(array_dir, 'processed_waveforms.pkl')
    waveforms_data = load_pickle(waveforms_pkl_path)
    all_waveforms_unfiltered = waveforms_data['waveforms'] # (2846,)
    all_sonic_depths_unfiltered = waveforms_data['sonic_depths'] # (2846,)

    print(f"数据加载完成 (TFRecord 中有 {len(all_labels)} 个样本)。")
    
    info_path = f"{tfrecord_path}.idx.pkl"
    if not os.path.exists(info_path):
        print(f"错误：找不到索引文件 '{info_path}'。")
        print("请先删除旧的 'fft_regression_data.tfrecord' 文件，然后重新运行 main.py。")
        return
        
    print(f"正在加载处理过的索引: {info_path}")
    processed_indices = load_pickle(info_path)['processed_indices'] # (2842,)
    
    # 使用索引过滤不匹配的数据
    all_waveforms = all_waveforms_unfiltered[processed_indices]
    all_sonic_depths = all_sonic_depths_unfiltered[processed_indices]
    
    # 验证
    if len(all_labels) != len(all_sonic_depths):
        print(f"!!! 严重错误：过滤后长度仍不匹配！ {len(all_labels)} != {len(all_sonic_depths)}")
        return
    else:
        print(f"数据过滤成功。样本数: {len(all_sonic_depths)}")

    print(f"应用 CWT 伪影掩码到 {len(all_cwts_original)} 个样本...")
    TIME_STEPS_TO_MASK = 30
    artifact_mask = np.ones_like(all_cwts_original[0], dtype=np.float32)
    artifact_mask[:, :TIME_STEPS_TO_MASK, :] = 0.0
    all_cwts_masked = [cwt * artifact_mask for cwt in all_cwts_original]

    print("正在对所有(已掩码)样本运行批量预测...")
    all_predictions = model.predict(np.array(all_cwts_masked), batch_size=BATCH_SIZE, verbose=1)
    print("预测完成。")

    # 1. 组合图
    generate_waveform_and_gradcam_plots(model, all_cwts_original, all_cwts_masked, all_waveforms, all_labels, output_plot_dir)
    
    # 2. 散点图
    scatter_output_path = os.path.join(output_plot_dir, '_overall_performance_scatter.png')
    plot_overall_performance_scatter(all_labels, all_predictions, scatter_output_path)
    
    # 3. 测井曲线图 (现在使用过滤后的 all_sonic_depths)
    log_plot_path = os.path.join(output_plot_dir, '_depth_log_comparison.png')
    plot_depth_log_comparison(all_labels, all_predictions, all_sonic_depths, log_plot_path)

    # 4. FFT频谱成像图 (现在使用过滤后的 all_sonic_depths)
    spectrum_image_path = os.path.join(output_plot_dir, '_fft_spectrum_image_comparison.png')
    create_profile_FFT_spectrum_image(all_labels, all_predictions, all_sonic_depths, spectrum_image_path)
    
    print("\n--- 全面分析完成！ ---")

if __name__ == '__main__':
    run_analysis_regressor()