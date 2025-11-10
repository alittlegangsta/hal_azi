# 文件路径: src/interpretation/run_analysis_regressor.py
# (适配原始模型版：增加波形与Grad-CAM对比图 + 虚拟成像图功能)

import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import h5py
from tqdm import tqdm
from collections import defaultdict
import random

# 添加项目根目录
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.file_io import create_dir, load_pickle
from config import (
    ARRAY_ID, PROCESSED_DATA_DIR, MODEL_DIR, RESULTS_DIR,
    INPUT_SHAPE, TIME_STEPS, CWT_FREQUENCIES_KHZ, BATCH_SIZE
)

# --- 模型与数据处理函数 (保持不变) ---
def make_gradcam_heatmap_for_regression(img_array, model, last_conv_layer_name):
    """为回归模型生成Grad-CAM热力图。"""
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

def classify_by_severity_profile(profile):
    """根据整个剖面图的平均值对样本进行分类。"""
    valid_indices = np.where(profile > 0.1)[0]
    if len(valid_indices) == 0: return "Negligible"
    mean_percentage = np.mean(profile[valid_indices])
    if mean_percentage < 1.0: return "Negligible"
    elif mean_percentage < 5.0: return "Low Severity"
    elif mean_percentage < 15.0: return "Medium Severity"
    else: return "High Severity"

# --- 核心功能1：波形与Grad-CAM组合绘图函数 ---
def generate_waveform_and_gradcam_plots(model, all_cwts, all_waveforms, all_labels, output_dir, num_samples_per_category=8):
    """
    为每个严重程度类别挑选N个样本，并生成包含原始波形和Grad-CAM的组合图。
    """
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
            cwt_image = all_cwts[i]
            img_array_expanded = np.expand_dims(cwt_image, axis=0)
            
            heatmap, used_layer_name = make_gradcam_heatmap_for_regression(img_array_expanded, model, last_conv_layer_name)
            if is_first_plot:
                last_conv_layer_name = used_layer_name
                print(f"Grad-CAM将作用于最终确定的层: '{last_conv_layer_name}'")
                is_first_plot = False
            heatmap_resized = tf.image.resize(np.expand_dims(heatmap, axis=-1), [INPUT_SHAPE[0], INPUT_SHAPE[1]]).numpy()
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 12), gridspec_kw={'height_ratios': [1, 2]})
            fig.suptitle(f'Analysis for Sample {i} (Category: {category}) - Original Model', fontsize=18)
            
            ax1.set_title('High-pass Filtered Waveforms (8 Channels)', fontsize=14)
            for channel in range(waveform.shape[0]):
                ax1.plot(time_axis_ms, waveform[channel, :], alpha=0.7)
            ax1.set_xlabel('Time (ms)'); ax1.set_ylabel('Amplitude')
            ax1.grid(True, linestyle='--', alpha=0.6); ax1.set_xlim(time_axis_ms[0], time_axis_ms[-1])
            
            ax2.set_title('Grad-CAM on CWT Spectrogram', fontsize=14)
            ax2.imshow(cwt_image[:, :, 0], aspect='auto', cmap='gray', extent=cwt_plot_extent)
            ax2.imshow(heatmap_resized, cmap='jet', alpha=0.5, extent=cwt_plot_extent, aspect='auto')
            ax2.set_xlabel('Time (ms)'); ax2.set_ylabel('Frequency (kHz)')
            
            plot_dir = os.path.join(output_dir, f'combo_plots_{category.replace(" ", "_")}')
            create_dir(plot_dir)
            plot_path = os.path.join(plot_dir, f'combo_sample_{i}.png')
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.savefig(plot_path, dpi=120)
            plt.close(fig)

# --- 核心修改：新增“虚拟成像图”函数 ---
def create_profile_comparison_image(all_labels, all_predictions, sonic_depths, output_path):
    """
    将所有1D剖面图堆叠起来，生成一个覆盖整个深度范围的“虚拟成像图”。
    """
    print(f"\n--- 正在生成虚拟成像对比图 ---")
    
    # 1. 将标签列表堆叠成一个2D图像数组
    truth_image = np.stack(all_labels, axis=0)
    
    # 2. 预测结果已经是 (N_samples, 70) 数组
    predictions_image = all_predictions
    
    # 3. 计算误差图像
    error_image = predictions_image - truth_image
    
    # 4. 绘图
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 15), sharey=True)
    fig.suptitle('Virtual Profile Log Comparison (Original Model)', fontsize=18)
    
    # 定义Y轴（深度）和X轴（剖面点）的范围
    # Y轴反转，因为深度是向下增加的
    plot_extent = [0, truth_image.shape[1], sonic_depths[-1], sonic_depths[0]]

    # --- 子图1: 真实标签 ---
    im1 = ax1.imshow(truth_image, aspect='auto', cmap='jet', extent=plot_extent, vmin=0, vmax=50)
    ax1.set_title('Ground Truth Profile Log', fontsize=14)
    ax1.set_xlabel('Relative Profile Depth Points (0-70)')
    ax1.set_ylabel('Absolute Sonic Depth (ft)')
    fig.colorbar(im1, ax=ax1, label='Channeling Percentage (%)')

    # --- 子图2: 预测结果 ---
    im2 = ax2.imshow(predictions_image, aspect='auto', cmap='jet', extent=plot_extent, vmin=0, vmax=50)
    ax2.set_title('Predicted Profile Log', fontsize=14)
    ax2.set_xlabel('Relative Profile Depth Points (0-70)')
    fig.colorbar(im2, ax=ax2, label='Channeling Percentage (%)')

    # --- 子图3: 误差 ---
    # 使用'coolwarm'色图，中心为0 (白色)，红色为高估，蓝色为低估
    error_limit = 20 # 将误差颜色限制在±20%
    im3 = ax3.imshow(error_image, aspect='auto', cmap='coolwarm', extent=plot_extent, vmin=-error_limit, vmax=error_limit)
    ax3.set_title('Prediction Error (Pred - True)', fontsize=14)
    ax3.set_xlabel('Relative Profile Depth Points (0-70)')
    fig.colorbar(im3, ax=ax3, label='Error (%)')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"虚拟成像对比图已保存至: {output_path}")

def run_analysis_regressor():
    print("--- 开始为【原始】回归模型进行全面的分析 ---")

    model_path = os.path.join(MODEL_DIR, 'best_advanced_regressor_model.h5')
    if not os.path.exists(model_path):
        print(f"错误：找不到模型文件 '{model_path}'。")
        return
    
    model = tf.keras.models.load_model(model_path)
    print("原始模型加载成功。")

    print("正在加载所有样本数据...")
    array_dir = os.path.join(PROCESSED_DATA_DIR, f'array_{str(ARRAY_ID).zfill(2)}')
    
    tfrecord_path = os.path.join(array_dir, 'tfrecords', 'profile_regression_data.tfrecord')
    raw_dataset = tf.data.TFRecordDataset(tfrecord_path)
    all_labels = [t.numpy() for t in raw_dataset.map(lambda p: tf.io.parse_tensor(tf.io.parse_single_example(p, {'label': tf.io.FixedLenFeature([], tf.string)})['label'], out_type=tf.float32))]
    
    cwt_h5_path = os.path.join(array_dir, 'cwt_images.h5')
    with h5py.File(cwt_h5_path, 'r') as cwt_file:
        all_cwts = [cwt_file['cwt_images'][i] for i in range(len(all_labels))]

    # --- 核心修改：同时加载波形和深度轴 ---
    waveforms_pkl_path = os.path.join(array_dir, 'processed_waveforms.pkl')
    if not os.path.exists(waveforms_pkl_path):
        print(f"错误：找不到波形文件 '{waveforms_pkl_path}'。")
        return
    waveforms_data = load_pickle(waveforms_pkl_path)
    all_waveforms = waveforms_data['waveforms']
    all_sonic_depths = waveforms_data['sonic_depths']

    print(f"数据加载完成，共 {len(all_labels)} 个样本。")
    
    output_plot_dir = os.path.join(RESULTS_DIR, 'original_model_analysis_plots')
    create_dir(output_plot_dir)
    print(f"所有分析图将被保存至: {output_plot_dir}")
    
    print("正在对所有样本运行批量预测...")
    all_predictions = model.predict(np.array(all_cwts), batch_size=BATCH_SIZE, verbose=1)
    print("预测完成。")

    # --- 核心修改：调用两个绘图函数 ---
    
    # 1. 生成波形+GradCAM组合图
    generate_waveform_and_gradcam_plots(model, all_cwts, all_waveforms, all_labels, output_plot_dir)
    
    # 2. 生成新的虚拟成像图
    image_output_path = os.path.join(output_plot_dir, '_virtual_log_comparison.png')
    create_profile_comparison_image(all_labels, all_predictions, all_sonic_depths, image_output_path)

    print("\n--- 全面分析完成！ ---")


if __name__ == '__main__':
    run_analysis_regressor()