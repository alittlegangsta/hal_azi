# 文件路径: src/interpretation/run_analysis_regressor.py

import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import h5py
from tqdm import tqdm
from collections import defaultdict
# --- 导入一个用于美化图表的库 ---
import seaborn as sns
from sklearn.metrics import r2_score

# 添加项目根目录
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.file_io import create_dir
from config import (
    ARRAY_ID, PROCESSED_DATA_DIR, MODEL_DIR, RESULTS_DIR,
    INPUT_SHAPE, TIME_STEPS, CWT_FREQUENCIES_KHZ, BATCH_SIZE
)

def make_gradcam_heatmap_for_regression(img_array, model, last_conv_layer_name):
    """
    为回归模型生成Grad-CAM热力图 (最终正确版)。
    """
    last_conv_layer = model.get_layer(last_conv_layer_name)
    grad_model = tf.keras.models.Model(
        model.inputs, [model.output, last_conv_layer.output]
    )
    with tf.GradientTape() as tape:
        final_preds, last_conv_layer_output = grad_model(img_array)
        class_channel = tf.reduce_sum(final_preds)
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

def classify_by_severity(profile):
    """
    根据一维剖面图的平均值，对窜槽严重程度进行分类。
    """
    valid_indices = np.where(profile > 0.1)[0] # 使用一个小的阈值以避免噪声
    if len(valid_indices) == 0:
        return "Negligible" # 完全没有窜槽点
    
    mean_percentage = np.mean(profile[valid_indices])

    if mean_percentage < 1.0:
        return "Negligible"
    elif mean_percentage < 5.0:
        return "Low Severity"
    elif mean_percentage < 15.0:
        return "Medium Severity"
    else:
        return "High Severity"

def analyze_and_plot_errors_by_category(all_labels, all_predictions, output_path):
    """
    计算每个严重程度类别下的误差指标，并绘制误差分布的箱形图。
    """
    print("\n--- 正在按类别分析预测误差 ---")
    
    # 1. 准备数据
    errors_by_category = defaultdict(list)
    
    for i in range(len(all_labels)):
        true_profile = all_labels[i]
        pred_profile = all_predictions[i]
        
        # 对每个深度点的真实值进行分类
        for j in range(len(true_profile)):
            true_val = true_profile[j]
            pred_val = pred_profile[j]
            
            # 我们只分析有效深度点
            if true_val < 0.1 and np.mean(true_profile) < 1.0:
                category = "Negligible"
            elif 1.0 <= true_val < 5.0:
                category = "Low Severity"
            elif 5.0 <= true_val < 15.0:
                category = "Medium Severity"
            elif true_val >= 15.0:
                category = "High Severity"
            else:
                continue # 忽略真实值为0-1%之间的点，以减少噪声

            error = pred_val - true_val
            errors_by_category[category].append(error)

    # 2. 计算并打印指标
    print("\n--- 各类别下的误差指标 ---")
    print("-" * 50)
    print(f"{'Category':<25} | {'MAE':>7} | {'RMSE':>7} | {'R-squared':>10}")
    print("-" * 50)
    
    categories = ["Negligible", "Low Severity", "Medium Severity", "High Severity"]
    
    for category in categories:
        errors = np.array(errors_by_category.get(category, [0]))
        
        # 找到该类别对应的真实值和预测值
        true_vals = []
        pred_vals = []
        for i in range(len(all_labels)):
            for j in range(len(all_labels[i])):
                 # 这里我们简化一下，只为MAE和RMSE计算，R2需要更精确的匹配
                 pass 

        mae = np.mean(np.abs(errors))
        rmse = np.sqrt(np.mean(np.square(errors)))
        
        # R2的计算比较复杂，我们暂时只展示MAE和RMSE
        print(f"{category:<25} | {mae:>7.3f} | {rmse:>7.3f} | {'N/A':>10}")
    print("-" * 50)

    # 3. 绘制误差分布箱形图
    plt.figure(figsize=(12, 8))
    
    # 准备用于绘图的数据
    plot_data = []
    plot_labels = []
    for category in categories:
        errors = errors_by_category.get(category)
        if errors:
            plot_data.append(errors)
            plot_labels.append(f"{category}\n(N={len(errors)})")

    sns.boxplot(data=plot_data)
    plt.xticks(ticks=range(len(plot_labels)), labels=plot_labels)
    
    plt.axhline(0, color='r', linestyle='--', label='Ideal (Zero Error)')
    plt.title('Prediction Error Distribution by Severity Category', fontsize=16)
    plt.xlabel('Ground Truth Severity Category', fontsize=12)
    plt.ylabel('Prediction Error (Predicted - True) (%)', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"\n误差分布图已保存至: {output_path}")

def run_analysis_regressor(num_samples_for_gradcam=20):
    print("--- 开始为高级回归模型进行全面的分析 ---")

    model_path = os.path.join(MODEL_DIR, 'best_advanced_regressor_model.h5')
    model = tf.keras.models.load_model(model_path)
    print("模型加载成功。")
    
    last_conv_layer_name = 'top_conv' 
    print(f"Grad-CAM将作用于: '{last_conv_layer_name}'")

    print("正在加载所有样本标签...")
    tfrecord_dir = os.path.join(PROCESSED_DATA_DIR, f'array_{str(ARRAY_ID).zfill(2)}', 'tfrecords')
    tfrecord_path = os.path.join(tfrecord_dir, 'profile_regression_data.tfrecord')
    
    raw_dataset = tf.data.TFRecordDataset(tfrecord_path)
    all_labels = []
    
    def _parse_label_only(example_proto):
        feature_description = {'label': tf.io.FixedLenFeature([], tf.string)}
        parsed = tf.io.parse_single_example(example_proto, feature_description)
        return tf.io.parse_tensor(parsed['label'], out_type=tf.float32)

    for label_tensor in raw_dataset.map(_parse_label_only):
        all_labels.append(label_tensor.numpy())
    
    num_total_samples = len(all_labels)
    
    print("正在预加载所有CWT图像...")
    cwt_h5_path = os.path.join(os.path.dirname(tfrecord_dir), 'cwt_images.h5')
    all_cwts = []
    with h5py.File(cwt_h5_path, 'r') as cwt_file:
        for i in range(num_total_samples):
            all_cwts.append(cwt_file['cwt_images'][i])
    
    print("正在对所有样本运行批量预测...")
    all_predictions = model.predict(np.array(all_cwts), batch_size=BATCH_SIZE)
    print("预测完成。")

    output_plot_dir = os.path.join(RESULTS_DIR, 'final_analysis_plots')
    create_dir(output_plot_dir)
    print(f"所有分析图将被保存至: {output_plot_dir}")

    # --- 1. 生成个体Grad-CAM热力图 ---
    print(f"\n正在为 {num_samples_for_gradcam} 个窜槽最严重的样本生成个体Grad-CAM图...")
    avg_channeling = [np.mean(label[np.where(label > 0.1)]) if len(np.where(label > 0.1)[0]) > 0 else 0 for label in all_labels]
    indices_to_analyze = np.argsort(avg_channeling)[-num_samples_for_gradcam:]
    
    time_axis_ms = np.arange(TIME_STEPS) * 0.01
    plot_extent = [time_axis_ms[0], time_axis_ms[-1], CWT_FREQUENCIES_KHZ[-1], CWT_FREQUENCIES_KHZ[0]]

    for i in tqdm(indices_to_analyze, desc="生成Grad-CAM热力图"):
        img_array = all_cwts[i]
        img_array_expanded = np.expand_dims(img_array, axis=0)
        heatmap = make_gradcam_heatmap_for_regression(img_array_expanded, model, last_conv_layer_name)
        heatmap_resized = tf.image.resize(np.expand_dims(heatmap, axis=-1), [INPUT_SHAPE[0], INPUT_SHAPE[1]]).numpy()
        
        fig, ax = plt.subplots(figsize=(18, 8))
        ax.imshow(img_array[:, :, 0], aspect='auto', cmap='gray', extent=plot_extent)
        ax.imshow(heatmap_resized, cmap='jet', alpha=0.5, extent=plot_extent, aspect='auto')
        ax.set_title(f'Grad-CAM for Sample {i}', fontsize=16)
        ax.set_xlabel('Time (ms)'); ax.set_ylabel('Frequency (kHz)')
        plot_path = os.path.join(output_plot_dir, f'gradcam_sample_{i}.png')
        plt.savefig(plot_path, dpi=150)
        plt.close(fig)

    # --- 2. 生成误差分析图与指标 ---
    error_plot_path = os.path.join(output_plot_dir, '_error_distribution_by_category.png')
    analyze_and_plot_errors_by_category(all_labels, all_predictions, error_plot_path)

    print("\n--- 全面分析完成！ ---")


if __name__ == '__main__':
    run_analysis_regressor()