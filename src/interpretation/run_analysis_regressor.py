# 文件路径: src/interpretation/run_analysis_regressor.py
# (最终完整版：包含散点图、箱形图和代表性Grad-CAM)

import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import h5py
from tqdm import tqdm
from collections import defaultdict
import seaborn as sns

# 添加项目根目录
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.file_io import create_dir
from config import (
    ARRAY_ID, PROCESSED_DATA_DIR, MODEL_DIR, RESULTS_DIR,
    INPUT_SHAPE, TIME_STEPS, CWT_FREQUENCIES_KHZ, BATCH_SIZE
)

def make_gradcam_heatmap_for_regression(img_array, model, last_conv_layer_name):
    # ... (此函数无需改变，代码已折叠以保持简洁)
    try:
        last_conv_layer = model.get_layer(last_conv_layer_name)
    except ValueError:
        for layer in reversed(model.layers):
            if isinstance(layer, (tf.keras.layers.Conv2D)):
                last_conv_layer_name = layer.name
                last_conv_layer = layer
                break
        if not last_conv_layer: raise ValueError("找不到卷积层")
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


def classify_by_severity_pointwise(true_value, mean_profile_value):
    """根据单个点的真实值对严重性进行分类"""
    if true_value < 0.1 and mean_profile_value < 1.0: return "Negligible"
    elif 1.0 <= true_value < 5.0: return "Low Severity"
    elif 5.0 <= true_value < 15.0: return "Medium Severity"
    elif true_value >= 15.0: return "High Severity"
    return None

def classify_by_severity_profile(profile):
    """根据整个剖面图的平均值对样本进行分类"""
    valid_indices = np.where(profile > 0.1)[0]
    if len(valid_indices) == 0: return "Negligible"
    mean_percentage = np.mean(profile[valid_indices])
    if mean_percentage < 1.0: return "Negligible"
    elif mean_percentage < 5.0: return "Low Severity"
    elif mean_percentage < 15.0: return "Medium Severity"
    else: return "High Severity"

# --- 新增功能：绘制总体性能散点图 ---
def plot_overall_performance(all_labels, all_predictions, output_path):
    """
    绘制“真实值 vs 预测值”的总体性能散点图，并按严重性着色。
    """
    print("\n--- 正在生成总体性能散点图 ---")
    
    # 1. 将数据扁平化并分类
    flat_true = []
    flat_pred = []
    categories = []
    
    for i in range(len(all_labels)):
        mean_profile_val = np.mean(all_labels[i][np.where(all_labels[i] > 0.1)]) if len(np.where(all_labels[i] > 0.1)[0]) > 0 else 0
        for j in range(len(all_labels[i])):
            true_val = all_labels[i][j]
            # 我们只绘制有意义的点（真实值 > 0.1%）
            if true_val > 0.1:
                flat_true.append(true_val)
                flat_pred.append(all_predictions[i][j])
                categories.append(classify_by_severity_pointwise(true_val, mean_profile_val))

    df = pd.DataFrame({'True': flat_true, 'Predicted': flat_pred, 'Category': categories})

    # 2. 绘图
    plt.figure(figsize=(10, 10))
    sns.scatterplot(data=df, x='True', y='Predicted', hue='Category', 
                    hue_order=['Negligible', 'Low Severity', 'Medium Severity', 'High Severity'],
                    palette='viridis', alpha=0.5, s=15) # s是点的大小

    # 绘制理想预测线
    max_val = max(max(flat_true), max(flat_pred))
    plt.plot([0, max_val], [0, max_val], 'r--', label='Ideal Prediction')
    
    plt.title('Overall Prediction Performance (Weighted Model)', fontsize=16)
    plt.xlabel('Ground Truth Channeling Percentage (%)', fontsize=12)
    plt.ylabel('Predicted Channeling Percentage (%)', fontsize=12)
    plt.legend(title='Severity Category')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.axis('equal') # 保证X和Y轴尺度相同
    plt.xlim(0, max_val)
    plt.ylim(0, max_val)
    
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"总体性能散点图已保存至: {output_path}")


def analyze_and_plot_errors_by_category(all_labels, all_predictions, output_path):
    # ... (此函数无需改变，代码已折叠以保持简洁)
    print("\n--- 正在按类别分析预测误差 ---")
    errors_by_category = defaultdict(list)
    for i in range(len(all_labels)):
        mean_profile_val = np.mean(all_labels[i][np.where(all_labels[i] > 0.1)]) if len(np.where(all_labels[i] > 0.1)[0]) > 0 else 0
        for j in range(len(all_labels[i])):
            true_val = all_labels[i][j]
            category = classify_by_severity_pointwise(true_val, mean_profile_val)
            if category:
                errors_by_category[category].append(all_predictions[i][j] - true_val)
    plt.figure(figsize=(12, 8))
    categories_order = ["Negligible", "Low Severity", "Medium Severity", "High Severity"]
    plot_data = [errors_by_category.get(cat, []) for cat in categories_order]
    plot_labels = [f"{cat}\n(N={len(errors_by_category.get(cat, []))})" for cat in categories_order]
    sns.boxplot(data=plot_data); plt.xticks(ticks=range(len(plot_labels)), labels=plot_labels)
    plt.axhline(0, color='r', linestyle='--', label='Ideal (Zero Error)')
    plt.title('Prediction Error Distribution (Weighted Model)', fontsize=16)
    plt.xlabel('Ground Truth Severity Category'); plt.ylabel('Prediction Error (%)')
    plt.legend(); plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(output_path, dpi=150); plt.close()
    print(f"\n误差分布图已保存至: {output_path}")


def generate_representative_gradcam_plots(model, all_cwts, all_labels, output_dir):
    # ... (此函数无需改变，代码已折叠以保持简洁)
    print("\n--- 正在为每个类别生成代表性Grad-CAM图 ---")
    indices_by_category = defaultdict(list)
    avg_channeling_by_index = {}
    for i, label in enumerate(all_labels):
        category = classify_by_severity_profile(label)
        valid_indices = np.where(label > 0.1)[0]
        avg_channeling = np.mean(label[valid_indices]) if len(valid_indices) > 0 else 0
        indices_by_category[category].append(i)
        avg_channeling_by_index[i] = avg_channeling
    last_conv_layer_name = 'top_conv'
    is_first_plot = True
    time_axis_ms = np.arange(TIME_STEPS) * 0.01
    plot_extent = [time_axis_ms[0], time_axis_ms[-1], CWT_FREQUENCIES_KHZ[-1], CWT_FREQUENCIES_KHZ[0]]
    for category, indices in indices_by_category.items():
        if not indices: continue
        sorted_indices = sorted(indices, key=lambda i: avg_channeling_by_index[i])
        indices_to_plot = []
        if len(sorted_indices) > 0: indices_to_plot.append(sorted_indices[0])
        if len(sorted_indices) > 2:
            indices_to_plot.append(sorted_indices[len(sorted_indices) // 2])
            indices_to_plot.append(sorted_indices[-1])
        indices_to_plot = sorted(list(set(indices_to_plot)))
        for i in tqdm(indices_to_plot, desc=f"生成 {category} Grad-CAM"):
            img_array_expanded = np.expand_dims(all_cwts[i], axis=0)
            heatmap, used_layer_name = make_gradcam_heatmap_for_regression(img_array_expanded, model, last_conv_layer_name)
            if is_first_plot: last_conv_layer_name = used_layer_name; is_first_plot = False
            heatmap_resized = tf.image.resize(np.expand_dims(heatmap, axis=-1), [INPUT_SHAPE[0], INPUT_SHAPE[1]]).numpy()
            fig, ax = plt.subplots(figsize=(18, 8))
            ax.imshow(all_cwts[i][:, :, 0], aspect='auto', cmap='gray', extent=plot_extent)
            ax.imshow(heatmap_resized, cmap='jet', alpha=0.5, extent=plot_extent, aspect='auto')
            ax.set_title(f'Grad-CAM Sample {i} (Category: {category})', fontsize=16)
            ax.set_xlabel('Time (ms)'); ax.set_ylabel('Frequency (kHz)')
            plot_filename = f'gradcam_{category.replace(" ", "_")}_sample_{i}.png'
            plt.savefig(os.path.join(output_dir, plot_filename), dpi=150); plt.close(fig)


def run_analysis_regressor():
    print("--- 开始为高级【加权】回归模型进行全面的分析 ---")

    # 为了让散点图更好看，需要额外导入pandas
    try:
        global pd
        import pandas as pd
    except ImportError:
        print("错误：请安装pandas库 (`pip install pandas`) 以绘制散点图。")
        return

    model_path = os.path.join(MODEL_DIR, 'best_weighted_regressor_model.h5')
    if not os.path.exists(model_path):
        print(f"错误：找不到模型文件 '{model_path}'。")
        return
        
    model = tf.keras.models.load_model(model_path, custom_objects={'asymmetric_huber_loss': tf.keras.losses.Huber()})
    print("加权模型加载成功。")

    print("正在加载所有样本数据...")
    tfrecord_dir = os.path.join(PROCESSED_DATA_DIR, f'array_{str(ARRAY_ID).zfill(2)}', 'tfrecords')
    tfrecord_path = os.path.join(tfrecord_dir, 'profile_regression_data.tfrecord')
    raw_dataset = tf.data.TFRecordDataset(tfrecord_path)
    all_labels = [t.numpy() for t in raw_dataset.map(lambda p: tf.io.parse_tensor(tf.io.parse_single_example(p, {'label': tf.io.FixedLenFeature([], tf.string)})['label'], out_type=tf.float32))]
    cwt_h5_path = os.path.join(os.path.dirname(tfrecord_dir), 'cwt_images.h5')
    with h5py.File(cwt_h5_path, 'r') as cwt_file:
        all_cwts = [cwt_file['cwt_images'][i] for i in range(len(all_labels))]
    
    print("正在对所有样本运行批量预测...")
    all_predictions = model.predict(np.array(all_cwts), batch_size=BATCH_SIZE, verbose=1)
    
    output_plot_dir = os.path.join(RESULTS_DIR, 'weighted_model_analysis_plots')
    create_dir(output_plot_dir)
    print(f"所有分析图将被保存至: {output_plot_dir}")

    # --- 核心修改：调用所有三个分析函数 ---
    scatter_plot_path = os.path.join(output_plot_dir, '_overall_performance_scatter_weighted.png')
    plot_overall_performance(all_labels, all_predictions, scatter_plot_path)
    
    error_plot_path = os.path.join(output_plot_dir, '_error_distribution_by_category_weighted.png')
    analyze_and_plot_errors_by_category(all_labels, all_predictions, error_plot_path)

    generate_representative_gradcam_plots(model, all_cwts, all_labels, output_plot_dir)

    print("\n--- 全面分析完成！ ---")


if __name__ == '__main__':
    run_analysis_regressor()