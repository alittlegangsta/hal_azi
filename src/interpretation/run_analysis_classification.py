# 文件路径: src/interpretation/run_analysis_classification.py

import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import h5py
from tqdm import tqdm

# 添加项目根目录
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.file_io import create_dir, load_pickle
from config import (
    ARRAY_ID, PROCESSED_DATA_DIR, MODEL_DIR, RESULTS_DIR,
    INPUT_SHAPE, TIME_STEPS, CWT_FREQUENCIES_KHZ
)

# ==============================================================================
# >>>>>>>>>> 最终分析工具：分类模型的Grad-CAM <<<<<<<<<<<
# ==============================================================================

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    为给定的图像和分类模型生成Grad-CAM热力图。
    """
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

def run_classification_analysis(num_samples_to_find=5):
    """
    主函数：加载分类模型，找到窜槽样本，并生成Grad-CAM热力图。
    """
    print("--- 开始为分类模型进行Grad-CAM分析 ---")

    # --- 1. 加载模型 ---
    model_path = os.path.join(MODEL_DIR, 'best_classifier_model.h5')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件未找到: {model_path}。请先运行训练。")
    model = tf.keras.models.load_model(model_path)
    print("模型加载成功。")

    # --- 2. 寻找有窜槽的样本进行分析 ---
    print(f"正在寻找 {num_samples_to_find} 个有窜槽的样本...")
    array_dir = os.path.join(PROCESSED_DATA_DIR, f'array_{str(ARRAY_ID).zfill(2)}')
    cwt_h5_path = os.path.join(array_dir, 'cwt_images.h5')
    tfrecord_path = os.path.join(array_dir, 'tfrecords', 'classification_data.tfrecord')

    cwt_file = h5py.File(cwt_h5_path, 'r')
    cwt_dset = cwt_file['cwt_images']
    
    positive_indices = []
    # 从TFRecord文件中读取标签来找到正样本
    raw_dataset = tf.data.TFRecordDataset(tfrecord_path)
    for i, raw_record in enumerate(raw_dataset):
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        label = example.features.feature['label'].float_list.value[0]
        if label == 1.0:
            positive_indices.append(i)
            if len(positive_indices) >= num_samples_to_find:
                break
    
    if not positive_indices:
        print("错误：数据集中未找到任何正样本（有窜槽的样本）。")
        return
        
    print(f"已找到以下窜槽样本索引进行分析: {positive_indices}")

    # --- 3. 生成并保存Grad-CAM图 ---
    output_plot_dir = os.path.join(RESULTS_DIR, 'classification_gradcam_plots')
    create_dir(output_plot_dir)
    print(f"Grad-CAM热力图将被保存至: {output_plot_dir}")

    last_conv_layer_name = [layer.name for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)][-1]
    print(f"Grad-CAM将作用于最后一层卷积层: '{last_conv_layer_name}'")
    
    time_axis_ms = np.arange(TIME_STEPS) * 0.01
    plot_extent = [time_axis_ms[0], time_axis_ms[-1], CWT_FREQUENCIES_KHZ[-1], CWT_FREQUENCIES_KHZ[0]]

    for i in tqdm(positive_indices, desc="生成热力图"):
        img_array = cwt_dset[i]
        
        # 扩展维度以匹配模型输入
        img_array_expanded = np.expand_dims(img_array, axis=0)
        
        # 生成热力图
        heatmap = make_gradcam_heatmap(img_array_expanded, model, last_conv_layer_name)
        
        # 调整大小以匹配原图
        heatmap_resized = tf.image.resize(np.expand_dims(heatmap, axis=-1), [img_array.shape[0], img_array.shape[1]]).numpy()

        # --- 绘图 ---
        fig, ax = plt.subplots(figsize=(18, 8))
        ax.imshow(img_array[:, :, 0], aspect='auto', cmap='gray', extent=plot_extent)
        ax.imshow(heatmap_resized, cmap='jet', alpha=0.5, extent=plot_extent, aspect='auto')
        
        ax.set_title(f'Grad-CAM for Channeling Detection - Sample {i}', fontsize=16)
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Frequency (kHz)')
        
        plot_path = os.path.join(output_plot_dir, f'gradcam_sample_{i}.png')
        plt.savefig(plot_path, dpi=150)
        plt.close(fig)

    cwt_file.close()
    print("\n--- Grad-CAM分析完成！---")

if __name__ == '__main__':
    run_classification_analysis()