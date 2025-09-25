# 文件路径: src/interpretation/run_analysis_classification.py

import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import math
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.modeling.dataset import create_dataset
from src.modeling.model import build_attention_unet
from src.utils.file_io import create_dir
from config import (
    ARRAY_ID, PROCESSED_DATA_DIR, MODEL_DIR, RESULTS_DIR,
    INPUT_SHAPE, FFT_COEFFICIENTS, MAX_PATH_DEPTH_POINTS, BATCH_SIZE
)

# ==============================================================================
# >>>>>>>>>> 全新功能：分割任务评估指标 <<<<<<<<<<<
# ==============================================================================
def dice_coefficient(y_true, y_pred, smooth=1e-6):
    """
    计算Dice系数，用于衡量两个二元掩码的重合度。
    """
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

def reconstruct_mask_from_fft(fft_image_channel):
    """
    对单个通道的FFT幅度图进行逆变换，重建出二元掩码。
    """
    n_depths, n_coeffs = fft_image_channel.shape
    full_fft_coeffs = np.zeros((n_depths, 180), dtype=np.complex64)
    # 注意：我们不再需要exp变换，因为标签本身就是FFT幅度
    full_fft_coeffs[:, :n_coeffs] = fft_image_channel
    
    reconstructed_mask_complex = np.fft.ifft(full_fft_coeffs, axis=1)
    reconstructed_mask = np.real(reconstructed_mask_complex)
    
    # 将重建结果二值化，以得到清晰的0/1掩码
    # 我们取0.5作为阈值，因为模型的sigmoid输出在(0,1)之间
    binary_mask = (reconstructed_mask > 0.5).astype(np.float32)
    return binary_mask
# ==============================================================================
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# ==============================================================================


def run_model_analysis(num_samples_to_visualize=10):
    print("--- Starting AVIP Phase 4: Classification Model Analysis ---")
    
    model_path = os.path.join(MODEL_DIR, 'best_a2inet_model.h5')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}. Please run training first.")
    
    # 加载不包含自定义损失函数的模型
    model = tf.keras.models.load_model(model_path, compile=False)
    print("Model loaded successfully.")
    
    tfrecord_dir = os.path.join(PROCESSED_DATA_DIR, f'array_{str(ARRAY_ID).zfill(2)}', 'tfrecords')
    tfrecord_path = os.path.join(tfrecord_dir, 'translation_data.tfrecord')
    analysis_dataset = create_dataset(tfrecord_path, batch_size=BATCH_SIZE, is_training=False)
    
    print(f"Running predictions on {num_samples_to_visualize} samples...")
    features_to_visualize, labels_to_visualize = [], []
    num_batches_needed = math.ceil(num_samples_to_visualize / BATCH_SIZE)
    
    for features, labels in analysis_dataset.take(num_batches_needed):
        features_to_visualize.append(features.numpy())
        labels_to_visualize.append(labels.numpy())
    features_to_visualize = np.concatenate(features_to_visualize, axis=0)[:num_samples_to_visualize]
    labels_to_visualize = np.concatenate(labels_to_visualize, axis=0)[:num_samples_to_visualize]
    
    predictions = model.predict(features_to_visualize, batch_size=BATCH_SIZE)
    print("Prediction complete.")

    output_plot_dir = os.path.join(RESULTS_DIR, 'classification_analysis_plots')
    create_dir(output_plot_dir)
    print(f"Saving visualization plots to: {output_plot_dir}")
    
    all_dice_scores = []

    for i in tqdm(range(num_samples_to_visualize), desc="Generating plots and metrics"):
        true_label = labels_to_visualize[i]
        pred_label = predictions[i]
        
        # --- 分别重建真实和预测的“窜槽”掩码 ---
        true_channeling_fft = true_label[:, :, 0]
        pred_channeling_fft = pred_label[:, :, 0]
        
        true_mask = reconstruct_mask_from_fft(true_channeling_fft)
        pred_mask = reconstruct_mask_from_fft(pred_channeling_fft)
        
        # --- 计算Dice系数 ---
        dice_score = dice_coefficient(true_mask, pred_mask)
        all_dice_scores.append(dice_score)
        
        print(f"\n--- Metrics for Sample {i} ---")
        print(f"  - Channeling Mask Dice Coefficient: {dice_score:.4f}")

        # --- 可视化对比 ---
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        fig.suptitle(f'Channeling Mask Prediction - Sample {i}\nDice Score: {dice_score:.4f}', fontsize=16)

        ax1.imshow(true_mask, cmap='gray_r', vmin=0, vmax=1)
        ax1.set_title('Ground Truth Channeling Mask')
        ax1.set_xlabel('Azimuthal Angle')
        ax1.set_ylabel('Relative Depth Points')

        ax2.imshow(pred_mask, cmap='gray_r', vmin=0, vmax=1)
        ax2.set_title('Predicted Channeling Mask')
        ax2.set_xlabel('Azimuthal Angle')

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        plot_path = os.path.join(output_plot_dir, f'sample_{i}_mask_comparison.png')
        plt.savefig(plot_path, dpi=150)
        plt.close(fig)

    print("\n" + "="*50)
    print("--- Average Metrics Across All Visualized Samples ---")
    print(f"  - Average Channeling Mask Dice Score: {np.mean(all_dice_scores):.4f}")
    print("="*50)

    print("\n--- Analysis and Visualization Complete! ---")

if __name__ == '__main__':
    run_model_analysis()