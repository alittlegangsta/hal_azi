import os
import sys
import numpy as np
import h5py
import tensorflow as tf
import matplotlib.pyplot as plt
import math
from tqdm import tqdm  # <-- BUG FIX: Added this missing import

#  添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.modeling.dataset import create_dataset
from src.modeling.model import build_attention_unet
from src.utils.file_io import create_dir
from config import (
    ARRAY_ID, PROCESSED_DATA_DIR, MODEL_DIR, RESULTS_DIR,
    INPUT_SHAPE, FFT_COEFFICIENTS, MAX_PATH_DEPTH_POINTS, BATCH_SIZE, CWT_SCALES
)

def reconstruct_zc_from_fft_magnitude(fft_mag_image):
    """
    从FFT幅度图像通过逆傅里叶变换重建Zc空间图像。
    """
    n_depths, n_coeffs = fft_mag_image.shape
    
    full_fft_coeffs = np.zeros((n_depths, 180), dtype=np.complex64)
    full_fft_coeffs[:, :n_coeffs] = fft_mag_image
    
    reconstructed_zc = np.fft.ifft(full_fft_coeffs, axis=1)
    
    return np.real(reconstructed_zc)

def run_model_analysis(num_samples_to_visualize=10):
    """
    主函数：加载模型，进行预测，并可视化对比结果。
    """
    print("--- Starting AVIP Phase 4: Scientific Evaluation & Visualization ---")

    # --- 1. 加载模型和数据 ---
    print("Loading the best trained model...")
    model_path = os.path.join(MODEL_DIR, 'best_a2inet_model.h5')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}. Please run the training script first.")
    
    model = build_attention_unet(
        input_shape=INPUT_SHAPE,
        output_height=MAX_PATH_DEPTH_POINTS,
        output_width=FFT_COEFFICIENTS
    )
    model.load_weights(model_path)
    print("Model loaded successfully.")

    print("Loading dataset for analysis...")
    tfrecord_dir = os.path.join(PROCESSED_DATA_DIR, f'array_{str(ARRAY_ID).zfill(2)}', 'tfrecords')
    tfrecord_path = os.path.join(tfrecord_dir, 'translation_data.tfrecord')
    
    analysis_dataset = create_dataset(tfrecord_path, batch_size=BATCH_SIZE, is_training=False)
    
    # --- 2. 进行预测 ---
    print(f"Running predictions on {num_samples_to_visualize} samples...")
    features_to_visualize = []
    labels_to_visualize = []
    
    num_batches_needed = math.ceil(num_samples_to_visualize / BATCH_SIZE)
    
    for features, labels in analysis_dataset.take(num_batches_needed):
        features_to_visualize.append(features.numpy())
        labels_to_visualize.append(labels.numpy())

    features_to_visualize = np.concatenate(features_to_visualize, axis=0)[:num_samples_to_visualize]
    labels_to_visualize = np.concatenate(labels_to_visualize, axis=0)[:num_samples_to_visualize]

    predictions = model.predict(features_to_visualize)
    print("Prediction complete.")

    # --- 3. 可视化结果 ---
    output_plot_dir = os.path.join(RESULTS_DIR, 'analysis_plots')
    create_dir(output_plot_dir)
    print(f"Saving visualization plots to: {output_plot_dir}")

    for i in tqdm(range(num_samples_to_visualize), desc="Generating plots"):
        input_cwt = features_to_visualize[i]
        true_fft_label = labels_to_visualize[i]
        pred_fft_label = predictions[i]
        
        true_zc_image = reconstruct_zc_from_fft_magnitude(true_fft_label)
        pred_zc_image = reconstruct_zc_from_fft_magnitude(pred_fft_label)
        
        fig, axes = plt.subplots(1, 3, figsize=(20, 7))
        fig.suptitle(f'AVIP Analysis - Sample {i}', fontsize=16)
        
        ax1 = axes[0]
        im1 = ax1.imshow(input_cwt[:, :, 0], aspect='auto', cmap='jet', 
                         extent=[0, input_cwt.shape[1] * 0.01, CWT_SCALES[-1], CWT_SCALES[0]])
        ax1.set_title('Input: Sonic CWT Image (Channel A)')
        ax1.set_xlabel('Time (ms)')
        ax1.set_ylabel('CWT Scale')
        fig.colorbar(im1, ax=ax1)

        ax2 = axes[1]
        im2 = ax2.imshow(true_zc_image, aspect='auto', cmap='viridis', vmin=0, vmax=5)
        ax2.set_title('Ground Truth: Reconstructed Zc Image')
        ax2.set_xlabel('Azimuthal Angle (0-360 deg)')
        ax2.set_ylabel('Relative Depth Points')
        fig.colorbar(im2, ax=ax2)

        ax3 = axes[2]
        im3 = ax3.imshow(pred_zc_image, aspect='auto', cmap='viridis', vmin=0, vmax=5)
        ax3.set_title('Prediction: Reconstructed Zc Image')
        ax3.set_xlabel('Azimuthal Angle (0-360 deg)')
        ax3.set_ylabel('Relative Depth Points')
        fig.colorbar(im3, ax=ax3)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        plot_path = os.path.join(output_plot_dir, f'sample_{i}_comparison.png')
        plt.savefig(plot_path)
        plt.close(fig)

    print("\n--- Analysis and Visualization Complete! ---")

if __name__ == '__main__':
    run_model_analysis()