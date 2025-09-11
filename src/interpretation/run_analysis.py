import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import math
from tqdm import tqdm

#  添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.modeling.dataset import create_dataset
from src.modeling.model import build_attention_unet
from src.utils.file_io import create_dir
from config import (
    ARRAY_ID, PROCESSED_DATA_DIR, MODEL_DIR, RESULTS_DIR,
    INPUT_SHAPE, FFT_COEFFICIENTS, MAX_PATH_DEPTH_POINTS, BATCH_SIZE,
    CWT_FREQUENCIES_KHZ, TIME_STEPS, # 导入新变量
    LAST_CONV_LAYER_NAME
)


def get_grad_cam_heatmap(model, img_array, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        img_array_expanded = np.expand_dims(img_array, axis=0)
        last_conv_layer_output, preds = grad_model(img_array_expanded)
        class_channel = tf.reduce_mean(preds[0], axis=[0, 1])
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy()


def reconstruct_zc_from_fft_magnitude(fft_mag_image):
    n_depths, n_coeffs = fft_mag_image.shape
    full_fft_coeffs = np.zeros((n_depths, 180), dtype=np.complex64)
    full_fft_coeffs[:, :n_coeffs] = fft_mag_image
    reconstructed_zc = np.fft.ifft(full_fft_coeffs, axis=1)
    return np.real(reconstructed_zc)


def run_model_analysis(num_samples_to_visualize=10):
    print("--- Starting AVIP Phase 4: Scientific Evaluation & Visualization with Grad-CAM ---")
    print("Loading the best trained model...")
    model_path = os.path.join(MODEL_DIR, 'best_a2inet_model.h5')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}. Please run the training script first.")
    
    model = build_attention_unet(
        input_shape=INPUT_SHAPE, output_height=MAX_PATH_DEPTH_POINTS, output_width=FFT_COEFFICIENTS
    )
    model.load_weights(model_path)
    print("Model loaded successfully.")
    print("Loading dataset for analysis...")
    tfrecord_dir = os.path.join(PROCESSED_DATA_DIR, f'array_{str(ARRAY_ID).zfill(2)}', 'tfrecords')
    tfrecord_path = os.path.join(tfrecord_dir, 'translation_data.tfrecord')
    analysis_dataset = create_dataset(tfrecord_path, batch_size=BATCH_SIZE, is_training=False)
    
    print(f"Running predictions and Grad-CAM on {num_samples_to_visualize} samples...")
    features_to_visualize, labels_to_visualize = [], []
    num_batches_needed = math.ceil(num_samples_to_visualize / BATCH_SIZE)
    
    for features, labels in analysis_dataset.take(num_batches_needed):
        features_to_visualize.append(features.numpy())
        labels_to_visualize.append(labels.numpy())
    features_to_visualize = np.concatenate(features_to_visualize, axis=0)[:num_samples_to_visualize]
    labels_to_visualize = np.concatenate(labels_to_visualize, axis=0)[:num_samples_to_visualize]
    predictions = model.predict(features_to_visualize, batch_size=BATCH_SIZE)
    print("Prediction complete.")

    output_plot_dir = os.path.join(RESULTS_DIR, 'analysis_plots_with_gradcam')
    create_dir(output_plot_dir)
    print(f"Saving visualization plots to: {output_plot_dir}")
    
    time_axis_ms = np.arange(TIME_STEPS) * 0.01 # 采样间隔 10us = 0.01ms

    for i in tqdm(range(num_samples_to_visualize), desc="Generating plots"):
        input_cwt, true_fft_label, pred_fft_label = features_to_visualize[i], labels_to_visualize[i], predictions[i]
        
        heatmap = get_grad_cam_heatmap(model, input_cwt, LAST_CONV_LAYER_NAME)
        heatmap_resized = tf.image.resize(np.expand_dims(heatmap, axis=-1), [input_cwt.shape[0], input_cwt.shape[1]]).numpy()
        true_zc_image = reconstruct_zc_from_fft_magnitude(true_fft_label)
        pred_zc_image = reconstruct_zc_from_fft_magnitude(pred_fft_label)
        
        fig, axes = plt.subplots(1, 4, figsize=(26, 6))
        fig.suptitle(f'AVIP Analysis with Grad-CAM - Sample {i}', fontsize=16)
        
        # --- 绘图修正 ---
        plot_extent = [time_axis_ms[0], time_axis_ms[-1], CWT_FREQUENCIES_KHZ[-1], CWT_FREQUENCIES_KHZ[0]]
        
        ax1 = axes[0]
        ax1.imshow(input_cwt[:, :, 0], aspect='auto', cmap='jet', extent=plot_extent)
        ax1.set_title('Input: Sonic CWT Image')
        ax1.set_xlabel('Time (ms)'), ax1.set_ylabel('Frequency (kHz)')
        
        ax2 = axes[1]
        ax2.imshow(input_cwt[:, :, 0], aspect='auto', cmap='jet', extent=plot_extent)
        ax2.imshow(heatmap_resized, cmap='hot', alpha=0.5, extent=plot_extent)
        ax2.set_title('Explanation: Grad-CAM Heatmap')
        ax2.set_xlabel('Time (ms)'), ax2.set_ylabel('Frequency (kHz)')
        
        ax3 = axes[2]
        im3 = ax3.imshow(true_zc_image, aspect='auto', cmap='viridis', vmin=0, vmax=5)
        ax3.set_title('Ground Truth: Reconstructed Zc Image')
        ax3.set_xlabel('Azimuthal Angle (0-360 deg)'), ax3.set_ylabel('Relative Depth Points')
        
        ax4 = axes[3]
        im4 = ax4.imshow(pred_zc_image, aspect='auto', cmap='viridis', vmin=0, vmax=5)
        ax4.set_title('Prediction: Reconstructed Zc Image')
        ax4.set_xlabel('Azimuthal Angle (0-360 deg)'), ax4.set_ylabel('Relative Depth Points')
        fig.colorbar(im4, ax=[ax3, ax4], shrink=0.8)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plot_path = os.path.join(output_plot_dir, f'sample_{i}_comparison_gradcam.png')
        plt.savefig(plot_path)
        plt.close(fig)

    print("\n--- Analysis and Visualization Complete! ---")

if __name__ == '__main__':
    run_model_analysis()