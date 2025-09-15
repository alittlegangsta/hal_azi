import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import math
from tqdm import tqdm
from scipy.ndimage import zoom
from mpl_toolkits.axes_grid1 import make_axes_locatable

#  添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.modeling.dataset import create_dataset
from src.modeling.model import build_attention_unet
from src.utils.file_io import create_dir
from config import (
    ARRAY_ID, PROCESSED_DATA_DIR, MODEL_DIR, RESULTS_DIR,
    INPUT_SHAPE, FFT_COEFFICIENTS, MAX_PATH_DEPTH_POINTS, BATCH_SIZE,
    CWT_FREQUENCIES_KHZ, TIME_STEPS,
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
    
    zoom_factors = [
        img_array.shape[0] / heatmap.shape[0],
        img_array.shape[1] / heatmap.shape[1]
    ]
    heatmap_resized = zoom(heatmap, zoom_factors)
    
    return heatmap_resized


def reconstruct_zc_from_fft_magnitude(fft_mag_image):
    n_depths, n_coeffs = fft_mag_image.shape
    full_fft_coeffs = np.zeros((n_depths, 180), dtype=np.complex64)
    full_fft_coeffs[:, :n_coeffs] = fft_mag_image
    reconstructed_zc = np.fft.ifft(full_fft_coeffs, axis=1)
    return np.real(reconstructed_zc)


def run_model_analysis(num_samples_to_visualize=10):
    print("--- Starting AVIP Phase 4: Scientific Evaluation & Visualization (Final Version) ---")
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
    
    time_axis_ms = np.arange(TIME_STEPS) * 0.01

    for i in tqdm(range(num_samples_to_visualize), desc="Generating plots"):
        input_cwt, true_fft_label, pred_fft_label = features_to_visualize[i], labels_to_visualize[i], predictions[i]
        
        heatmap = get_grad_cam_heatmap(model, input_cwt, LAST_CONV_LAYER_NAME)
        true_zc_image = reconstruct_zc_from_fft_magnitude(true_fft_label)
        pred_zc_image = reconstruct_zc_from_fft_magnitude(pred_fft_label)
        
        plot_extent = [time_axis_ms[0], time_axis_ms[-1], CWT_FREQUENCIES_KHZ[-1], CWT_FREQUENCIES_KHZ[0]]
        
        # --- Figure 1: CWT 和 Grad-CAM (最终修正：灰度背景 + 彩色叠加) ---
        fig1, ax1 = plt.subplots(1, 1, figsize=(18, 8))
        fig1.suptitle(f'Input & Explanation - Sample {i}', fontsize=16)
        
        # 1. 将底层的CWT图绘制为灰度图
        ax1.imshow(input_cwt[:, :, 0], aspect='auto', cmap='gray', extent=plot_extent)
        
        # 2. 直接在同一个ax1上，叠加绘制半透明的彩色热力图
        ax1.imshow(heatmap, cmap='jet', alpha=0.5, extent=plot_extent, aspect='auto')
        
        ax1.set_title('Input CWT (Grayscale) & Grad-CAM Heatmap (Color Overlay)')
        ax1.set_xlabel('Time (ms)')
        ax1.set_ylabel('Frequency (kHz)')
        
        fig1.tight_layout(rect=[0, 0, 1, 0.96])
        plot_path1 = os.path.join(output_plot_dir, f'sample_{i}_input_and_gradcam.png')
        plt.savefig(plot_path1, dpi=150)
        plt.close(fig1)

        # --- Figure 2: 真值和预测结果对比 ---
        fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(12, 8))
        fig2.suptitle(f'Ground Truth vs. Prediction - Sample {i}', fontsize=16)

        im3 = ax3.imshow(true_zc_image, aspect='auto', cmap='viridis', vmin=0, vmax=5)
        ax3.set_title('Ground Truth')
        ax3.set_xlabel('Azimuthal Angle')
        ax3.set_ylabel('Relative Depth Points')
        divider3 = make_axes_locatable(ax3)
        cax3 = divider3.append_axes("right", size="5%", pad=0.1)
        fig2.colorbar(im3, cax=cax3, label='Acoustic Impedance (Zc)')
        
        im4 = ax4.imshow(pred_zc_image, aspect='auto', cmap='viridis', vmin=0, vmax=5)
        ax4.set_title('Prediction')
        ax4.set_xlabel('Azimuthal Angle')
        divider4 = make_axes_locatable(ax4)
        cax4 = divider4.append_axes("right", size="5%", pad=0.1)
        fig2.colorbar(im4, cax=cax4, label='Acoustic Impedance (Zc)')

        fig2.tight_layout(rect=[0, 0, 1, 0.95])
        plot_path2 = os.path.join(output_plot_dir, f'sample_{i}_truth_vs_prediction.png')
        plt.savefig(plot_path2, dpi=150)
        plt.close(fig2)

    print("\n--- Analysis and Visualization Complete! ---")

if __name__ == '__main__':
    run_model_analysis()