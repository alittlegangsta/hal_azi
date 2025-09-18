# 文件路径: src/visualization/visualize_data_pipeline.py

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import h5py
import pywt

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data_processing.main_preprocess import apply_high_pass_filter
from src.utils.file_io import create_dir, load_pickle
from config import (
    ARRAY_ID, PROCESSED_DATA_DIR, RAW_DATA_DIR, CWT_SCALES, CWT_WAVELET,
    FFT_COEFFICIENTS, MAX_PATH_DEPTH_POINTS, ROOT_DIR,
    CWT_FREQUENCIES_KHZ, TIME_STEPS # 导入新变量
)

def visualize_data_processing_steps(sample_index=100):
    """
    可视化单个样本在整个数据处理流程中的变化。
    """
    print("--- Starting Visualization of Data Processing Pipeline ---")

    output_dir = os.path.join(ROOT_DIR, 'output', 'visualization_plots')
    create_dir(output_dir)

    # --- 1. 加载数据 ---
    sonic_filename = f'XSILMR{str(ARRAY_ID).zfill(2)}.mat'
    mat_content = loadmat(os.path.join(RAW_DATA_DIR, 'XSILMR', sonic_filename))
    struct_name = f'XSILMR{str(ARRAY_ID).zfill(2)}'
    sonic_data = mat_content[struct_name][0, 0]
    raw_waveform = sonic_data[f'WaveRng{str(ARRAY_ID).zfill(2)}SideA'][:TIME_STEPS, sample_index]

    array_dir = os.path.join(PROCESSED_DATA_DIR, f'array_{str(ARRAY_ID).zfill(2)}')
    processed_waveforms_path = os.path.join(array_dir, 'processed_waveforms.pkl')
    cwt_h5_path = os.path.join(array_dir, 'cwt_images.h5')
    ground_truth_db_path = os.path.join(array_dir, f'ground_truth_db_array_{str(ARRAY_ID).zfill(2)}.h5')

    processed_data = load_pickle(processed_waveforms_path)
    filtered_waveform_stack = processed_data['waveforms'][sample_index]
    sonic_depth = processed_data['sonic_depths'][sample_index]

    with h5py.File(cwt_h5_path, 'r') as hf:
        cwt_image = hf['cwt_images'][sample_index]

    with h5py.File(ground_truth_db_path, 'r') as hf:
        sonic_depth_key = str(sonic_depth).replace('.', '_')
        zc_slice = hf['path_data'][sonic_depth_key][:]

    # --- 2. 可视化滤波效果 ---
    time_axis_ms = np.arange(TIME_STEPS) * 0.01 # 采样间隔 10us = 0.01ms

    plt.figure(figsize=(15, 5))
    plt.suptitle('Step 1: High-Pass Filtering Effect', fontsize=16)
    
    plt.subplot(1, 2, 1)
    plt.plot(time_axis_ms, raw_waveform)
    plt.title('Raw Waveform (A-channel)')
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude')

    plt.subplot(1, 2, 2)
    plt.plot(time_axis_ms, filtered_waveform_stack[0, :])
    plt.title('Waveform After 1000Hz High-Pass Filter')
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude')
    plt.savefig(os.path.join(output_dir, '01_filtering_effect.png'))
    plt.close()
    print("Saved plot: 01_filtering_effect.png")

    # --- 3. 可视化CWT结果 ---
    plt.figure(figsize=(8, 6))
    # *** 绘图修正 ***
    plt.imshow(cwt_image[:, :, 0], aspect='auto', cmap='jet',
               extent=[time_axis_ms[0], time_axis_ms[-1], CWT_FREQUENCIES_KHZ[-1], CWT_FREQUENCIES_KHZ[0]])
    plt.title('Step 2: CWT Result (A-channel)')
    plt.xlabel('Time (ms)')
    plt.ylabel('Frequency (kHz)')
    plt.colorbar(label='Amplitude')
    plt.savefig(os.path.join(output_dir, '02_cwt_result.png'))
    plt.close()
    print("Saved plot: 02_cwt_result.png")
    
    # --- 4. 可视化标签生成过程 ---
    avg_zc_profile = np.mean(zc_slice, axis=0)
    
    # ==============================================================================
    # >>>>>>>>>> 代码修改区域 <<<<<<<<<<<
    # ==============================================================================
    # 同样应用对数变换，以确保可视化结果与模型训练时使用的标签一致
    fft_magnitudes_raw = np.abs(np.fft.fft(avg_zc_profile))
    fft_magnitudes = np.log1p(fft_magnitudes_raw)[:FFT_COEFFICIENTS]
    # ==============================================================================
    # <<<<<<<<<<<<<<<<<<<<<<<<<< 修改区域结束 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # ==============================================================================
    
    plt.figure(figsize=(15, 5))
    plt.suptitle('Step 3: Label Generation Process (with Log Scaling)', fontsize=16)
    plt.subplot(1, 2, 1)
    plt.imshow(zc_slice, aspect='auto', cmap='viridis', vmin=0, vmax=5)
    plt.title('Original Zc Slice (Ground Truth)')
    plt.xlabel('Azimuthal Angle')
    plt.ylabel('Relative Depth Points')
    plt.subplot(1, 2, 2)
    plt.plot(fft_magnitudes)
    plt.title(f'Final Label: Log-Scaled FFT Magnitudes (First {FFT_COEFFICIENTS} coeffs)')
    plt.xlabel('FFT Coefficient Index')
    plt.ylabel('Log(1 + Magnitude)')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, '03_label_generation.png'))
    plt.close()
    print("Saved plot: 03_label_generation.png")
    
    print("\n--- Visualization of Data Pipeline Complete! ---")
    print(f"All plots saved in: {output_dir}")

if __name__ == '__main__':
    visualize_data_processing_steps(sample_index=150)