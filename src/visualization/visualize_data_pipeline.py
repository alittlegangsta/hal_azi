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
    CWT_FREQUENCIES_KHZ, TIME_STEPS
)


def visualize_single_sample(sample_index, output_dir_base):
    """
    对单个指定样本进行完整的可视化流程。
    """
    print(f"\n--- Visualizing Sample Index: {sample_index} ---")

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

    try:
        processed_data = load_pickle(processed_waveforms_path)
        sonic_depth = processed_data['sonic_depths'][sample_index]
        with h5py.File(ground_truth_db_path, 'r') as hf:
            sonic_depth_key = str(sonic_depth).replace('.', '_')
            if sonic_depth_key not in hf['path_data']:
                print(f"  - WARNING: Sonic depth for index {sample_index} not found in DB. Skipping.")
                return
            zc_slice = hf['path_data'][sonic_depth_key][:]
    except (FileNotFoundError, IndexError) as e:
        print(f"  - ERROR: Could not load data for index {sample_index}. Have you run preprocessing? Error: {e}")
        return

    # --- 2. 可视化滤波与CWT (与之前版本类似) ---
    time_axis_ms = np.arange(TIME_STEPS) * 0.01
    plt.figure(figsize=(10, 5))
    plt.plot(time_axis_ms, raw_waveform)
    plt.title(f'Step 1: Raw Waveform (Sample {sample_index})')
    plt.xlabel('Time (ms)'); plt.ylabel('Amplitude')
    plt.savefig(os.path.join(output_dir_base, f'{sample_index}_01_raw_waveform.png'))
    plt.close()
    
    with h5py.File(cwt_h5_path, 'r') as hf:
        cwt_image = hf['cwt_images'][sample_index]
    plt.figure(figsize=(8, 6))
    plt.imshow(cwt_image[:, :, 0], aspect='auto', cmap='jet',
               extent=[time_axis_ms[0], time_axis_ms[-1], CWT_FREQUENCIES_KHZ[-1], CWT_FREQUENCIES_KHZ[0]])
    plt.title(f'Step 2: CWT Result (Sample {sample_index})')
    plt.xlabel('Time (ms)'); plt.ylabel('Frequency (kHz)')
    plt.colorbar(label='Amplitude')
    plt.savefig(os.path.join(output_dir_base, f'{sample_index}_02_cwt_result.png'))
    plt.close()

    # ==============================================================================
    # >>>>>>>>>> 代码修改区域 V5.0：详细可视化新的标签生成流程 <<<<<<<<<<<
    # ==============================================================================

    # --- 3. 可视化掩码生成与FFT系数 ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Step 3: Label Generation Process (Sample {sample_index})', fontsize=18)

    # a. 原始Zc图
    im0 = axes[0, 0].imshow(zc_slice, aspect='auto', cmap='viridis', vmin=0, vmax=5)
    axes[0, 0].set_title('Original Zc Slice')
    axes[0, 0].set_xlabel('Azimuthal Angle'); axes[0, 0].set_ylabel('Relative Depth')
    plt.colorbar(im0, ax=axes[0, 0])

    # b. 窜槽掩码
    channeling_mask = (zc_slice < 2.5).astype(np.float32)
    im1 = axes[0, 1].imshow(channeling_mask, aspect='auto', cmap='gray_r', vmin=0, vmax=1)
    axes[0, 1].set_title('Channeling Mask (Zc < 2.5)')
    axes[0, 1].set_xlabel('Azimuthal Angle'); axes[0, 1].set_ylabel('Relative Depth')
    
    # c. 良好掩码
    good_bonding_mask = (zc_slice >= 2.5).astype(np.float32)
    im2 = axes[1, 0].imshow(good_bonding_mask, aspect='auto', cmap='gray_r', vmin=0, vmax=1)
    axes[1, 0].set_title('Good Bonding Mask (Zc >= 2.5)')
    axes[1, 0].set_xlabel('Azimuthal Angle'); axes[1, 0].set_ylabel('Relative Depth')

    # d. 两个通道的平均FFT谱
    avg_fft_channeling = np.mean(np.abs(np.fft.fft(channeling_mask, axis=1)), axis=0)
    avg_fft_good = np.mean(np.abs(np.fft.fft(good_bonding_mask, axis=1)), axis=0)
    axes[1, 1].plot(avg_fft_channeling[:FFT_COEFFICIENTS], label='Channeling FFT Spectrum', color='red')
    axes[1, 1].plot(avg_fft_good[:FFT_COEFFICIENTS], label='Good Bonding FFT Spectrum', color='green')
    axes[1, 1].set_title(f'Average FFT Magnitudes (First {FFT_COEFFICIENTS} Coeffs)')
    axes[1, 1].set_xlabel('FFT Coefficient Index'); axes[1, 1].set_ylabel('Magnitude')
    axes[1, 1].legend(); axes[1, 1].grid(True)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir_base, f'{sample_index}_03_label_generation.png'))
    plt.close()

    # --- 4. 可视化最终作为标签的二维图 ---
    from src.data_processing.create_tfrecords import process_zc_slice_to_label
    final_label_image = process_zc_slice_to_label(zc_slice, FFT_COEFFICIENTS, MAX_PATH_DEPTH_POINTS)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f'Step 4: Final 2D Label for Model (Sample {sample_index})', fontsize=16)
    
    im_ch0 = axes[0].imshow(final_label_image[:, :, 0], aspect='auto', cmap='Reds')
    axes[0].set_title('Channel 0: Channeling FFT Label')
    axes[0].set_xlabel('FFT Coefficient Index'); axes[0].set_ylabel('Padded Depth')
    plt.colorbar(im_ch0, ax=axes[0])

    im_ch1 = axes[1].imshow(final_label_image[:, :, 1], aspect='auto', cmap='Greens')
    axes[1].set_title('Channel 1: Good Bonding FFT Label')
    axes[1].set_xlabel('FFT Coefficient Index'); axes[1].set_ylabel('Padded Depth')
    plt.colorbar(im_ch1, ax=axes[1])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir_base, f'{sample_index}_04_final_label_image.png'))
    plt.close()

    print(f"  - Visualizations for sample {sample_index} saved successfully.")

# ==============================================================================
# <<<<<<<<<<<<<<<<<<<<<<<<<< 修改区域结束 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# ==============================================================================

if __name__ == '__main__':
    """
    主执行函数：为几个有代表性的样本生成完整的可视化流程图。
    """
    print("--- Starting Enhanced Visualization of Data Processing Pipeline ---")
    output_dir = os.path.join(ROOT_DIR, 'output', 'visualization_plots')
    create_dir(output_dir)
    print(f"All plots will be saved in: {output_dir}")

    # --- 选择几个有代表性的样本进行可视化 ---
    # 您可以根据需要修改这些索引
    # - 150: 通常是一个有窜槽的典型样本
    # - 2500: 通常是一个胶结良好的典型样本
    representative_indices = [150, 2500] 

    for index in representative_indices:
        visualize_single_sample(index, output_dir)
    
    print("\n--- Visualization of Data Pipeline Complete! ---")