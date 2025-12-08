# 文件路径: src/visualization/visualize_data_pipeline.py
# (新分支 'percentage_' 的版本)

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import h5py

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.file_io import create_dir, load_pickle
# 导入两个处理函数
from src.data_processing.create_tfrecords import _process_zc_slice_to_1d_label, _process_zc_slice_to_fft_label
from config import (
    ARRAY_ID, PROCESSED_DATA_DIR, ROOT_DIR,
    MAX_PATH_DEPTH_POINTS, TASK_TYPE
)

def visualize_label_generation(sample_index=150):
    
    print(f"\n--- Visualizing Label Generation for TASK_TYPE: {TASK_TYPE} (Sample Index: {sample_index}) ---")

    # --- 1. 加载数据 ---
    array_dir = os.path.join(PROCESSED_DATA_DIR, f'array_{str(ARRAY_ID).zfill(2)}')
    processed_waveforms_path = os.path.join(array_dir, 'processed_waveforms.pkl')
    ground_truth_db_path = os.path.join(array_dir, f'ground_truth_db_array_{str(ARRAY_ID).zfill(2)}.h5')
    try:
        processed_data = load_pickle(processed_waveforms_path)
        sonic_depth = processed_data['sonic_depths'][sample_index]
        with h5py.File(ground_truth_db_path, 'r') as hf:
            sonic_depth_key = str(sonic_depth).replace('.', '_')
            if sonic_depth_key not in hf['path_data']:
                print(f"  - WARNING: Sonic depth not found. Skipping.")
                return
            zc_slice = hf['path_data'][sonic_depth_key][:]
    except (FileNotFoundError, IndexError) as e:
        print(f"  - ERROR: Could not load data. Error: {e}")
        return

    # --- 2. 智能可视化 ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={'width_ratios': [1, 1.5]})
    fig.suptitle(f'Label Generation Process (Sample {sample_index}) - {TASK_TYPE}', fontsize=18)

    # a. 左图: 原始 Zc 切片 (共用)
    im_zc = axes[0].imshow(zc_slice, aspect='auto', cmap='viridis', vmin=0, vmax=5,
                           extent=[0, 180, zc_slice.shape[0], 0])
    axes[0].set_title('Step 1: Original Zc Slice (Input)')
    axes[0].set_xlabel('Azimuthal Angle')
    axes[0].set_ylabel('Relative Depth Points')
    plt.colorbar(im_zc, ax=axes[0], label='Acoustic Impedance (Zc)')

    if TASK_TYPE == 'fft_regression':
        # --- FFT 任务的可视化 ---
        final_label = _process_zc_slice_to_fft_label(zc_slice, MAX_PATH_DEPTH_POINTS)
        im_fft = axes[1].imshow(final_label, aspect='auto', cmap='magma', 
                                extent=[0, final_label.shape[1], final_label.shape[0], 0])
        axes[1].set_title('Step 2: 2D FFT Magnitude Profile (Final Label)')
        axes[1].set_xlabel('FFT Coefficient Index')
        axes[1].set_yticklabels([])
        plt.colorbar(im_fft, ax=axes[1], label='Log(1 + Magnitude)')
        center_y = zc_slice.shape[0] / 2
        axes[0].annotate('Perform 1D-FFT\nat each depth point',
                         xy=(185, center_y), xycoords='data',
                         xytext=(240, center_y), textcoords='data',
                         arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
                         fontsize=12, ha='center', va='center')
    else:
        # --- 1D 剖面图任务的可视化 ---
        final_label = _process_zc_slice_to_1d_label(zc_slice, MAX_PATH_DEPTH_POINTS)
        axes[1].plot(final_label, np.arange(len(final_label)), color='darkorange', linewidth=2)
        axes[1].set_title('Step 2: 1D "Channeling Percentage" Profile (Final Label)')
        axes[1].set_xlabel('Channeling Percentage (%)')
        axes[1].set_yticklabels([])
        axes[1].set_xlim(0, 100)
        axes[1].set_ylim(len(final_label), 0)
        axes[1].grid(True, linestyle='--', alpha=0.7)
        center_y = zc_slice.shape[0] / 2
        axes[0].annotate('Calculate percentage of\npixels with Zc < 2.5\nat each depth point',
                         xy=(185, center_y), xycoords='data',
                         xytext=(240, center_y), textcoords='data',
                         arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
                         fontsize=12, ha='center', va='center')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    output_dir = os.path.join(ROOT_DIR, 'output', 'visualization_plots', TASK_TYPE)
    create_dir(output_dir)
    output_path = os.path.join(output_dir, f'06_label_generation_{TASK_TYPE}.png')
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"  - Visualization plot saved successfully to: {output_path}")

if __name__ == '__main__':
    visualize_label_generation()