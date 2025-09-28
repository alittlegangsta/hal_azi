# 文件路径: src/visualization/visualize_data_pipeline.py

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import h5py

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.file_io import create_dir, load_pickle
from src.data_processing.create_tfrecords import process_zc_slice_to_label
from config import (
    ARRAY_ID, PROCESSED_DATA_DIR, RAW_DATA_DIR, ROOT_DIR,
    MAX_PATH_DEPTH_POINTS
)

def visualize_profile_label_generation(sample_index=150):
    """
    Visualizes the entire process of generating the 1D "Channeling Percentage" profile label from a raw Zc slice.
    """
    print(f"\n--- Visualizing 1D Profile Label Generation for Sample Index: {sample_index} ---")

    # --- 1. Load required raw data ---
    array_dir = os.path.join(PROCESSED_DATA_DIR, f'array_{str(ARRAY_ID).zfill(2)}')
    processed_waveforms_path = os.path.join(array_dir, 'processed_waveforms.pkl')
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
        print(f"  - ERROR: Could not load data for index {sample_index}. Please ensure preprocessing has been run. Error: {e}")
        return

    # --- 2. Generate the final 1D label ---
    final_1d_label = process_zc_slice_to_label(zc_slice, MAX_PATH_DEPTH_POINTS)
    
    # --- 3. Create visualization plot ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={'width_ratios': [1, 1.5]})
    fig.suptitle(f'Label Generation Process (Sample {sample_index})', fontsize=18)

    # a. Left plot: Original 2D Zc Slice
    im = axes[0].imshow(zc_slice, aspect='auto', cmap='viridis', vmin=0, vmax=5,
                        extent=[0, 180, zc_slice.shape[0], 0])
    axes[0].set_title('Step 1: Original Zc Slice (Input)')
    axes[0].set_xlabel('Azimuthal Angle')
    axes[0].set_ylabel('Relative Depth Points')
    plt.colorbar(im, ax=axes[0], label='Acoustic Impedance (Zc)')

    # b. Right plot: Generated 1D Profile Label
    axes[1].plot(final_1d_label, np.arange(len(final_1d_label)), color='darkorange', linewidth=2)
    axes[1].set_title('Step 2: 1D "Channeling Percentage" Profile (Final Label)')
    axes[1].set_xlabel('Channeling Percentage (%)')
    axes[1].set_yticklabels([])
    axes[1].set_xlim(0, 100)
    axes[1].set_ylim(len(final_1d_label), 0)
    axes[1].grid(True, linestyle='--', alpha=0.7)
    
    # Add arrow and text annotation
    center_y = zc_slice.shape[0] / 2
    axes[0].annotate('Calculate percentage of\npixels with Zc < 2.5\nat each depth point',
                     xy=(185, center_y), xycoords='data',
                     xytext=(240, center_y), textcoords='data',
                     arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
                     fontsize=12, ha='center', va='center')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    output_dir = os.path.join(ROOT_DIR, 'output', 'visualization_plots')
    create_dir(output_dir)
    output_path = os.path.join(output_dir, '06_profile_label_generation.png')
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"  - Visualization plot saved successfully to: {output_path}")

if __name__ == '__main__':
    visualize_profile_label_generation()