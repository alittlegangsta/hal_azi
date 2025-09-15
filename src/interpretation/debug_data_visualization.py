import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.file_io import create_dir, load_pickle
from src.interpretation.run_analysis import get_grad_cam_heatmap # 从主分析脚本导入函数
from src.modeling.model import build_attention_unet
from config import (
    MODEL_DIR, PROCESSED_DATA_DIR, ARRAY_ID, RESULTS_DIR,
    INPUT_SHAPE, FFT_COEFFICIENTS, MAX_PATH_DEPTH_POINTS,
    LAST_CONV_LAYER_NAME
)


def debug_arrays(sample_index=0):
    """
    加载单个样本，生成CWT和Heatmap的数组，并将其直接保存为图片以供检查。
    """
    print("--- Running Data Visualization Debugger ---")

    # --- 1. 加载所需数据 ---
    print("Loading model and data for a single sample...")
    model_path = os.path.join(MODEL_DIR, 'best_a2inet_model.h5')
    model = build_attention_unet(
        input_shape=INPUT_SHAPE,
        output_height=MAX_PATH_DEPTH_POINTS,
        output_width=FFT_COEFFICIENTS
    )
    model.load_weights(model_path)

    array_dir = os.path.join(PROCESSED_DATA_DIR, f'array_{str(ARRAY_ID).zfill(2)}')
    cwt_h5_path = os.path.join(array_dir, 'cwt_images.h5')
    
    with h5py.File(cwt_h5_path, 'r') as hf:
        input_cwt = hf['cwt_images'][sample_index]

    # --- 2. 生成热力图 ---
    print("Generating Grad-CAM heatmap...")
    heatmap = get_grad_cam_heatmap(model, input_cwt, LAST_CONV_LAYER_NAME)

    # --- 3. 打印调试信息 ---
    print(f"\nShape of Input CWT (channel A): {input_cwt[:, :, 0].shape}")
    print(f"Shape of Heatmap array: {heatmap.shape}")
    
    # --- 4. 将原始数组保存为独立的图片 ---
    output_dir = os.path.join(RESULTS_DIR, 'debug_plots')
    create_dir(output_dir)
    print(f"Saving debug plots to: {output_dir}")

    # 保存CWT图
    plt.imsave(os.path.join(output_dir, 'debug_cwt_array.png'), input_cwt[:, :, 0], cmap='jet')
    # 保存Heatmap图
    plt.imsave(os.path.join(output_dir, 'debug_heatmap_array.png'), heatmap, cmap='hot')

    print("\n--- Debugging complete ---")
    print("Please check the images in the 'debug_plots' directory.")
    print("If these two images have the same dimensions, the data is correct, and the problem is purely in the plotting layout.")

if __name__ == '__main__':
    import h5py
    debug_arrays()