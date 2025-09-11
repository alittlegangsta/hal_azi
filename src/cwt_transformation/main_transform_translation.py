import os
import sys
import numpy as np
import h5py
import pywt
from tqdm import tqdm
import math

#  添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.file_io import load_pickle, create_dir
# 注意：config 模块的导入需要根据您的项目结构进行调整
from config import (
    ARRAY_ID, PROCESSED_DATA_DIR, CWT_SCALES, CWT_WAVELET,
    INPUT_SHAPE, DEBUG_MODE, DEBUG_SONIC_DEPTH_POINTS, CWT_CHUNK_SIZE
)

def batch_cwt_transformer(waveforms, scales, wavelet):
    """
    对一个批次的8通道波形并行执行CWT。
    输入 waveforms: (batch_size, 8, time_steps)
    输出 cwt_images: (batch_size, n_scales, time_steps, 8)
    """
    batch_size, n_channels, time_steps = waveforms.shape
    n_scales = len(scales)
    
    # 初始化输出数组
    cwt_images = np.zeros((batch_size, n_scales, time_steps, n_channels), dtype=np.float32)
    
    # 遍历批次中的每个样本
    for i in range(batch_size):
        all_coeffs = []
        for j in range(n_channels): # 遍历8个通道
            coeffs, _ = pywt.cwt(waveforms[i, j, :], scales, wavelet)
            all_coeffs.append(np.abs(coeffs).astype(np.float32))
        
        # 堆叠成 (n_scales, time_steps, n_channels)
        cwt_images[i, :, :, :] = np.stack(all_coeffs, axis=-1)
        
    return cwt_images

def transform_waveforms_to_cwt_images():
    """
    主函数：读取处理后的波形，分块执行CWT，并将结果保存到HDF5文件。
    """
    print("--- Starting AVIP Phase 2a: Chunked CWT Transformation ---")

    # 定义输入和输出路径
    array_dir = os.path.join(PROCESSED_DATA_DIR, f'array_{str(ARRAY_ID).zfill(2)}')
    processed_waveforms_path = os.path.join(array_dir, 'processed_waveforms.pkl')
    output_h5_path = os.path.join(array_dir, 'cwt_images.h5')
    create_dir(array_dir)

    print(f"Loading processed waveforms from: {processed_waveforms_path}")
    data = load_pickle(processed_waveforms_path)
    waveforms = data['waveforms'] # 期望形状 (n_samples, 8, 400)
    
    num_samples = waveforms.shape[0]

    if DEBUG_MODE:
        num_samples = min(num_samples, DEBUG_SONIC_DEPTH_POINTS)
        waveforms = waveforms[:num_samples]
        print(f"--- DEBUG MODE: Processing only {num_samples} samples. ---")

    # 计算分块数量
    num_chunks = math.ceil(num_samples / CWT_CHUNK_SIZE)
    print(f"Total samples: {num_samples}, Chunk size: {CWT_CHUNK_SIZE}, Number of chunks: {num_chunks}")

    # 创建并初始化HDF5文件
    with h5py.File(output_h5_path, 'w') as hf:
        # 创建一个可调整大小的数据集来存储所有CWT图像
        dset = hf.create_dataset(
            'cwt_images',
            shape=(num_samples, INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[2]),
            maxshape=(None, INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[2]),
            dtype='float32',
            chunks=(1, INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[2]) # 为快速查询优化
        )

        for i in tqdm(range(num_chunks), desc="Processing chunks"):
            # 计算当前块的起止索引
            start_idx = i * CWT_CHUNK_SIZE
            end_idx = min((i + 1) * CWT_CHUNK_SIZE, num_samples)
            
            # 获取当前块的波形数据
            waveforms_chunk = waveforms[start_idx:end_idx]
            
            # 对当前块执行CWT变换
            cwt_images_chunk = batch_cwt_transformer(waveforms_chunk, CWT_SCALES, CWT_WAVELET)
            
            # 将变换结果写入HDF5文件
            dset[start_idx:end_idx] = cwt_images_chunk

    print("\n--- Chunked CWT Transformation Complete ---")
    print(f"All CWT images saved to: {output_h5_path}")

if __name__ == '__main__':
    transform_waveforms_to_cwt_images()