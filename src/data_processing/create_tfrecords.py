# 文件路径: src/data_processing/create_tfrecords.py

import os
import sys
import numpy as np
import h5py
import tensorflow as tf
from tqdm import tqdm
import math

#  添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.file_io import create_dir, load_pickle
# 注意：config 模块的导入需要根据您的项目结构进行调整
from config import (
    ARRAY_ID, PROCESSED_DATA_DIR, GROUND_TRUTH_DB_PATH, FFT_COEFFICIENTS,
    MAX_PATH_DEPTH_POINTS, DEBUG_MODE, DEBUG_SONIC_DEPTH_POINTS, CWT_CHUNK_SIZE
)

def _bytes_feature(value):
    """返回一个bytes_list."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def process_zc_slice_to_label(zc_slice, n_fft_coeffs, max_len):
    """
    将Zc切片处理成固定大小的FFT幅度图像标签。
    """
    # 1. 沿方位角轴（180个点）执行FFT
    fft_coeffs = np.fft.fft(zc_slice, axis=1)
    
    # 2. 取绝对值幅度
    fft_magnitudes_raw = np.abs(fft_coeffs)
    
    # ==============================================================================
    # >>>>>>>>>> 代码修改区域 <<<<<<<<<<<
    # ==============================================================================
    # 使用对数变换 log(1 + x) 来压缩数值范围，提升高频（结构）信号的权重。
    # 这是本次优化的核心步骤。
    fft_magnitudes = np.log1p(fft_magnitudes_raw).astype(np.float32)
    # ==============================================================================
    # <<<<<<<<<<<<<<<<<<<<<<<<<< 修改区域结束 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # ==============================================================================

    # 3. 截取前N个低频系数
    label_image = fft_magnitudes[:, :n_fft_coeffs]
    
    # 4. 填充或截断深度轴以匹配max_len
    current_len = label_image.shape[0]
    padded_label = np.zeros((max_len, n_fft_coeffs), dtype=np.float32)
    
    if current_len > 0:
        if current_len > max_len:
            # 截断
            padded_label = label_image[:max_len, :]
        else:
            # 填充 (只填充有效部分)
            padded_label[:current_len, :] = label_image
            
    return padded_label

def create_tfrecord_example(cwt_image, fft_label):
    """
    创建一个tf.train.Example protobuf。
    """
    feature = {
        'feature': _bytes_feature(tf.io.serialize_tensor(cwt_image)),
        'label': _bytes_feature(tf.io.serialize_tensor(fft_label)),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))

def create_tfrecords_for_translation():
    """
    主函数：读取CWT图像和Zc真值，创建用于图像翻译的TFRecord文件。
    """
    print("--- Starting AVIP Phase 2b: TFRecord Generation (with Log Scaling) ---")
    
    # 定义输入路径
    array_dir = os.path.join(PROCESSED_DATA_DIR, f'array_{str(ARRAY_ID).zfill(2)}')
    cwt_h5_path = os.path.join(array_dir, 'cwt_images.h5')
    processed_waveforms_path = os.path.join(array_dir, 'processed_waveforms.pkl')
    
    # 定义输出路径
    output_dir = os.path.join(array_dir, 'tfrecords')
    create_dir(output_dir)
    tfrecord_path = os.path.join(output_dir, 'translation_data.tfrecord')

    print(f"Loading CWT images from: {cwt_h5_path}")
    print(f"Loading Ground-Truth HDF5 DB from: {GROUND_TRUTH_DB_PATH}")
    print(f"Loading sonic depths for mapping from: {processed_waveforms_path}")

    # 加载声波深度信息，用于在两个HDF5文件之间建立映射
    mapping_data = load_pickle(processed_waveforms_path)
    sonic_depths = mapping_data['sonic_depths']

    num_samples = len(sonic_depths)
    if DEBUG_MODE:
        num_samples = min(num_samples, DEBUG_SONIC_DEPTH_POINTS)
        sonic_depths = sonic_depths[:num_samples]
        print(f"--- DEBUG MODE: Processing only {num_samples} samples. ---")

    # 使用上下文管理器确保文件被正确关闭
    with h5py.File(cwt_h5_path, 'r') as cwt_hf, \
         h5py.File(GROUND_TRUTH_DB_PATH, 'r') as gt_hf, \
         tf.io.TFRecordWriter(tfrecord_path) as writer:
        
        cwt_dset = cwt_hf['cwt_images']
        path_data_group = gt_hf['path_data']
        
        print(f"Writing TFRecord file to: {tfrecord_path}")
        
        for i in tqdm(range(num_samples), desc="Generating TFRecords"):
            current_sonic_depth = sonic_depths[i]
            
            # --- 1. 读取预先计算好的CWT图像 ---
            cwt_image = cwt_dset[i]
            
            # --- 2. 读取Zc真值切片并生成标签 ---
            sonic_depth_key = str(current_sonic_depth).replace('.', '_')
            if sonic_depth_key in path_data_group:
                zc_slice = path_data_group[sonic_depth_key][:]
                
                fft_label = process_zc_slice_to_label(
                    zc_slice, FFT_COEFFICIENTS, MAX_PATH_DEPTH_POINTS
                )
                
                # --- 3. 创建并写入TFRecord样本 ---
                example = create_tfrecord_example(cwt_image, fft_label)
                writer.write(example.SerializeToString())
            else:
                print(f"Warning: Sonic depth {current_sonic_depth} not found in Ground-Truth DB. Skipping.")

    print("\n--- TFRecord Generation Complete ---")
    print(f"TFRecord file saved at: {tfrecord_path}")

if __name__ == '__main__':
    create_tfrecords_for_translation()