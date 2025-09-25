# 文件路径: src/data_processing/create_tfrecords.py

import os
import sys
import numpy as np
import h5py
import tensorflow as tf
from tqdm import tqdm
import math

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.file_io import create_dir, load_pickle
from config import (
    ARRAY_ID, PROCESSED_DATA_DIR, GROUND_TRUTH_DB_PATH, FFT_COEFFICIENTS,
    MAX_PATH_DEPTH_POINTS, DEBUG_MODE, DEBUG_SONIC_DEPTH_POINTS, CWT_CHUNK_SIZE
)

def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def process_zc_slice_to_label(zc_slice, n_fft_coeffs, max_len):
    """
    将Zc切片处理成双通道的、代表“窜槽”与“良好”分类的FFT标签。
    """
    # ==============================================================================
    # >>>>>>>>>> 代码修改区域 V5.0：“分类”思想与双通道标签 <<<<<<<<<<<
    # ==============================================================================
    # 1. 创建两个二元掩码 (Binary Masks)
    #    - 窜槽掩码: Zc < 2.5 的地方为1, 其他为0
    #    - 良好掩码: Zc >= 2.5 的地方为1, 其他为0
    channeling_mask = (zc_slice < 2.5).astype(np.float32)
    good_bonding_mask = (zc_slice >= 2.5).astype(np.float32)

    # 2. 分别对两个掩码进行FFT并取幅度
    fft_channeling = np.abs(np.fft.fft(channeling_mask, axis=1))
    fft_good_bonding = np.abs(np.fft.fft(good_bonding_mask, axis=1))
    
    # 3. 截取前N个系数
    fft_channeling_coeffs = fft_channeling[:, :n_fft_coeffs].astype(np.float32)
    fft_good_bonding_coeffs = fft_good_bonding[:, :n_fft_coeffs].astype(np.float32)
    
    # 4. 将两个通道的FFT结果堆叠起来
    #    我们不使用对数变换，因为Focal Loss会处理不平衡问题
    stacked_fft_labels = np.stack([fft_channeling_coeffs, fft_good_bonding_coeffs], axis=-1)
    
    # ==============================================================================
    # <<<<<<<<<<<<<<<<<<<<<<<<<< 修改区域结束 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # ==============================================================================
    
    # 5. 填充或截断深度轴以匹配max_len
    current_len = stacked_fft_labels.shape[0]
    # 注意：输出形状现在是 (max_len, n_fft_coeffs, 2)
    padded_label = np.zeros((max_len, n_fft_coeffs, 2), dtype=np.float32)
    
    if current_len > 0:
        if current_len > max_len:
            padded_label = stacked_fft_labels[:max_len, :, :]
        else:
            padded_label[:current_len, :, :] = stacked_fft_labels
            
    return padded_label

def create_tfrecord_example(cwt_image, fft_label):
    feature = {
        'feature': _bytes_feature(tf.io.serialize_tensor(cwt_image)),
        'label': _bytes_feature(tf.io.serialize_tensor(fft_label)),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))

def create_tfrecords_for_translation():
    print("--- Starting AVIP Phase 2b: TFRecord Generation (Binary Classification Labels) ---")
    array_dir = os.path.join(PROCESSED_DATA_DIR, f'array_{str(ARRAY_ID).zfill(2)}')
    cwt_h5_path = os.path.join(array_dir, 'cwt_images.h5')
    processed_waveforms_path = os.path.join(array_dir, 'processed_waveforms.pkl')
    output_dir = os.path.join(array_dir, 'tfrecords')
    create_dir(output_dir)
    tfrecord_path = os.path.join(output_dir, 'translation_data.tfrecord')

    print(f"Loading data from: {cwt_h5_path}, {GROUND_TRUTH_DB_PATH}, {processed_waveforms_path}")
    mapping_data = load_pickle(processed_waveforms_path)
    sonic_depths = mapping_data['sonic_depths']
    num_samples = len(sonic_depths)

    if DEBUG_MODE:
        num_samples = min(num_samples, DEBUG_SONIC_DEPTH_POINTS)
        sonic_depths = sonic_depths[:num_samples]
        print(f"--- DEBUG MODE: Processing only {num_samples} samples. ---")

    with h5py.File(cwt_h5_path, 'r') as cwt_hf, \
         h5py.File(GROUND_TRUTH_DB_PATH, 'r') as gt_hf, \
         tf.io.TFRecordWriter(tfrecord_path) as writer:
        
        cwt_dset = cwt_hf['cwt_images']
        path_data_group = gt_hf['path_data']
        
        print(f"Writing TFRecord file to: {tfrecord_path}")
        for i in tqdm(range(num_samples), desc="Generating TFRecords"):
            current_sonic_depth = sonic_depths[i]
            sonic_depth_key = str(current_sonic_depth).replace('.', '_')
            if sonic_depth_key in path_data_group:
                cwt_image = cwt_dset[i]
                zc_slice = path_data_group[sonic_depth_key][:]
                fft_label = process_zc_slice_to_label(zc_slice, FFT_COEFFICIENTS, MAX_PATH_DEPTH_POINTS)
                example = create_tfrecord_example(cwt_image, fft_label)
                writer.write(example.SerializeToString())
            else:
                print(f"Warning: Sonic depth {current_sonic_depth} not found in Ground-Truth DB. Skipping.")

    print("\n--- TFRecord Generation Complete ---")

if __name__ == '__main__':
    create_tfrecords_for_translation()