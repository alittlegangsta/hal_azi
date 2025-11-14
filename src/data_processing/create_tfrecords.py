# 文件路径: src/data_processing/create_tfrecords.py
# (已应用“窜槽严重性图”方案的最终版本)

import os
import sys
import numpy as np
import h5py
import tensorflow as tf
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.file_io import create_dir, load_pickle
from config import (
    ARRAY_ID, PROCESSED_DATA_DIR, GROUND_TRUTH_DB_PATH, 
    MAX_PATH_DEPTH_POINTS, DEBUG_MODE, DEBUG_SONIC_DEPTH_POINTS,
    TASK_TYPE, FFT_COEFFICIENTS
)

def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _process_zc_slice_to_1d_label(zc_slice, max_len):
    """
    (旧逻辑) 将Zc切片处理成一个一维的“窜槽百分比”深度剖面图。
    """
    channeling_mask = (zc_slice < 2.5)
    channeling_percentage_profile = np.mean(channeling_mask.astype(np.float32), axis=1) * 100.0
    padded_label = np.zeros((max_len,), dtype=np.float32)
    current_len = len(channeling_percentage_profile)
    if current_len > 0:
        if current_len > max_len:
            padded_label = channeling_percentage_profile[:max_len]
        else:
            padded_label[:current_len] = channeling_percentage_profile
    return padded_label

def _process_zc_slice_to_fft_label(zc_slice, max_len):
    """
    (新逻辑 - 已修正) 根据您的要求，使用“窜槽严重性图” (max(0, 2.5 - Zc))。
    """
    num_depth_points = zc_slice.shape[0]
    fft_label_image = np.zeros((max_len, FFT_COEFFICIENTS), dtype=np.float32)
    
    valid_len = min(num_depth_points, max_len)
    if valid_len == 0:
        return fft_label_image
        
    for i in range(valid_len):
        
        # --- 核心修正：应用您的“窜槽严重性”变换 ---
        # 1. 计算 "Severity(Zc) = max(0, 2.5 - Zc)"
        #    这将把所有 > 2.5 的“噪声”像素归零。
        severity_map = np.maximum(0, 2.5 - zc_slice[i, :])
        
        # 2. 对这个信噪比极高的“严重性图”进行FFT
        fft_result = np.fft.fft(severity_map)
        # --- 修正结束 ---
        
        # 3. 取绝对幅度
        magnitude = np.abs(fft_result)
        
        # 4. 截取前N个低频系数并进行对数变换 (借鉴 改进memo.md)
        coefficients = magnitude[:FFT_COEFFICIENTS]
        log_coefficients = np.log(1 + coefficients)
        
        fft_label_image[i, :] = log_coefficients.astype(np.float32)
        
    return fft_label_image

def create_tfrecord_example(cwt_image, profile_label):
    feature = {
        'feature': _bytes_feature(tf.io.serialize_tensor(cwt_image)),
        'label': _bytes_feature(tf.io.serialize_tensor(profile_label)),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))

def create_tfrecords_for_regression():
    """
    主函数，根据 TASK_TYPE 自动生成正确的TFRecord文件。
    """
    
    if TASK_TYPE == 'fft_regression':
        print("--- 开始为 FFT 回归任务生成 TFRecord (使用“严重性图”逻辑) ---")
        process_func = _process_zc_slice_to_fft_label
        tfrecord_filename = 'fft_regression_data.tfrecord'
    else:
        print("--- 开始为一维剖面图回归任务生成 TFRecord ---")
        process_func = _process_zc_slice_to_1d_label
        tfrecord_filename = 'profile_regression_data.tfrecord'
    
    array_dir = os.path.join(PROCESSED_DATA_DIR, f'array_{str(ARRAY_ID).zfill(2)}')
    cwt_h5_path = os.path.join(array_dir, 'cwt_images.h5')
    processed_waveforms_path = os.path.join(array_dir, 'processed_waveforms.pkl')
    
    output_dir = os.path.join(array_dir, 'tfrecords')
    create_dir(output_dir)
    tfrecord_path = os.path.join(output_dir, tfrecord_filename)

    mapping_data = load_pickle(processed_waveforms_path)
    sonic_depths = mapping_data['sonic_depths']
    num_samples = len(sonic_depths)

    if DEBUG_MODE:
        num_samples = min(num_samples, DEBUG_SONIC_DEPTH_POINTS)
        sonic_depths = sonic_depths[:num_samples]

    with h5py.File(cwt_h5_path, 'r') as cwt_hf, \
         h5py.File(GROUND_TRUTH_DB_PATH, 'r') as gt_hf, \
         tf.io.TFRecordWriter(tfrecord_path) as writer:
        
        cwt_dset = cwt_hf['cwt_images']
        path_data_group = gt_hf['path_data']
        
        for i in tqdm(range(num_samples), desc=f"生成 {tfrecord_filename}"):
            current_sonic_depth = sonic_depths[i]
            sonic_depth_key = str(current_sonic_depth).replace('.', '_')
            if sonic_depth_key in path_data_group:
                cwt_image = cwt_dset[i].astype(np.float32)
                zc_slice = path_data_group[sonic_depth_key][:]
                
                profile_label = process_func(zc_slice, MAX_PATH_DEPTH_POINTS)
                
                example = create_tfrecord_example(cwt_image, profile_label)
                writer.write(example.SerializeToString())

    print("\n--- TFRecord生成完成 ---")
    print(f"TFRecord文件已保存至: {tfrecord_path}")

if __name__ == '__main__':
    create_tfrecords_for_regression()