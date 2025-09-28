# 文件路径: src/data_processing/create_tfrecords.py

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
    MAX_PATH_DEPTH_POINTS, DEBUG_MODE, DEBUG_SONIC_DEPTH_POINTS
)

# ==============================================================================
# >>>>>>>>>> 战略转折：新的一维剖面图标签 <<<<<<<<<<<
# ==============================================================================

def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def process_zc_slice_to_label(zc_slice, max_len):
    """
    将Zc切片处理成一个一维的“窜槽百分比”深度剖面图。
    """
    # 1. 计算每个深度点的窜槽百分比
    #    axis=1 表示沿着方位角轴（180个点）进行计算
    channeling_mask = (zc_slice < 2.5)
    channeling_percentage_profile = np.mean(channeling_mask.astype(np.float32), axis=1) * 100.0

    # 2. 填充或截断深度轴以匹配max_len
    current_len = len(channeling_percentage_profile)
    # 最终标签是一个一维向量
    padded_label = np.zeros((max_len,), dtype=np.float32)

    if current_len > 0:
        if current_len > max_len:
            padded_label = channeling_percentage_profile[:max_len]
        else:
            padded_label[:current_len] = channeling_percentage_profile
            
    return padded_label

def create_tfrecord_example(cwt_image, profile_label):
    """
    为回归任务创建tf.train.Example。
    """
    feature = {
        'feature': _bytes_feature(tf.io.serialize_tensor(cwt_image)),
        'label': _bytes_feature(tf.io.serialize_tensor(profile_label)),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))

def create_tfrecords_for_profile_regression():
    """
    为剖面图回归任务创建TFRecord文件的主函数。
    """
    print("--- 开始为一维剖面图回归任务生成TFRecord ---")
    
    array_dir = os.path.join(PROCESSED_DATA_DIR, f'array_{str(ARRAY_ID).zfill(2)}')
    cwt_h5_path = os.path.join(array_dir, 'cwt_images.h5')
    processed_waveforms_path = os.path.join(array_dir, 'processed_waveforms.pkl')
    
    output_dir = os.path.join(array_dir, 'tfrecords')
    create_dir(output_dir)
    tfrecord_path = os.path.join(output_dir, 'profile_regression_data.tfrecord')

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
        
        for i in tqdm(range(num_samples), desc="生成回归TFRecord"):
            current_sonic_depth = sonic_depths[i]
            sonic_depth_key = str(current_sonic_depth).replace('.', '_')
            if sonic_depth_key in path_data_group:
                cwt_image = cwt_dset[i].astype(np.float32)
                zc_slice = path_data_group[sonic_depth_key][:]
                
                profile_label = process_zc_slice_to_label(zc_slice, MAX_PATH_DEPTH_POINTS)
                
                example = create_tfrecord_example(cwt_image, profile_label)
                writer.write(example.SerializeToString())

    print("\n--- TFRecord生成完成 ---")
    print(f"TFRecord文件已保存至: {tfrecord_path}")

if __name__ == '__main__':
    create_tfrecords_for_profile_regression()