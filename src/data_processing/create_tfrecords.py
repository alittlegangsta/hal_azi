# File Path: src/data_processing/create_tfrecords.py

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
    ARRAY_ID, PROCESSED_DATA_DIR, GROUND_TRUTH_DB_PATH, DEBUG_MODE, DEBUG_SONIC_DEPTH_POINTS
)

# ==============================================================================
# >>>>>>>>>> Strategic Pivot: New Binary Label Generation <<<<<<<<<<<
# ==============================================================================

# Define the threshold for what constitutes a "significant" amount of channeling.
# If the percentage of channeling pixels is above this, the label is 1.
CHANNELING_PIXEL_THRESHOLD_PERCENT = 1.0 

def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def process_zc_slice_to_label(zc_slice):
    """
    Processes a Zc slice into a single binary label (0 or 1).
    Returns 1 if significant channeling exists, otherwise 0.
    """
    if zc_slice.size == 0:
        return 0.0 # Return float for consistency

    channeling_mask = (zc_slice < 2.5)
    channeling_pixel_count = np.sum(channeling_mask)
    total_pixels = zc_slice.size
    
    channeling_percentage = (channeling_pixel_count / total_pixels) * 100
    
    if channeling_percentage > CHANNELING_PIXEL_THRESHOLD_PERCENT:
        return 1.0
    else:
        return 0.0

def create_tfrecord_example(cwt_image, binary_label):
    """
    Creates a tf.train.Example protobuf for the classification task.
    """
    feature = {
        'feature': _bytes_feature(tf.io.serialize_tensor(cwt_image)),
        'label': _float_feature(binary_label), # Label is now a single float
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))

def create_tfrecords_for_classification():
    """
    Main function to create TFRecord files for the binary classification task.
    """
    print(f"--- Starting TFRecord Generation for Binary Classification ---")
    print(f"Channeling threshold set to: {CHANNELING_PIXEL_THRESHOLD_PERCENT}%")
    
    array_dir = os.path.join(PROCESSED_DATA_DIR, f'array_{str(ARRAY_ID).zfill(2)}')
    cwt_h5_path = os.path.join(array_dir, 'cwt_images.h5')
    processed_waveforms_path = os.path.join(array_dir, 'processed_waveforms.pkl')
    
    output_dir = os.path.join(array_dir, 'tfrecords')
    create_dir(output_dir)
    tfrecord_path = os.path.join(output_dir, 'classification_data.tfrecord')

    mapping_data = load_pickle(processed_waveforms_path)
    sonic_depths = mapping_data['sonic_depths']
    num_samples = len(sonic_depths)

    if DEBUG_MODE:
        num_samples = min(num_samples, DEBUG_SONIC_DEPTH_POINTS)
        sonic_depths = sonic_depths[:num_samples]
        print(f"--- DEBUG MODE: Processing only {num_samples} samples. ---")

    positive_samples = 0
    with h5py.File(cwt_h5_path, 'r') as cwt_hf, \
         h5py.File(GROUND_TRUTH_DB_PATH, 'r') as gt_hf, \
         tf.io.TFRecordWriter(tfrecord_path) as writer:
        
        cwt_dset = cwt_hf['cwt_images']
        path_data_group = gt_hf['path_data']
        
        for i in tqdm(range(num_samples), desc="Generating Classification TFRecords"):
            current_sonic_depth = sonic_depths[i]
            sonic_depth_key = str(current_sonic_depth).replace('.', '_')
            if sonic_depth_key in path_data_group:
                cwt_image = cwt_dset[i].astype(np.float32)
                zc_slice = path_data_group[sonic_depth_key][:]
                
                binary_label = process_zc_slice_to_label(zc_slice)
                if binary_label == 1.0:
                    positive_samples += 1

                example = create_tfrecord_example(cwt_image, binary_label)
                writer.write(example.SerializeToString())

    print("\n--- TFRecord Generation Complete ---")
    print(f"TFRecord file saved at: {tfrecord_path}")
    print(f"Total Samples: {num_samples}")
    print(f"Positive Samples (Has Channeling): {positive_samples} ({positive_samples/num_samples:.2%})")
    print(f"Negative Samples (No Channeling): {num_samples - positive_samples} ({(num_samples - positive_samples)/num_samples:.2%})")


if __name__ == '__main__':
    create_tfrecords_for_classification()