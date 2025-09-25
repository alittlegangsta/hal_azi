# File Path: src/modeling/dataset.py
import os
import tensorflow as tf

def create_dataset(tfrecord_path, batch_size, shuffle_buffer_size=1024, is_training=True, parse_fn=None):
    """
    Creates an efficient tf.data.Dataset from a TFRecord file.
    Can accept a custom parsing function.
    """
    if not os.path.exists(tfrecord_path):
        raise FileNotFoundError(f"TFRecord file not found at: {tfrecord_path}")

    dataset = tf.data.TFRecordDataset(tfrecord_path)
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    
    if not parse_fn:
        raise ValueError("A parsing function (`parse_fn`) must be provided.")

    dataset = dataset.map(parse_fn, num_parallel_calls=AUTOTUNE)
    
    if is_training:
        dataset = dataset.cache()
        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    
    return dataset