import os
import sys
import tensorflow as tf

#  添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 注意：config 模块的导入需要根据您的项目结构进行调整
from config import (
    INPUT_SHAPE, FFT_COEFFICIENTS, MAX_PATH_DEPTH_POINTS, BATCH_SIZE
)

def _parse_translation_tfrecord_fn(example_proto):
    """
    为图像翻译任务解析一个tf.train.Example。
    """
    # 定义特征描述
    feature_description = {
        'feature': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.string),
    }
    
    # 解析一个样本
    parsed_example = tf.io.parse_single_example(example_proto, feature_description)
    
    # 反序列化张量
    feature_tensor = tf.io.parse_tensor(parsed_example['feature'], out_type=tf.float32)
    label_tensor = tf.io.parse_tensor(parsed_example['label'], out_type=tf.float32)
    
    # 设置正确的形状
    feature_tensor = tf.reshape(feature_tensor, INPUT_SHAPE)
    label_tensor = tf.reshape(label_tensor, [MAX_PATH_DEPTH_POINTS, FFT_COEFFICIENTS])
    
    return feature_tensor, label_tensor

def create_dataset(tfrecord_path, batch_size, shuffle_buffer_size=1024, is_training=True):
    """
    从TFRecord文件创建一个高效的tf.data.Dataset。
    """
    if not os.path.exists(tfrecord_path):
        raise FileNotFoundError(f"TFRecord file not found at: {tfrecord_path}")

    dataset = tf.data.TFRecordDataset(tfrecord_path)
    
    # 使用AUTOTUNE动态调整并行处理的线程数
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    
    # 应用解析函数
    dataset = dataset.map(_parse_translation_tfrecord_fn, num_parallel_calls=AUTOTUNE)
    
    if is_training:
        # 缓存数据以提高性能
        dataset = dataset.cache()
        # 乱序数据
        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
    
    # 批处理
    dataset = dataset.batch(batch_size)
    
    # 预取数据，让GPU在处理当前批次时，CPU可以准备下一批次
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    
    return dataset

if __name__ == '__main__':
    # 这是一个用于测试数据集管道的示例
    # 假设的TFRecord路径，请替换为您的实际路径
    from config import PROCESSED_DATA_DIR, ARRAY_ID
    
    print("--- Testing AVIP Dataset Pipeline ---")
    
    tfrecord_dir = os.path.join(PROCESSED_DATA_DIR, f'array_{str(ARRAY_ID).zfill(2)}', 'tfrecords')
    test_tfrecord_path = os.path.join(tfrecord_dir, 'translation_data.tfrecord')

    if os.path.exists(test_tfrecord_path):
        print(f"Attempting to load data from: {test_tfrecord_path}")
        
        # 创建数据集
        train_dataset = create_dataset(test_tfrecord_path, batch_size=BATCH_SIZE)
        
        # 从数据集中取一个批次进行检查
        for features, labels in train_dataset.take(1):
            print("\n--- Batch Loaded Successfully ---")
            print(f"Features batch shape: {features.shape}")
            print(f"Labels batch shape: {labels.shape}")
            print(f"Features batch dtype: {features.dtype}")
            print(f"Labels batch dtype: {labels.dtype}")
            print("---------------------------------")
        
        print("Dataset pipeline test complete.")
    else:
        print(f"Test TFRecord file not found at '{test_tfrecord_path}'.")
        print("Please run the 'create_tfrecords.py' script first.")