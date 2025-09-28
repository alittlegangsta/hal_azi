# 文件路径: src/modeling/train.py

import os
import sys
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# --- 导入升级后的模型 ---
from src.modeling.model import build_advanced_profile_regressor
from src.modeling.dataset import create_dataset
from src.utils.file_io import create_dir, save_pickle
from config import (
    ARRAY_ID, PROCESSED_DATA_DIR, MODEL_DIR, LOG_DIR,
    INPUT_SHAPE, MAX_PATH_DEPTH_POINTS, BATCH_SIZE, EPOCHS, LEARNING_RATE
)

# ==============================================================================
# >>>>>>>>>> 代码升级：引入数据增强与更优的损失函数 <<<<<<<<<<<
# ==============================================================================

def augment_cwt(image, label):
    """
    对CWT图像进行简单的数据增强。
    """
    # 1. 添加高斯噪声
    noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=0.01, dtype=tf.float32)
    image = image + noise
    
    # 2. 频率遮挡 (Frequency Masking)
    #    随机遮挡掉一部分频率通道
    freq_mask_size = tf.random.uniform(shape=[], minval=5, maxval=20, dtype=tf.int32)
    freq_mask_pos = tf.random.uniform(shape=[], minval=0, maxval=INPUT_SHAPE[0] - freq_mask_size, dtype=tf.int32)
    
    mask_start = freq_mask_pos
    mask_end = freq_mask_pos + freq_mask_size
    
    # 创建一个遮挡用的张量
    mask_part1 = tf.ones([mask_start, INPUT_SHAPE[1], INPUT_SHAPE[2]], dtype=tf.float32)
    mask_part2 = tf.zeros([mask_end - mask_start, INPUT_SHAPE[1], INPUT_SHAPE[2]], dtype=tf.float32)
    mask_part3 = tf.ones([INPUT_SHAPE[0] - mask_end, INPUT_SHAPE[1], INPUT_SHAPE[2]], dtype=tf.float32)
    
    freq_mask = tf.concat([mask_part1, mask_part2, mask_part3], axis=0)
    image = image * freq_mask

    return image, label

def _parse_regression_tfrecord_fn(example_proto):
    feature_description = {
        'feature': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.string),
    }
    parsed_example = tf.io.parse_single_example(example_proto, feature_description)
    feature_tensor = tf.io.parse_tensor(parsed_example['feature'], out_type=tf.float32)
    label_tensor = tf.io.parse_tensor(parsed_example['label'], out_type=tf.float32)
    
    feature_tensor = tf.reshape(feature_tensor, INPUT_SHAPE)
    label_tensor = tf.reshape(label_tensor, [MAX_PATH_DEPTH_POINTS])
    return feature_tensor, label_tensor

def train_regressor_model():
    print("--- 开始为一维剖面图回归任务进行高级训练 ---")

    tfrecord_dir = os.path.join(PROCESSED_DATA_DIR, f'array_{str(ARRAY_ID).zfill(2)}', 'tfrecords')
    tfrecord_path = os.path.join(tfrecord_dir, 'profile_regression_data.tfrecord')
    print(f"从以下路径加载数据集: {tfrecord_path}")

    full_dataset = create_dataset(tfrecord_path, batch_size=BATCH_SIZE, is_training=True,
                                  parse_fn=_parse_regression_tfrecord_fn)
    
    dataset_size = sum(1 for _ in full_dataset)
    if dataset_size == 0:
        return

    val_size = max(1, int(0.2 * dataset_size))
    train_size = dataset_size - val_size
    
    # 重新加载数据集以进行分割和增强
    full_dataset = create_dataset(tfrecord_path, batch_size=BATCH_SIZE, is_training=True,
                                  parse_fn=_parse_regression_tfrecord_fn)
    train_dataset = full_dataset.take(train_size)
    val_dataset = full_dataset.skip(train_size)
    
    # --- 在训练集上应用数据增强 ---
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    train_dataset = train_dataset.map(augment_cwt, num_parallel_calls=AUTOTUNE)

    print(f"数据集分割: {train_size} 个训练批次, {val_size} 个验证批次。 (已启用数据增强)")

    print("构建高级回归器模型...")
    model = build_advanced_profile_regressor(input_shape=INPUT_SHAPE)
    
    print("使用Huber损失函数编译模型...")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=tf.keras.losses.Huber(), # 使用对异常值更鲁棒的Huber损失
        metrics=['mae']
    )
    model.summary()

    create_dir(MODEL_DIR); create_dir(LOG_DIR)
    checkpoint_path = os.path.join(MODEL_DIR, 'best_advanced_regressor_model.h5') # 新的模型名
    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR),
        tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss', save_best_only=True, mode='min', verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, mode='min', verbose=1, restore_best_weights=True), # 增加一点耐心
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=7, min_lr=1e-6, verbose=1)
    ]

    print("\n--- 开始模型训练 ---")
    history = model.fit(
        train_dataset,
        epochs=EPOCHS,
        validation_data=val_dataset,
        callbacks=callbacks
    )
    
    history_path = os.path.join(LOG_DIR, 'training_history_advanced.pkl')
    save_pickle(history.history, history_path)
    
    print("\n--- 模型训练完成 ---")
    print(f"最佳模型已保存至: {checkpoint_path}")

if __name__ == '__main__':
    train_regressor_model()