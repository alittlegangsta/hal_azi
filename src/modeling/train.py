# 文件路径: src/modeling/train.py

import os
import sys
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau
import math
import numpy as np # 导入numpy用于创建权重

#  添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.modeling.model import build_attention_unet
from src.modeling.dataset import create_dataset
from src.utils.file_io import create_dir, save_pickle
from config import (
    ARRAY_ID, PROCESSED_DATA_DIR, MODEL_DIR, LOG_DIR,
    INPUT_SHAPE, FFT_COEFFICIENTS, MAX_PATH_DEPTH_POINTS,
    BATCH_SIZE, EPOCHS, LEARNING_RATE, VALIDATION_SPLIT
)


def gradient_difference_loss(y_true, y_pred):
    # 为 image_gradients 确保输入是4D的
    y_true_4d = tf.expand_dims(y_true, axis=-1)
    y_pred_4d = tf.expand_dims(y_pred, axis=-1)

    true_dy, true_dx = tf.image.image_gradients(y_true_4d)
    pred_dy, pred_dx = tf.image.image_gradients(y_pred_4d)
    
    grad_diff_dx = tf.abs(true_dx - pred_dx)
    grad_diff_dy = tf.abs(true_dy - pred_dy)
    
    return tf.reduce_mean(grad_diff_dx) + tf.reduce_mean(grad_diff_dy)


# ==============================================================================
# >>>>>>>>>> 代码修改区域 V2.0：实现频率加权损失 <<<<<<<<<<<
# ==============================================================================

# --- 步骤1：定义频率权重 ---
# 创建一个权重向量，其长度等于FFT系数的数量 (FFT_COEFFICIENTS)
# 我们使用np.linspace来创建一个从1到100线性增加的权重。
# 这意味着：
# - 第0个FFT系数（代表平均值）的误差权重为1。
# - 最后一个高频系数的误差权重为100。
# 这样可以极大地放大模型在高频结构上所犯错误的惩罚。
# 你可以根据实验效果调整这个范围，比如 np.linspace(1, 200, ...)
frequency_weights = tf.constant(
    np.linspace(1, 100, FFT_COEFFICIENTS), dtype=tf.float32
)

def weighted_mse_loss(y_true, y_pred):
    """计算加权的均方误差"""
    # 计算逐元素的平方误差
    squared_error = tf.square(y_true - y_pred)
    
    # 将平方误差与我们的频率权重相乘
    # frequency_weights 会被自动广播到 squared_error 的形状
    weighted_squared_error = squared_error * frequency_weights
    
    # 返回加权误差的平均值
    return tf.reduce_mean(weighted_squared_error)

def hybrid_loss(y_true, y_pred, alpha=0.5):
    """
    结合了加权MSE和梯度差异损失的新混合损失函数。
    """
    # --- 步骤2：使用新的加权MSE替换旧的MSE ---
    w_mse_loss = weighted_mse_loss(y_true, y_pred)
    grad_loss = gradient_difference_loss(y_true, y_pred)
    
    # 保持混合权重不变，但现在MSE部分对结构的惩罚已经大大增强
    return (1 - alpha) * w_mse_loss + alpha * grad_loss

# ==============================================================================
# <<<<<<<<<<<<<<<<<<<<<<<<<< 修改区域结束 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# ==============================================================================


def train_translation_model():
    """主函数：训练 A²INet 模型"""
    print("--- Starting Phase 3: Model Training for AVIP (with Frequency-Weighted Loss) ---")
    
    # --- 1. 准备数据 ---
    tfrecord_dir = os.path.join(PROCESSED_DATA_DIR, f'array_{str(ARRAY_ID).zfill(2)}', 'tfrecords')
    tfrecord_path = os.path.join(tfrecord_dir, 'translation_data.tfrecord')
    
    print(f"Loading dataset from: {tfrecord_path}")
    full_dataset = create_dataset(tfrecord_path, batch_size=BATCH_SIZE, is_training=True)
    
    dataset_size = sum(1 for _ in full_dataset)
    
    print(f"Total batches found: {dataset_size}")
    if dataset_size < 2:
        print("Dataset is too small to split. Training without validation.")
        train_dataset = full_dataset
        val_dataset = None
        train_size = dataset_size
        val_size = 0
    else:
        val_size = max(1, int(VALIDATION_SPLIT * dataset_size))
        train_size = dataset_size - val_size
        if train_size == 0:
            train_size = 1
            val_size = dataset_size - 1
        train_dataset = full_dataset.take(train_size)
        val_dataset = full_dataset.skip(train_size)
    
    print(f"Dataset split: {train_size} training batches, {val_size} validation batches.")

    # --- 2. 构建模型 ---
    print("Building A²INet model...")
    model = build_attention_unet(
        input_shape=INPUT_SHAPE,
        output_height=MAX_PATH_DEPTH_POINTS,
        output_width=FFT_COEFFICIENTS
    )
    
    # --- 3. 编译模型 ---
    print("Compiling model with NEW hybrid loss and gradient clipping...")
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, clipvalue=1.0)
    # --- 关键改动：确保编译时使用的是我们新定义的 hybrid_loss ---
    model.compile(optimizer=optimizer, loss=hybrid_loss, metrics=['mae'])
    model.summary()

    # --- 4. 设置回调函数 ---
    create_dir(MODEL_DIR)
    create_dir(LOG_DIR)
    
    checkpoint_path = os.path.join(MODEL_DIR, 'best_a2inet_model.h5')
    callbacks = [TensorBoard(log_dir=LOG_DIR, histogram_freq=1)]
    
    if val_dataset:
        callbacks.extend([
            ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss', save_best_only=True, mode='min', verbose=1),
            EarlyStopping(monitor='val_loss', patience=15, mode='min', verbose=1, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6, verbose=1)
        ])

    # --- 5. 开始训练 ---
    print("\n--- Starting Model Training ---")
    history = model.fit(
        train_dataset,
        epochs=EPOCHS,
        validation_data=val_dataset,
        callbacks=callbacks
    )
    
    print("Saving final model state...")
    model.save(checkpoint_path)

    # --- 6. 保存训练历史 ---
    history_path = os.path.join(LOG_DIR, 'training_history.pkl')
    save_pickle(history.history, history_path)
    
    print("\n--- Model Training Complete ---")
    print(f"Best model saved at: {checkpoint_path}")
    print(f"Training history saved at: {history_path}")

if __name__ == '__main__':
    train_translation_model()