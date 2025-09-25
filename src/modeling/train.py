# 文件路径: src/modeling/train.py

import os
import sys
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.modeling.model import build_classifier
from src.modeling.dataset import create_dataset
from src.utils.file_io import create_dir, save_pickle
from config import (
    ARRAY_ID, PROCESSED_DATA_DIR, MODEL_DIR, LOG_DIR,
    INPUT_SHAPE, BATCH_SIZE, EPOCHS, LEARNING_RATE
)

def _parse_classification_tfrecord_fn(example_proto):
    feature_description = {
        'feature': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.float32),
    }
    parsed_example = tf.io.parse_single_example(example_proto, feature_description)
    feature_tensor = tf.io.parse_tensor(parsed_example['feature'], out_type=tf.float32)
    feature_tensor = tf.reshape(feature_tensor, INPUT_SHAPE)
    label_tensor = tf.reshape(parsed_example['label'], [1])
    return feature_tensor, label_tensor

def train_classifier_model():
    print("--- 开始为二元分类任务进行训练 ---")

    # --- 1. 准备数据 ---
    tfrecord_dir = os.path.join(PROCESSED_DATA_DIR, f'array_{str(ARRAY_ID).zfill(2)}', 'tfrecords')
    tfrecord_path = os.path.join(tfrecord_dir, 'classification_data.tfrecord')
    print(f"从以下路径加载数据集: {tfrecord_path}")
    
    full_dataset = create_dataset(tfrecord_path, batch_size=BATCH_SIZE, is_training=True,
                                  parse_fn=_parse_classification_tfrecord_fn)

    dataset_size = sum(1 for _ in full_dataset)
    if dataset_size == 0:
        print("\n致命错误：数据集为空！请使用 '--force_run' 重新生成您的数据文件。")
        return

    val_size = max(1, int(0.2 * dataset_size))
    train_size = dataset_size - val_size
    
    full_dataset = create_dataset(tfrecord_path, batch_size=BATCH_SIZE, is_training=True,
                                  parse_fn=_parse_classification_tfrecord_fn)
    train_dataset = full_dataset.take(train_size)
    val_dataset = full_dataset.skip(train_size)
    print(f"数据集分割: {train_size} 个训练批次, {val_size} 个验证批次。")

    # --- 2. 构建模型 ---
    print("构建分类器模型...")
    model = build_classifier(input_shape=INPUT_SHAPE)
    
    # --- 3. 编译模型 ---
    print("使用BinaryCrossentropy损失函数编译模型...")
    # ==============================================================================
    # >>>>>>>>>> 代码修正区域：将 from_logits 设置为 False <<<<<<<<<<<
    # ==============================================================================
    # 因为模型现在直接输出概率，我们需要告诉损失函数 from_logits=False
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    # ==============================================================================
    # <<<<<<<<<<<<<<<<<<<<<<<<<< 修正区域结束 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # ==============================================================================
    model.summary()

    # --- 4. 设置回调函数 ---
    create_dir(MODEL_DIR)
    create_dir(LOG_DIR)
    checkpoint_path = os.path.join(MODEL_DIR, 'best_classifier_model.h5')
    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR),
        tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, monitor='val_auc', save_best_only=True, mode='max', verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor='val_auc', patience=15, mode='max', verbose=1, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6, verbose=1)
    ]

    # --- 5. 开始训练 ---
    print("\n--- 开始模型训练 ---")
    history = model.fit(
        train_dataset,
        epochs=EPOCHS,
        validation_data=val_dataset,
        callbacks=callbacks
    )
    
    history_path = os.path.join(LOG_DIR, 'training_history.pkl')
    save_pickle(history.history, history_path)
    
    print("\n--- 模型训练完成 ---")
    print(f"最佳模型已保存至: {checkpoint_path}")

if __name__ == '__main__':
    train_classifier_model()