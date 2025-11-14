# 文件路径: src/modeling/model.py
# (新分支 'percentage_' 的版本)

import os
import sys
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dense, Dropout, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras.utils import get_file

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import (
    INPUT_SHAPE, MAX_PATH_DEPTH_POINTS, 
    TASK_TYPE, FFT_COEFFICIENTS  # 导入新变量
)

def build_advanced_profile_regressor(input_shape, name="AVIP_AdvancedRegressor"):
    """
    使用EfficientNetV2作为主干网络，并根据TASK_TYPE构建正确的回归头。
    """
    inputs = Input(shape=input_shape, name="cwt_input")
    x = Conv2D(3, (1, 1), padding='same', name="channel_adapter")(inputs)
    
    base_model = EfficientNetV2B0(
        include_top=False, 
        weights=None,
        input_tensor=x
    )

    WEIGHTS_PATH = 'https://storage.googleapis.com/tensorflow/keras-applications/efficientnet_v2/efficientnetv2-b0_notop.h5'
    weights_path = get_file(
        'efficientnetv2-b0_notop.h5',
        WEIGHTS_PATH,
        cache_subdir='models'
    )
    
    print("正在手动按名字加载预训练权重...")
    base_model.load_weights(weights_path, by_name=True, skip_mismatch=True)
    print("权重加载成功。")

    base_model.trainable = True

    y = base_model.output
    y = GlobalAveragePooling2D()(y)
    y = Dropout(0.5)(y)

    # --- 智能识别功能 ---
    if TASK_TYPE == 'fft_regression':
        print(f"构建 FFT 回归头 (输出维度: {MAX_PATH_DEPTH_POINTS} x {FFT_COEFFICIENTS})")
        total_outputs = MAX_PATH_DEPTH_POINTS * FFT_COEFFICIENTS
        y = Dense(total_outputs, activation='relu', name='flat_output')(y)
        # 将扁平向量重塑为 (70, 30) 的二维图像
        outputs = Reshape((MAX_PATH_DEPTH_POINTS, FFT_COEFFICIENTS), name='profile_output_fft')(y)
    else:
        print(f"构建 1D 剖面图回归头 (输出维度: {MAX_PATH_DEPTH_POINTS})")
        outputs = Dense(MAX_PATH_DEPTH_POINTS, activation='relu', name='profile_output')(y)
    # --- 识别结束 ---

    return Model(inputs, outputs, name=name)


if __name__ == '__main__':
    # 打印将要构建的模型类型
    print(f"--- 准备构建模型，当前 TASK_TYPE: {TASK_TYPE} ---")
    model = build_advanced_profile_regressor(input_shape=INPUT_SHAPE)
    print("\n--- 模型摘要 ---")
    model.summary()