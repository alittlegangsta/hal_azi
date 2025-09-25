# 文件路径: src/modeling/model.py

import os
import sys
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, Activation, MaxPooling2D,
    GlobalAveragePooling2D, Dense, Dropout
)
from tensorflow.keras.models import Model

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import INPUT_SHAPE

def build_classifier(input_shape, name="AVIP_Classifier"):
    """
    构建一个标准的CNN二元分类模型。
    """
    inputs = Input(shape=input_shape)

    # Entry block
    x = Conv2D(32, (3, 3), strides=2, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(64, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    # Block 2
    x = MaxPooling2D(3, strides=2, padding="same")(x)
    x = Conv2D(128, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    # Block 3
    x = MaxPooling2D(3, strides=2, padding="same")(x)
    x = Conv2D(256, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    # Global pooling and output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)

    # ==============================================================================
    # >>>>>>>>>> 代码修正区域：为输出层添加sigmoid激活 <<<<<<<<<<<
    # ==============================================================================
    # 将激活函数改为 'sigmoid'，确保输出值在 [0, 1] 范围内
    outputs = Dense(1, activation='sigmoid')(x)
    # ==============================================================================
    # <<<<<<<<<<<<<<<<<<<<<<<<<< 修正区域结束 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # ==============================================================================

    return Model(inputs, outputs, name=name)

if __name__ == '__main__':
    model = build_classifier(INPUT_SHAPE)
    print("--- AVIP 分类器摘要 ---")
    model.summary()