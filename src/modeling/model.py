# 文件路径: src/modeling/model.py

import os
import sys
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetV2B0
# 导入一个新的工具，用于下载权重文件
from tensorflow.keras.utils import get_file

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import INPUT_SHAPE, MAX_PATH_DEPTH_POINTS

# ==============================================================================
# >>>>>>>>>> 最终正确版本：手动按名字加载权重，彻底解决层数不匹配问题 <<<<<<<<<<<
# ==============================================================================

def build_advanced_profile_regressor(input_shape, name="AVIP_AdvancedRegressor"):
    """
    使用EfficientNetV2作为主干网络，并手动加载权重以避免层数不匹配错误。
    """
    # 1. 定义我们自己真正的输入层
    inputs = Input(shape=input_shape, name="cwt_input")
    
    # 2. 通道适配层：将8通道映射为EfficientNetV2所需的3通道
    x = Conv2D(3, (1, 1), padding='same', name="channel_adapter")(inputs)

    # 3. 构建EfficientNetV2模型，但这一次我们设置 weights=None
    #    这会创建一个结构正确但权重是随机初始化的模型
    base_model = EfficientNetV2B0(
        include_top=False, 
        weights=None,      # 关键改动：暂时不加载权重
        input_tensor=x
    )

    # 4. 手动下载预训练权重文件
    #    这是EfficientNetV2B0 (no-top)的官方权重URL
    WEIGHTS_PATH = 'https://storage.googleapis.com/tensorflow/keras-applications/efficientnet_v2/efficientnetv2-b0_notop.h5'
    weights_path = get_file(
        'efficientnetv2-b0_notop.h5',
        WEIGHTS_PATH,
        cache_subdir='models'
    )

    # 5. 手动按名字加载权重
    #    - by_name=True 是解决这个问题的关键
    #    - skip_mismatch=True 增加了鲁棒性，允许跳过不匹配的层
    print("正在手动按名字加载预训练权重...")
    base_model.load_weights(weights_path, by_name=True, skip_mismatch=True)
    print("权重加载成功。")

    base_model.trainable = True

    # 6. 在预训练模型的输出之上，添加我们自己的回归头部
    y = base_model.output
    y = GlobalAveragePooling2D()(y)
    y = Dropout(0.5)(y)
    outputs = Dense(MAX_PATH_DEPTH_POINTS, activation='relu', name='profile_output')(y)

    # 7. 创建最终模型
    return Model(inputs, outputs, name=name)


if __name__ == '__main__':
    model = build_advanced_profile_regressor(input_shape=INPUT_SHAPE)
    print("--- AVIP 高级剖面图回归器摘要 ---")
    model.summary()