# 文件路径: src/modeling/model.py

import os
import sys
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate,
    BatchNormalization, Activation, Conv2DTranspose, Multiply, Add, Cropping2D,
    Permute, Dense, Lambda, LeakyReLU, Flatten, Dropout
)
from tensorflow.keras.models import Model

#  添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import (
    INPUT_SHAPE, FFT_COEFFICIENTS, MAX_PATH_DEPTH_POINTS, LAST_CONV_LAYER_NAME
)

# ==============================================================================
# --- 生成器 (Generator) ---
# 您原来的 A²INet 模型现在作为生成器。我们保持它不变。
# ==============================================================================

def attention_gate(g, x, inter_channels):
    """
    注意力门模块
    """
    g = Conv2D(inter_channels, kernel_size=1, strides=1, padding='same')(g)
    g = BatchNormalization()(g)
    x = Conv2D(inter_channels, kernel_size=1, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    gx = Add()([g, x])
    gx = Activation('relu')(gx)
    psi = Conv2D(1, kernel_size=1, strides=1, padding='same')(gx)
    psi = BatchNormalization()(psi)
    alpha = Activation('sigmoid')(psi)
    f = Multiply()([x, alpha])
    return f

def build_attention_unet(input_shape, output_height, output_width, name="Generator_A2INet"):
    """
    构建注意力增强型U-Net (A²INet)，作为GAN的生成器。
    """
    inputs = Input(input_shape)

    # --- 编码器 (Encoder) ---
    conv1 = Conv2D(32, 3, padding='same', activation='relu')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(32, 3, padding='same', activation='relu')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, 3, padding='same', activation='relu')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(64, 3, padding='same', activation='relu')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, 3, padding='same', activation='relu')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(128, 3, padding='same', activation='relu')(conv3)
    conv3 = BatchNormalization()(conv3)

    # --- 解码器 (Decoder) ---
    up4 = Conv2DTranspose(64, kernel_size=2, strides=2, padding='same')(conv3)
    ch, cw = up4.shape[1], up4.shape[2]
    crop_conv2 = Cropping2D(cropping=((0, conv2.shape[1] - ch), (0, conv2.shape[2] - cw)))(conv2)
    attn2 = attention_gate(g=up4, x=crop_conv2, inter_channels=32)
    merge4 = concatenate([attn2, up4], axis=3)
    conv4 = Conv2D(64, 3, padding='same', activation='relu')(merge4)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(64, 3, padding='same', activation='relu')(conv4)
    conv4 = BatchNormalization()(conv4)

    up5 = Conv2DTranspose(32, kernel_size=2, strides=2, padding='same')(conv4)
    ch, cw = up5.shape[1], up5.shape[2]
    crop_conv1 = Cropping2D(cropping=((0, conv1.shape[1] - ch), (0, conv1.shape[2] - cw)))(conv1)
    attn1 = attention_gate(g=up5, x=crop_conv1, inter_channels=16)
    merge5 = concatenate([attn1, up5], axis=3)
    conv5 = Conv2D(32, 3, padding='same', activation='relu')(merge5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(32, 3, padding='same', activation='relu', name='last_spatial_conv')(conv5)
    conv5 = BatchNormalization()(conv5)

    final_conv_head = Conv2D(output_width, (1, 1), activation='linear', padding='same')(conv5)
    reshaped_for_dense = Permute((1, 3, 2))(final_conv_head)
    dense_on_width = Dense(1, activation='linear')(reshaped_for_dense)
    squeezed = Lambda(lambda x: tf.squeeze(x, axis=-1))(dense_on_width)
    permuted_for_height = Permute((2, 1))(squeezed)
    final_dense_height = Dense(output_height, activation='linear')(permuted_for_height)
    final_output = Permute((2, 1), name="final_output")(final_dense_height)
    
    model = Model(inputs=inputs, outputs=final_output, name=name)
    return model

# ==============================================================================
# >>>>>>>>>> 新增代码：判别器 (Discriminator) <<<<<<<<<<<
# ==============================================================================
def build_discriminator(input_height, input_width, name="Discriminator"):
    """
    构建一个判别器模型 (PatchGAN a-like)。
    它接收一个图像作为输入，并输出一个概率值，判断该图像是真实的还是伪造的。
    """
    # 输入层：FFT标签图像
    inp = Input(shape=[input_height, input_width, 1])

    # 下采样模块
    down1 = Conv2D(64, 4, strides=2, padding='same')(inp)
    down1 = LeakyReLU()(down1)

    down2 = Conv2D(128, 4, strides=2, padding='same')(down1)
    down2 = BatchNormalization()(down2)
    down2 = LeakyReLU()(down2)

    down3 = Conv2D(256, 4, strides=2, padding='same')(down2)
    down3 = BatchNormalization()(down3)
    down3 = LeakyReLU()(down3)

    # 最后一个卷积层，输出一个单通道的概率图
    last = Conv2D(1, 4, strides=1, padding='same')(down3)

    return Model(inputs=inp, outputs=last, name=name)

if __name__ == '__main__':
    generator = build_attention_unet(
        input_shape=INPUT_SHAPE,
        output_height=MAX_PATH_DEPTH_POINTS,
        output_width=FFT_COEFFICIENTS
    )
    print("--- Generator Summary ---")
    generator.summary()

    discriminator = build_discriminator(
        input_height=MAX_PATH_DEPTH_POINTS,
        input_width=FFT_COEFFICIENTS
    )
    print("\n--- Discriminator Summary ---")
    discriminator.summary()