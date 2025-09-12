import os
import sys
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate,
    BatchNormalization, Activation, Conv2DTranspose, Multiply, Add, Cropping2D,
    Permute, Dense, Lambda
)
from tensorflow.keras.models import Model

#  Add project root to system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import (
    INPUT_SHAPE, FFT_COEFFICIENTS, MAX_PATH_DEPTH_POINTS, LAST_CONV_LAYER_NAME
)

def attention_gate(g, x, inter_channels):
    """
    Attention Gate module.
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

def build_attention_unet(input_shape, output_height, output_width, name="A2INet"):
    """
    Builds the Attention-Augmented U-Net (A²INet).
    """
    inputs = Input(input_shape)

    # --- Encoder ---
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

    # --- Decoder ---
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
    conv5 = Conv2D(32, 3, padding='same', activation='relu')(conv5)
    conv5 = BatchNormalization()(conv5)

    # *** FINAL, COMPATIBLE OUTPUT HEAD ***
    final_conv_head = Conv2D(output_width, (1, 1), activation='relu', padding='same', name=LAST_CONV_LAYER_NAME)(conv5)
    
    permuted = Permute((1, 3, 2))(final_conv_head)
    
    dense_on_width = Dense(1, activation='linear')(permuted)
    
    # *** BUG FIX STARTS HERE ***
    # Use a Lambda layer to wrap the tf.squeeze function for compatibility
    squeezed = Lambda(lambda x: tf.squeeze(x, axis=-1))(dense_on_width)
    # *** BUG FIX ENDS HERE ***

    permuted_for_height = Permute((2, 1))(squeezed)

    final_dense_height = Dense(output_height, activation='linear')(permuted_for_height)

    final_output = Permute((2, 1), name="final_output")(final_dense_height)
    
    model = Model(inputs=inputs, outputs=final_output, name="A2INet_v4")
    return model


if __name__ == '__main__':
    a2inet_model = build_attention_unet(
        input_shape=INPUT_SHAPE,
        output_height=MAX_PATH_DEPTH_POINTS,
        output_width=FFT_COEFFICIENTS
    )
    a2inet_model.summary()
    print(f"\nGrad-CAM will target this layer: '{LAST_CONV_LAYER_NAME}'")