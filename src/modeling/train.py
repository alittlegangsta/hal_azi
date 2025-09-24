# 文件路径: src/modeling/train.py

import os
import sys
import time # 导入time模块用于性能监控
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
import numpy as np

#  添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# --- 关键改动：从 model.py 中同时导入生成器和判别器 ---
from src.modeling.model import build_attention_unet, build_discriminator
from src.modeling.dataset import create_dataset
from src.utils.file_io import create_dir, save_pickle
from config import (
    ARRAY_ID, PROCESSED_DATA_DIR, MODEL_DIR, LOG_DIR,
    INPUT_SHAPE, FFT_COEFFICIENTS, MAX_PATH_DEPTH_POINTS,
    BATCH_SIZE, EPOCHS, LEARNING_RATE, VALIDATION_SPLIT
)


# ==============================================================================
# >>>>>>>>>> 代码修改区域 V3.0：实现GAN训练流程 <<<<<<<<<<<
# ==============================================================================

# --- 步骤1：定义损失函数 ---

# 使用BinaryCrossentropy来计算对抗损失
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
# from_logits=True 表示模型的输出没有经过sigmoid激活

# LAMBDA 用于控制 L1 损失在总生成器损失中的权重
# 这是一个非常重要的超参数，可以根据效果进行调整
LAMBDA = 100 

def discriminator_loss(real_output, fake_output):
    """判别器损失：鼓励判别器将真实图像识别为1，伪造图像识别为0"""
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(discriminator_fake_output, generated_image, target_image):
    """生成器损失：由两部分组成"""
    # 1. 对抗损失：鼓励生成器生成能被判别器识别为1的图像
    gan_loss = cross_entropy(tf.ones_like(discriminator_fake_output), discriminator_fake_output)
    
    # 2. 内容损失 (L1 Loss)：鼓励生成的图像在像素上接近真实图像
    #    这确保了生成器不会为了欺骗判别器而生成完全不相关的图像
    l1_loss = tf.reduce_mean(tf.abs(target_image - generated_image))
    
    total_gen_loss = gan_loss + (LAMBDA * l1_loss)
    return total_gen_loss, gan_loss, l1_loss


# --- 步骤2：定义训练主函数 ---

def train_gan_model():
    """主函数：训练 AVIP-GAN 模型"""
    print("--- Starting Phase 3: Model Training for AVIP (GAN Implementation) ---")

    # --- 准备数据 ---
    tfrecord_dir = os.path.join(PROCESSED_DATA_DIR, f'array_{str(ARRAY_ID).zfill(2)}', 'tfrecords')
    tfrecord_path = os.path.join(tfrecord_dir, 'translation_data.tfrecord')
    print(f"Loading dataset from: {tfrecord_path}")
    full_dataset = create_dataset(tfrecord_path, batch_size=BATCH_SIZE, is_training=True)
    
    # --- 构建模型和优化器 ---
    print("Building Generator and Discriminator...")
    generator = build_attention_unet(
        input_shape=INPUT_SHAPE,
        output_height=MAX_PATH_DEPTH_POINTS,
        output_width=FFT_COEFFICIENTS
    )
    discriminator = build_discriminator(
        input_height=MAX_PATH_DEPTH_POINTS,
        input_width=FFT_COEFFICIENTS
    )

    generator_optimizer = tf.keras.optimizers.Adam(LEARNING_RATE, beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(LEARNING_RATE, beta_1=0.5)

    # --- 设置检查点，用于保存模型 ---
    create_dir(MODEL_DIR)
    checkpoint_prefix = os.path.join(MODEL_DIR, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer,
                                     generator=generator,
                                     discriminator=discriminator)

    # --- 自定义训练步骤 ---
    @tf.function
    def train_step(input_image, target, epoch):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # 1. 生成器生成伪造图像
            generated_image = generator(input_image, training=True)

            # 2. 判别器对真实图像和伪造图像进行判断
            real_output = discriminator(target, training=True)
            fake_output = discriminator(generated_image, training=True)

            # 3. 计算生成器和判别器的损失
            gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(fake_output, generated_image, target)
            disc_loss = discriminator_loss(real_output, fake_output)

        # 4. 计算梯度并应用到优化器
        generator_gradients = gen_tape.gradient(gen_total_loss, generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))
        
        return disc_loss, gen_total_loss, gen_gan_loss, gen_l1_loss

    # --- 自定义训练循环 ---
    print("\n--- Starting Custom GAN Training Loop ---")
    for epoch in range(EPOCHS):
        start = time.time()
        
        # 记录每个epoch的损失
        epoch_disc_loss = []
        epoch_gen_loss = []

        for n, (input_image, target) in full_dataset.enumerate():
            disc_loss, gen_loss, _, _ = train_step(input_image, target, epoch)
            epoch_disc_loss.append(disc_loss)
            epoch_gen_loss.append(gen_loss)

            if (n + 1) % 10 == 0:
                print('.', end='', flush=True)

        # 在每个epoch结束后打印损失
        print(f"\nTime for epoch {epoch + 1} is {time.time()-start:.2f} sec")
        print(f"  Discriminator Loss: {np.mean(epoch_disc_loss):.4f}")
        print(f"  Generator Loss: {np.mean(epoch_gen_loss):.4f}")

        # 每20个epoch保存一次模型
        if (epoch + 1) % 20 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)
            print(f"  Checkpoint saved for epoch {epoch + 1}")

    # 训练结束后，保存最终的生成器模型，以便 run_analysis.py 使用
    final_model_path = os.path.join(MODEL_DIR, 'best_a2inet_model.h5')
    generator.save(final_model_path)

    print("\n--- Model Training Complete ---")
    print(f"Final generator model saved at: {final_model_path}")


if __name__ == '__main__':
    train_gan_model()

# ==============================================================================
# <<<<<<<<<<<<<<<<<<<<<<<<<< 修改区域结束 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# ==============================================================================