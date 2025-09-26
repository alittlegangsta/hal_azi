# 文件路径: src/modeling/train.py

import os
import sys
import time
import tensorflow as tf
import numpy as np

#  添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.modeling.model import build_attention_unet, build_discriminator
from src.modeling.dataset import create_dataset
from src.utils.file_io import create_dir
from config import (
    ARRAY_ID, PROCESSED_DATA_DIR, MODEL_DIR, LOG_DIR,
    INPUT_SHAPE, FFT_COEFFICIENTS, MAX_PATH_DEPTH_POINTS,
    BATCH_SIZE, EPOCHS, LEARNING_RATE
)

# --- Hyperparameters for GAN Stability ---
# LAMBDA controls the weight of the L1 loss (pixel-wise accuracy)
LAMBDA = 100 
# Using a slightly lower learning rate for the discriminator can help balance the training
DISCRIMINATOR_LR_FACTOR = 0.5 

# --- Loss Functions ---
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    # Technique 1: Label Smoothing (using 0.9 instead of 1.0 for real labels)
    real_loss = cross_entropy(tf.ones_like(real_output) * 0.9, real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

def generator_loss(discriminator_fake_output, generated_image, target_image):
    # Adversarial loss (how well it fools the discriminator)
    gan_loss = cross_entropy(tf.ones_like(discriminator_fake_output), discriminator_fake_output)
    # Content loss (how close it is to the real image)
    l1_loss = tf.reduce_mean(tf.abs(target_image - generated_image))
    total_gen_loss = gan_loss + (LAMBDA * l1_loss)
    return total_gen_loss, gan_loss, l1_loss

def train_gan_model():
    """Main function to train the AVIP-GAN model with stability enhancements."""
    print("--- Starting Phase 3: Model Training for AVIP (GAN with Stability Tricks) ---")

    # --- Data Pipeline ---
    tfrecord_dir = os.path.join(PROCESSED_DATA_DIR, f'array_{str(ARRAY_ID).zfill(2)}', 'tfrecords')
    tfrecord_path = os.path.join(tfrecord_dir, 'translation_data.tfrecord')
    print(f"Loading dataset from: {tfrecord_path}")
    full_dataset = create_dataset(tfrecord_path, batch_size=BATCH_SIZE, is_training=True)
    
    # --- Build Models and Optimizers ---
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

    # Technique 2: Slower learning rate for the discriminator
    generator_optimizer = tf.keras.optimizers.Adam(LEARNING_RATE, beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(LEARNING_RATE * DISCRIMINATOR_LR_FACTOR, beta_1=0.5)

    # --- Checkpoint Manager ---
    create_dir(MODEL_DIR)
    checkpoint_prefix = os.path.join(MODEL_DIR, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer,
                                     generator=generator,
                                     discriminator=discriminator)

    # ==============================================================================
    # >>>>>>>>>> 代码修改区域 V5.5：修正并强化的GAN训练步骤 <<<<<<<<<<<
    # ==============================================================================
    @tf.function
    def train_step(input_image, target):
        # Decompose the 2-channel target into individual 1-channel masks
        target_channeling = target[..., 0:1] # Shape: (batch, H, W, 1)
        target_good = target[..., 1:2]       # Shape: (batch, H, W, 1)

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # 1. Generator creates a fake 2-channel image
            generated_image = generator(input_image, training=True)
            # Decompose the fake image into 1-channel masks
            generated_channeling = generated_image[..., 0:1]
            generated_good = generated_image[..., 1:2]

            # 2. Discriminator evaluates real and fake masks for BOTH channels
            real_output_channeling = discriminator(target_channeling, training=True)
            fake_output_channeling = discriminator(generated_channeling, training=True)
            
            real_output_good = discriminator(target_good, training=True)
            fake_output_good = discriminator(generated_good, training=True)

            # 3. Calculate losses for both models
            # Discriminator loss is the sum of losses from both channels
            disc_loss_channeling = discriminator_loss(real_output_channeling, fake_output_channeling)
            disc_loss_good = discriminator_loss(real_output_good, fake_output_good)
            total_disc_loss = disc_loss_channeling + disc_loss_good

            # Generator loss is also the sum of losses from both channels
            gen_total_loss_channeling, _, _ = generator_loss(fake_output_channeling, generated_channeling, target_channeling)
            gen_total_loss_good, _, _ = generator_loss(fake_output_good, generated_good, target_good)
            total_gen_loss = gen_total_loss_channeling + gen_total_loss_good

        # 4. Calculate and apply gradients
        generator_gradients = gen_tape.gradient(total_gen_loss, generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(total_disc_loss, discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))
        
        return total_disc_loss, total_gen_loss
    # ==============================================================================
    # <<<<<<<<<<<<<<<<<<<<<<<<<< 修改区域结束 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # ==============================================================================

    # --- The Training Loop ---
    print("\n--- Starting Custom GAN Training Loop ---")
    for epoch in range(EPOCHS):
        start = time.time()
        
        epoch_disc_loss, epoch_gen_loss = [], []

        for n, (input_image, target) in full_dataset.enumerate():
            disc_loss, gen_loss = train_step(input_image, target)
            epoch_disc_loss.append(disc_loss)
            epoch_gen_loss.append(gen_loss)

            if (n + 1) % 10 == 0:
                print('.', end='', flush=True)

        print(f"\nTime for epoch {epoch + 1} is {time.time()-start:.2f} sec")
        print(f"  Discriminator Loss: {np.mean(epoch_disc_loss):.4f}")
        print(f"  Generator Loss: {np.mean(epoch_gen_loss):.4f}")

        if (epoch + 1) % 20 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)
            print(f"  Checkpoint saved for epoch {epoch + 1}")

    # Save the final generator model for analysis
    final_model_path = os.path.join(MODEL_DIR, 'best_a2inet_model.h5')
    generator.save(final_model_path)

    print("\n--- Model Training Complete ---")
    print(f"Final generator model saved at: {final_model_path}")

if __name__ == '__main__':
    train_gan_model()