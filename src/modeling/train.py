# 文件路径: src/modeling/train.py

import os
import sys
import time
import tensorflow as tf
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.modeling.model import build_attention_unet, build_discriminator
from src.modeling.dataset import create_dataset
from src.utils.file_io import create_dir
from config import (
    ARRAY_ID, PROCESSED_DATA_DIR, MODEL_DIR, LOG_DIR,
    INPUT_SHAPE, FFT_COEFFICIENTS, MAX_PATH_DEPTH_POINTS,
    BATCH_SIZE, EPOCHS, LEARNING_RATE
)

# --- Hyperparameters ---
LAMBDA = 100
DISCRIMINATOR_LR_FACTOR = 0.5

# --- Loss Functions ---
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output) * 0.9, real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

def generator_loss(discriminator_fake_output, generated_image, target_image):
    gan_loss = cross_entropy(tf.ones_like(discriminator_fake_output), discriminator_fake_output)
    l1_loss = tf.reduce_mean(tf.abs(target_image - generated_image))
    total_gen_loss = gan_loss + (LAMBDA * l1_loss)
    return total_gen_loss, gan_loss, l1_loss

# ==============================================================================
# >>>>>>>>>> 代码修正区域 V5.6：修正过拟合测试的执行逻辑 <<<<<<<<<<<
# ==============================================================================

def run_overfitting_test():
    """
    This function now contains the entire logic for the overfitting test.
    """
    print("--- Starting FINAL SANITY CHECK: Overfitting Test on a Single Sample ---")

    # --- Data Pipeline ---
    tfrecord_dir = os.path.join(PROCESSED_DATA_DIR, f'array_{str(ARRAY_ID).zfill(2)}', 'tfrecords')
    tfrecord_path = os.path.join(tfrecord_dir, 'translation_data.tfrecord')
    print(f"Loading dataset from: {tfrecord_path}")
    
    # Load just one batch to start
    full_dataset = create_dataset(tfrecord_path, batch_size=BATCH_SIZE, is_training=False)
    
    # --- Failsafe: Check if the dataset is empty ---
    try:
        single_batch = next(iter(full_dataset))
        print("Successfully loaded one batch of data for the test.")
    except StopIteration:
        print("\nFATAL ERROR: The dataset is empty. No data found in the TFRecord file.")
        print("Please use '--force_run' in main.py to regenerate your data files correctly.")
        return # Exit the function

    # Create an infinitely repeating dataset from this single batch
    overfit_dataset = tf.data.Dataset.from_tensors(single_batch).repeat()

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
    
    generator_optimizer = tf.keras.optimizers.Adam(LEARNING_RATE, beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(LEARNING_RATE * DISCRIMINATOR_LR_FACTOR, beta_1=0.5)

    @tf.function
    def train_step(input_image, target):
        target_channeling = target[..., 0:1]
        target_good = target[..., 1:2]
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_image = generator(input_image, training=True)
            generated_channeling = generated_image[..., 0:1]
            generated_good = generated_image[..., 1:2]
            real_output_channeling = discriminator(target_channeling, training=True)
            fake_output_channeling = discriminator(generated_channeling, training=True)
            real_output_good = discriminator(target_good, training=True)
            fake_output_good = discriminator(generated_good, training=True)
            disc_loss_channeling = discriminator_loss(real_output_channeling, fake_output_channeling)
            disc_loss_good = discriminator_loss(real_output_good, fake_output_good)
            total_disc_loss = disc_loss_channeling + disc_loss_good
            
            gen_total_loss_channeling, _, _ = generator_loss(fake_output_channeling, generated_channeling, target_channeling)
            gen_total_loss_good, _, _ = generator_loss(fake_output_good, generated_good, target_good)
            total_gen_loss = gen_total_loss_channeling + gen_total_loss_good
            
        generator_gradients = gen_tape.gradient(total_gen_loss, generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(total_disc_loss, discriminator.trainable_variables)
        generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))
        return total_disc_loss, total_gen_loss

    # --- The Training Loop ---
    print("\n--- Starting Overfitting Test Training Loop ---")
    for step, (input_image, target) in enumerate(overfit_dataset):
        if step >= 300: # Use >= for clarity
            break
        
        start = time.time()
        disc_loss, gen_loss = train_step(input_image, target)
        
        if (step + 1) % 10 == 0:
            print(f"Step {step + 1}, Time: {time.time()-start:.2f} sec, Disc Loss: {disc_loss:.4f}, Gen Loss: {gen_loss:.4f}")

    print("\n--- Overfitting Test Complete ---")


if __name__ == '__main__':
    # This ensures the test is run when the script is executed
    run_overfitting_test()

# ==============================================================================
# <<<<<<<<<<<<<<<<<<<<<<<<<< 修正区域结束 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# ==============================================================================