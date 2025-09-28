# 文件路径: src/visualization/visualize_training_history.py

import os
import sys
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.file_io import load_pickle, create_dir
from config import LOG_DIR, ROOT_DIR

def plot_training_history():
    """
    Loads and plots the training and validation history curves for the profile regressor model.
    """
    print("--- Starting Visualization of Training History ---")

    history_path = os.path.join(LOG_DIR, 'training_history_advanced.pkl')
    if not os.path.exists(history_path):
        history_path = os.path.join(LOG_DIR, 'training_history.pkl')
        if not os.path.exists(history_path):
            print(f"Error: No training history file found in {LOG_DIR}.")
            print("Please run a full training session first.")
            return

    output_dir = os.path.join(ROOT_DIR, 'output', 'visualization_plots')
    create_dir(output_dir)
    
    print(f"Loading history from {history_path}...")
    history = load_pickle(history_path)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Profile Regressor Model Training History', fontsize=16)

    # --- Subplot 1: Loss ---
    ax1.plot(history['loss'], label='Training Loss')
    if 'val_loss' in history:
        ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_title('Training & Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (Huber or MSE)')
    ax1.legend()
    ax1.grid(True)

    # --- Subplot 2: Mean Absolute Error ---
    if 'mae' in history:
        ax2.plot(history['mae'], label='Training MAE')
        if 'val_mae' in history:
            ax2.plot(history['val_mae'], label='Validation MAE')
        ax2.set_title('Training & Validation MAE')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Mean Absolute Error (%)')
        ax2.legend()
        ax2.grid(True)

    output_path = os.path.join(output_dir, '05_training_history_regression.png')
    plt.savefig(output_path)
    plt.close()

    print(f"Plot saved successfully: {output_path}")

if __name__ == '__main__':
    plot_training_history()