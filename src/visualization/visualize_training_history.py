# 文件路径: src/visualization/visualize_training_history.py

import os
import sys
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.file_io import load_pickle, create_dir
from config import LOG_DIR, ROOT_DIR

def plot_training_history():
    print("--- Starting Visualization of Training History ---")
    history_path = os.path.join(LOG_DIR, 'training_history.pkl') # 注意：新的训练脚本可能没有保存这个文件，需要手动在train.py中添加
    output_dir = os.path.join(ROOT_DIR, 'output', 'visualization_plots')
    create_dir(output_dir)

    if not os.path.exists(history_path):
        print(f"Error: Training history file not found at {history_path}")
        print("Please ensure your training script saves the history.")
        # 提示：可以在 train.py 的 model.fit() 返回的 history 对象后，用 save_pickle 保存
        return

    history = load_pickle(history_path)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Model Training History', fontsize=16)

    # --- 子图1: 损失 (Loss) ---
    ax1.plot(history['loss'], label='Training Loss')
    if 'val_loss' in history:
        ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss (Focal Loss)')
    ax1.legend(); ax1.grid(True)

    # ==============================================================================
    # >>>>>>>>>> 代码修改区域 V5.0：绘制准确率 <<<<<<<<<<<
    # ==============================================================================
    # --- 子图2: 准确率 (Accuracy) ---
    if 'accuracy' in history:
        ax2.plot(history['accuracy'], label='Training Accuracy')
        if 'val_accuracy' in history:
            ax2.plot(history['val_accuracy'], label='Validation Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy')
        ax2.legend(); ax2.grid(True)
    # ==============================================================================
    # <<<<<<<<<<<<<<<<<<<<<<<<<< 修改区域结束 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # ==============================================================================

    plt.savefig(os.path.join(output_dir, '04_training_history.png'))
    plt.close()
    print(f"Saved plot: 04_training_history.png to {output_dir}")

if __name__ == '__main__':
    plot_training_history()