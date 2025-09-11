import os
import sys
import matplotlib.pyplot as plt

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.file_io import load_pickle, create_dir
from config import LOG_DIR, ROOT_DIR

def plot_training_history():
    """
    加载并绘制模型的训练和验证历史曲线。
    """
    print("--- Starting Visualization of Training History ---")

    # 定义输入和输出路径
    history_path = os.path.join(LOG_DIR, 'training_history.pkl')
    output_dir = os.path.join(ROOT_DIR, 'output', 'visualization_plots')
    create_dir(output_dir)

    if not os.path.exists(history_path):
        print(f"Error: Training history file not found at {history_path}")
        print("Please run the training script first.")
        return

    history = load_pickle(history_path)

    # 创建一个包含两个子图的画布
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Model Training History', fontsize=16)

    # --- 子图1: 损失 (Loss) ---
    ax1.plot(history['loss'], label='Training Loss')
    if 'val_loss' in history:
        ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (Hybrid)')
    ax1.legend()
    ax1.grid(True)

    # --- 子图2: 平均绝对误差 (Mean Absolute Error) ---
    ax2.plot(history['mae'], label='Training MAE')
    if 'val_mae' in history:
        ax2.plot(history['val_mae'], label='Validation MAE')
    ax2.set_title('Training and Validation MAE')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Mean Absolute Error')
    ax2.legend()
    ax2.grid(True)

    # 保存图像
    plt.savefig(os.path.join(output_dir, '04_training_history.png'))
    plt.close()

    print(f"Saved plot: 04_training_history.png")
    print(f"Plot saved in: {output_dir}")

if __name__ == '__main__':
    plot_training_history()