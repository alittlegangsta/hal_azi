# 文件路径: src/visualization/visualize_training_history.py
# (新分支 'percentage_' 的版本)
import os
import sys
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.file_io import load_pickle, create_dir
from config import LOG_DIR, ROOT_DIR, TASK_TYPE

def plot_training_history():
    print("--- Starting Visualization of Training History ---")

    # --- 智能识别功能 ---
    if TASK_TYPE == 'fft_regression':
        history_filename = 'training_history_fft.pkl'
        title_suffix = '(FFT Model)'
    else:
        history_filename = 'training_history_advanced.pkl'
        title_suffix = '(1D Profile Model)'
    # --- 识别结束 ---

    history_path = os.path.join(LOG_DIR, history_filename)
    
    if not os.path.exists(history_path):
        # 尝试回退到旧的旧的名称
        if TASK_TYPE != 'fft_regression' and os.path.exists(os.path.join(LOG_DIR, 'training_history.pkl')):
             history_path = os.path.join(LOG_DIR, 'training_history.pkl')
        else:
            print(f"Error: No training history file found: {history_path}")
            print("Please run a full training session first.")
            return

    output_dir = os.path.join(ROOT_DIR, 'output', 'visualization_plots', TASK_TYPE)
    create_dir(output_dir)
    
    print(f"Loading history from {history_path}...")
    history = load_pickle(history_path)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'Profile Regressor Model Training History {title_suffix}', fontsize=16)

    # ... (绘图代码保持不变) ...
    ax1.plot(history['loss'], label='Training Loss')
    if 'val_loss' in history:
        ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_title('Training & Validation Loss')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss (Huber or MSE)'); ax1.legend(); ax1.grid(True)
    if 'mae' in history:
        ax2.plot(history['mae'], label='Training MAE')
        if 'val_mae' in history:
            ax2.plot(history['val_mae'], label='Validation MAE')
        ax2.set_title('Training & Validation MAE')
        ax2.set_xlabel('Epoch'); ax2.set_ylabel('Mean Absolute Error (%)'); ax2.legend(); ax2.grid(True)
    # ... (绘图代码结束) ...

    output_path = os.path.join(output_dir, f'05_training_history_{TASK_TYPE}.png')
    plt.savefig(output_path)
    plt.close()
    print(f"Plot saved successfully: {output_path}")

if __name__ == '__main__':
    plot_training_history()