# 文件路径: config.py
# (新分支 'percentage_' 的版本)

import os
import numpy as np
import pywt

# ==============================================================================
# --- 项目根目录设置 (Project Root Directory) ---
# ==============================================================================
ROOT_DIR = '.'
ARRAY_ID = int(os.getenv('ARRAY_ID', '3'))

# ==============================================================================
# --- 核心实验参数 (Core Experiment Parameters) ---
# ==============================================================================
DEBUG_MODE = False
DEBUG_SONIC_DEPTH_POINTS = 260

# ==============================================================================
# --- 任务类型定义 (Task Type Definition) ---
# ==============================================================================
# 关键改动：将任务类型更改为 "fft_regression"
# 这将自动将所有输出重定向到新的目录，确保与旧分支的文件共存。
# 旧分支的 TASK_TYPE 是 'image_translation'
TASK_TYPE = 'fft_regression'

# ==============================================================================
# --- 目录与路径设置 (Directories and Paths) ---
# ==============================================================================
# 路径现在将自动指向 '.../fft_regression/...'
RAW_DATA_DIR = os.path.join(ROOT_DIR, 'data', 'raw')
PROCESSED_DATA_DIR = os.path.join(ROOT_DIR, 'data', 'processed', TASK_TYPE)
OUTPUT_DIR = os.path.join(ROOT_DIR, 'output', TASK_TYPE, f'array_{str(ARRAY_ID).zfill(2)}')
LOG_DIR = os.path.join(OUTPUT_DIR, 'logs')
MODEL_DIR = os.path.join(OUTPUT_DIR, 'models')
RESULTS_DIR = os.path.join(OUTPUT_DIR, 'results')
GROUND_TRUTH_DB_DIR = os.path.join(PROCESSED_DATA_DIR, f'array_{str(ARRAY_ID).zfill(2)}')
GROUND_TRUTH_DB_PATH = os.path.join(GROUND_TRUTH_DB_DIR, f'ground_truth_db_array_{str(ARRAY_ID).zfill(2)}.h5')
TFRECORD_DIR = os.path.join(PROCESSED_DATA_DIR, f'array_{str(ARRAY_ID).zfill(2)}', 'tfrecords')

# ==============================================================================
# --- 数据处理参数 (Data Processing Parameters) ---
# ==============================================================================
TARGET_DEPTH_RANGE = (2732, 4132)
SONIC_SOURCE_OFFSET = 4.0
SONIC_RECEIVER_OFFSET = {
    1: 3.0, 2: 2.5, 3: 2.0, 4: 1.5, 5: 1.0, 6: 0.5, 7: 0.0,
    8: -0.5, 9: -1.0, 10: -1.5, 11: -2.0, 12: -2.5, 13: -3.0
}
# FFT 标签所需的参数
FFT_COEFFICIENTS = 30
MAX_PATH_DEPTH_POINTS = 70

# ==============================================================================
# --- CWT和模型参数 (CWT & Model Parameters) ---
# ==============================================================================
# (CWT 部分保持不变)
SAMPLING_RATE = 1e5
CWT_WAVELET = 'cmor1.5-1.0'
TIME_STEPS = 400
N_CHANNELS = 8
CWT_CHUNK_SIZE = 2048

TARGET_FREQ_MIN_HZ = 1000
TARGET_FREQ_MAX_HZ = 30000
N_SCALES = 150

CENTRAL_FREQ = pywt.central_frequency(CWT_WAVELET)
MIN_SCALE = (CENTRAL_FREQ * SAMPLING_RATE) / TARGET_FREQ_MAX_HZ
MAX_SCALE = (CENTRAL_FREQ * SAMPLING_RATE) / TARGET_FREQ_MIN_HZ
CWT_SCALES = np.geomspace(MIN_SCALE, MAX_SCALE, N_SCALES)
FREQUENCIES_HZ = pywt.scale2frequency(CWT_WAVELET, CWT_SCALES) / (1 / SAMPLING_RATE)
CWT_FREQUENCIES_KHZ = FREQUENCIES_HZ / 1000

# --- 模型参数 ---
INPUT_SHAPE = (N_SCALES, TIME_STEPS, N_CHANNELS)
LAST_CONV_LAYER_NAME = 'top_conv' # 保持与分析脚本一致

# --- 训练参数 ---
BATCH_SIZE = 128
EPOCHS = 100
LEARNING_RATE = 1e-4
VALIDATION_SPLIT = 0.2