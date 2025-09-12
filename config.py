import os
import numpy as np
import pywt

# ==============================================================================
# --- 项目根目录设置 (Project Root Directory) ---
# ==============================================================================
ROOT_DIR = '.'
# 通过环境变量获取ARRAY_ID，确保main.py的命令行参数能被读取
# 在直接运行脚本时，默认值为3
ARRAY_ID = int(os.getenv('ARRAY_ID', '3'))

# ==============================================================================
# --- 核心实验参数 (Core Experiment Parameters) ---
# ==============================================================================
DEBUG_MODE = True
# 确保在调试时至少能产生3个批次的数据 (ceil(260/128) = 3)
DEBUG_SONIC_DEPTH_POINTS = 260

# ==============================================================================
# --- 任务类型定义 (Task Type Definition) ---
# ==============================================================================
TASK_TYPE = 'image_translation'

# ==============================================================================
# --- 目录与路径设置 (Directories and Paths) ---
# ==============================================================================
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
FFT_COEFFICIENTS = 30
MAX_PATH_DEPTH_POINTS = 70

# ==============================================================================
# --- CWT和模型参数 (CWT & Model Parameters) ---
# ==============================================================================
# --- CWT变换参数 ---
SAMPLING_RATE = 1e5  # Hz (采样间隔 10 us)
CWT_WAVELET = 'cmor1.5-1.0'
TIME_STEPS = 400
N_CHANNELS = 8
CWT_CHUNK_SIZE = 2048

# *** 逻辑修正：根据目标频率范围 (1-30kHz) 计算CWT尺度 ***
TARGET_FREQ_MIN_HZ = 1000  # 1 kHz
TARGET_FREQ_MAX_HZ = 30000 # 30 kHz
N_SCALES = 150 # 保持尺度的数量以维持图像分辨率

# 计算能产生目标频率的尺度范围
SAMPLING_PERIOD = 1 / SAMPLING_RATE
# 尺度 s = (小波中心频率 * 采样率) / 目标频率
# pywt.central_frequency('cmor1.5-1.0') -> 1.0
CENTRAL_FREQ = pywt.central_frequency(CWT_WAVELET)
MIN_SCALE = (CENTRAL_FREQ * SAMPLING_RATE) / TARGET_FREQ_MAX_HZ
MAX_SCALE = (CENTRAL_FREQ * SAMPLING_RATE) / TARGET_FREQ_MIN_HZ

# 使用对数间隔生成尺度数组，以在低频获得更好分辨率，符合原始方案建议
CWT_SCALES = np.geomspace(MIN_SCALE, MAX_SCALE, N_SCALES)

# 预计算CWT尺度对应的频率轴 (单位: kHz)
FREQUENCIES_HZ = pywt.scale2frequency(CWT_WAVELET, CWT_SCALES) / SAMPLING_PERIOD
CWT_FREQUENCIES_KHZ = FREQUENCIES_HZ / 1000
# *** 修正结束 ***

# --- 模型参数 ---
INPUT_SHAPE = (N_SCALES, TIME_STEPS, N_CHANNELS)
LAST_CONV_LAYER_NAME = 'final_conv_head'

# --- 训练参数 (针对A100优化) ---
BATCH_SIZE = 128
EPOCHS = 100
LEARNING_RATE = 1e-4
VALIDATION_SPLIT = 0.2