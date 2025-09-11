import os
import numpy as np

# ==============================================================================
# --- 项目根目录设置 (Project Root Directory) ---
# ==============================================================================
# 通常，config.py位于项目根目录，所以我们使用'.'
# 如果您从其他地方运行，可能需要调整
ROOT_DIR = '.'

# ==============================================================================
# --- 核心实验参数 (Core Experiment Parameters) ---
# ==============================================================================
# 通过命令行参数 --array 指定要处理的阵列接收器ID
# 默认值为3号阵列
ARRAY_ID = int(os.getenv('ARRAY_ID', '3'))

# 调试模式：如果为True，将只处理少量数据点以进行快速流程验证
# 这直接响应了备忘录中关于高效调试的要求
DEBUG_MODE = True
DEBUG_SONIC_DEPTH_POINTS = 260 # 在调试模式下处理的声波深度点数量

# ==============================================================================
# --- 任务类型定义 (Task Type Definition) ---
# ==============================================================================
# 定义当前要执行的任务类型，这允许我们在新旧方案之间切换
# 'regression': 旧的CSI回归任务
# 'image_translation': 新的AVIP项目，从声波CWT图像到Zc FFT图像的翻译
TASK_TYPE = 'image_translation'

# ==============================================================================
# --- 目录与路径设置 (Directories and Paths) ---
# ==============================================================================
# 原始数据目录
RAW_DATA_DIR = os.path.join(ROOT_DIR, 'data', 'raw')

# 处理后数据的根目录
PROCESSED_DATA_DIR = os.path.join(ROOT_DIR, 'data', 'processed', TASK_TYPE)

# 日志、模型和结果的输出目录
OUTPUT_DIR = os.path.join(ROOT_DIR, 'output', TASK_TYPE, f'array_{str(ARRAY_ID).zfill(2)}')
LOG_DIR = os.path.join(OUTPUT_DIR, 'logs')
MODEL_DIR = os.path.join(OUTPUT_DIR, 'models')
RESULTS_DIR = os.path.join(OUTPUT_DIR, 'results')

# --- AVIP项目特定路径 ---
# AVIP Phase 1: “真值数字孪生”数据库路径
GROUND_TRUTH_DB_DIR = os.path.join(PROCESSED_DATA_DIR, f'array_{str(ARRAY_ID).zfill(2)}')
GROUND_TRUTH_DB_PATH = os.path.join(GROUND_TRUTH_DB_DIR, f'ground_truth_db_array_{str(ARRAY_ID).zfill(2)}.h5')

# AVIP Phase 2: TFRecord数据集路径
TFRECORD_DIR = os.path.join(PROCESSED_DATA_DIR, f'array_{str(ARRAY_ID).zfill(2)}', 'tfrecords')
MAX_PATH_DEPTH_POINTS = 40
# (将在下一阶段使用)

# ==============================================================================
# --- 数据处理参数 (Data Processing Parameters) ---
# ==============================================================================
# 目标分析的深度范围 (ft)
TARGET_DEPTH_RANGE = (2732, 4132)

# --- 物理模型参数 (Physical Model Parameters) ---
# 基于 "数据信息.md" 的物理配置
# 声波阵列以7号接收器为深度基准点 (D)
# 接收器号越大，位置越浅。接收器间距0.5ft
# 声源在1号接收器下方1ft处

# 1号接收器深度: D + (7-1)*0.5 = D + 3.0 ft
# 声源深度: (D + 3.0) + 1.0 = D + 4.0 ft
SONIC_SOURCE_OFFSET = 4.0  # ft, 相对于7号接收器测量深度的偏移量

# 各接收器相对于7号接收器测量深度的偏移量 (ft)
# 正值表示更深，负值表示更浅
SONIC_RECEIVER_OFFSET = {
    1: 3.0, 2: 2.5, 3: 2.0, 4: 1.5, 5: 1.0, 6: 0.5, 7: 0.0,
    8: -0.5, 9: -1.0, 10: -1.5, 11: -2.0, 12: -2.5, 13: -3.0
}

# --- AVIP项目特定数据参数 ---
# 在FFT变换后保留的低频系数数量 (N)
# 最终标签向量长度为 N (因为我们只取幅度)
FFT_COEFFICIENTS = 30 # (N)

# ==============================================================================
# --- CWT和模型参数 (CWT & Model Parameters) ---
# ==============================================================================

# --- CWT分块处理参数 (CWT Chunking Parameter) ---
# 为了在转换巨大的数据集时避免内存溢出，我们一次只处理一小块数据
# 这个值可以根据系统RAM大小调整。对于拥有大量RAM的服务器，可以设大一些
CWT_CHUNK_SIZE = 2048  # 一次处理2048个样本的CWT变换

# CWT变换参数
SAMPLING_RATE = 1e5  # Hz (采样间隔 10 us)
CWT_SCALES = np.arange(1, 151) # 尺度数量
CWT_WAVELET = 'cmor1.5-1.0'  # 复数Morlet小波

# 模型输入形状
TIME_STEPS = 400
N_CHANNELS = 8
INPUT_SHAPE = (len(CWT_SCALES), TIME_STEPS, N_CHANNELS)

# 训练参数
BATCH_SIZE = 128
EPOCHS = 100
LEARNING_RATE = 1e-4
VALIDATION_SPLIT = 0.2

# ==============================================================================
# --- 旧回归任务参数 (Legacy Regression Task Parameters) - (存档) ---
# ==============================================================================
# # 以下参数仅用于旧的CSI回归任务，在AVIP项目中不使用
# CSI_THRESHOLD = 2.5
# VERTICAL_WINDOW_SIZE = 2.0  # ft
# DATA_BALANCE_CONFIG = {
#     'bins': [0, 0.1, 0.2, 0.4, 0.6, 1.0],
#     'samples_per_bin': 5000
# }