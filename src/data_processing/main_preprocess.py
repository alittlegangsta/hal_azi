import os
import sys
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.interpolate import interp1d
import h5py
from tqdm import tqdm

#  添加项目根目录到系统路径，以便导入模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.file_io import create_dir, save_pickle
# 注意：config 模块的导入需要根据您的项目结构进行调整
from config import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR, ARRAY_ID,
    TARGET_DEPTH_RANGE, SONIC_RECEIVER_OFFSET, SONIC_SOURCE_OFFSET, TIME_STEPS
)

def load_and_prepare_data(raw_data_dir, array_id):
    """
    加载所有必需的原始.mat文件并进行初步准备。
    """
    print("Loading raw data files...")
    # 加载超声数据
    cast_data = loadmat(os.path.join(raw_data_dir, 'CAST.mat'))['CAST'][0, 0]
    df_cast = pd.DataFrame({
        'Depth': cast_data['Depth'].flatten(),
    })
    zc_data = cast_data['Zc']

    # 加载指定的声波阵列数据
    sonic_filename = f'XSILMR{str(array_id).zfill(2)}.mat'
    mat_content = loadmat(os.path.join(raw_data_dir, 'XSILMR', sonic_filename))
    struct_name = f'XSILMR{str(array_id).zfill(2)}'
    sonic_data = mat_content[struct_name][0, 0]
    
    df_sonic = pd.DataFrame({'Depth': sonic_data['Depth'].flatten()})

    # 加载8个方位的波形数据
    waveforms_dict = {}
    for char_code in range(ord('A'), ord('H') + 1):
        side = chr(char_code)
        wave_key = f'WaveRng{str(array_id).zfill(2)}Side{side}'
        waveforms_dict[side] = sonic_data[wave_key]

    print("Initial data loading complete.")
    return df_cast, zc_data, df_sonic, waveforms_dict


def create_unified_depth_grid(target_range, resolution=0.1):
    """
    创建一个统一的高分辨率深度轴。
    """
    print("Creating a unified high-resolution depth grid...")
    min_depth, max_depth = target_range
    if min_depth > max_depth:
        min_depth, max_depth = max_depth, min_depth
        
    unified_depth = np.arange(min_depth, max_depth, resolution)
    print(f"Unified depth grid created from {min_depth} ft to {max_depth} ft with {resolution} ft resolution.")
    return unified_depth

def interpolate_to_grid(df, zc_data, unified_depth):
    """
    将Zc数据插值到统一的深度网格上。
    """
    print("Interpolating Zc data to the unified grid...")
    sort_indices = np.argsort(df['Depth'])
    original_depths = df['Depth'].iloc[sort_indices]
    original_zc = zc_data[:, sort_indices]

    interp_func = interp1d(original_depths, original_zc, kind='linear', bounds_error=False, fill_value='extrapolate')
    interpolated_zc = interp_func(unified_depth)
    interpolated_zc = interpolated_zc.T
    
    print(f"Zc data interpolated to shape: {interpolated_zc.shape}")
    return interpolated_zc

def build_and_save_outputs(df_sonic, waveforms_dict, interpolated_zc, unified_depth, config):
    """
    构建“真值数字孪生”HDF5数据库并保存处理后的波形pkl文件。
    """
    array_id = config['ARRAY_ID']
    output_dir = os.path.join(config['PROCESSED_DATA_DIR'], f'array_{str(array_id).zfill(2)}')
    create_dir(output_dir)
    
    db_path = os.path.join(output_dir, f'ground_truth_db_array_{str(array_id).zfill(2)}.h5')
    waveforms_pkl_path = os.path.join(output_dir, 'processed_waveforms.pkl')
    
    print(f"Building Ground-Truth Digital Twin database at: {db_path}")

    # 筛选出在目标深度范围内的声波深度点
    sonic_depths_df = df_sonic[
        (df_sonic['Depth'] >= min(config['TARGET_DEPTH_RANGE'])) & 
        (df_sonic['Depth'] <= max(config['TARGET_DEPTH_RANGE']))
    ].copy()

    # 移除重复的深度值，并保留第一次出现的索引
    unique_indices = sonic_depths_df['Depth'].drop_duplicates().index
    unique_sonic_depths_df = sonic_depths_df.loc[unique_indices].sort_values(by='Depth')
    
    unique_sonic_depths = unique_sonic_depths_df['Depth'].values
    print(f"Found {len(df_sonic)} total sonic depths, "
          f"processing {len(unique_sonic_depths)} unique depths in target range.")
    
    # *** NEW CODE STARTS HERE ***
    # 根据唯一深度点的索引，筛选和处理波形数据
    print("Processing and stacking waveforms for unique depths...")
    processed_waveforms_list = []
    original_indices = unique_sonic_depths_df.index
    
    for idx in tqdm(original_indices, desc="Stacking waveforms"):
        stacked_waves = np.zeros((8, config['TIME_STEPS']), dtype=np.float32)
        for i, char_code in enumerate(range(ord('A'), ord('H') + 1)):
            side = chr(char_code)
            # 波形数据形状是 (time, depth), 我们需要转置
            stacked_waves[i, :] = waveforms_dict[side][:config['TIME_STEPS'], idx]
        processed_waveforms_list.append(stacked_waves)
        
    processed_waveforms_array = np.stack(processed_waveforms_list, axis=0)

    # 保存处理好的波形和深度到PKL文件
    waveforms_output = {
        'waveforms': processed_waveforms_array,
        'sonic_depths': unique_sonic_depths
    }
    save_pickle(waveforms_output, waveforms_pkl_path)
    print(f"Processed waveforms and depths saved to: {waveforms_pkl_path}")
    # *** NEW CODE ENDS HERE ***

    with h5py.File(db_path, 'w') as hf:
        hf.create_dataset('unified_depth_axis', data=unified_depth)
        hf.create_dataset('interpolated_zc_full', data=interpolated_zc)
        path_data_group = hf.create_group('path_data')

        for sonic_depth in tqdm(unique_sonic_depths, desc="Building HDF5 database"):
            receiver_offset = config['SONIC_RECEIVER_OFFSET'][array_id]
            source_offset = config['SONIC_SOURCE_OFFSET']
            
            end_depth = sonic_depth + receiver_offset
            start_depth = sonic_depth + source_offset
            
            if start_depth > end_depth:
                start_depth, end_depth = end_depth, start_depth

            path_indices = np.where((unified_depth >= start_depth) & (unified_depth <= end_depth))[0]

            if len(path_indices) > 0:
                zc_slice = interpolated_zc[path_indices, :]
                sonic_depth_key = str(sonic_depth).replace('.', '_')
                path_data_group.create_dataset(sonic_depth_key, data=zc_slice)

    print("Ground-Truth Digital Twin database build complete.")
    print(f"Total unique sonic depth points processed: {len(unique_sonic_depths)}")


def main():
    """
    主执行函数
    """
    config = {
        'RAW_DATA_DIR': RAW_DATA_DIR,
        'PROCESSED_DATA_DIR': PROCESSED_DATA_DIR,
        'ARRAY_ID': ARRAY_ID,
        'TARGET_DEPTH_RANGE': TARGET_DEPTH_RANGE,
        'SONIC_RECEIVER_OFFSET': SONIC_RECEIVER_OFFSET,
        'SONIC_SOURCE_OFFSET': SONIC_SOURCE_OFFSET,
        'TIME_STEPS': TIME_STEPS
    }

    df_cast, zc_data, df_sonic, waveforms_dict = load_and_prepare_data(config['RAW_DATA_DIR'], config['ARRAY_ID'])
    unified_depth = create_unified_depth_grid(config['TARGET_DEPTH_RANGE'])
    interpolated_zc = interpolate_to_grid(df_cast, zc_data, unified_depth)
    build_and_save_outputs(df_sonic, waveforms_dict, interpolated_zc, unified_depth, config)

if __name__ == '__main__':
    main()