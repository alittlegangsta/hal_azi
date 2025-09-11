import os
import argparse
import time
import subprocess

# 导入config来获取文件路径
from config import ARRAY_ID, GROUND_TRUTH_DB_PATH, TFRECORD_DIR
# 注意：这是一个小技巧，让main.py能提前知道文件路径
# 我们需要确保config.py能处理环境变量不存在的情况
os.environ.setdefault('ARRAY_ID', '3')


def check_output_exists(step_name):
    """
    检查特定步骤的输出文件是否已存在。
    """
    # 更新ARRAY_ID以防命令行参数更改
    os.environ['ARRAY_ID'] = str(ARRAY_ID)
    
    # 重新导入以刷新路径
    from config import GROUND_TRUTH_DB_PATH, PROCESSED_DATA_DIR
    
    cwt_h5_path = os.path.join(PROCESSED_DATA_DIR, f'array_{str(ARRAY_ID).zfill(2)}', 'cwt_images.h5')
    tfrecord_path = os.path.join(TFRECORD_DIR, 'translation_data.tfrecord')
    
    if step_name == 'preprocess':
        # 预处理步骤检查两个文件
        waveforms_pkl_path = os.path.join(os.path.dirname(GROUND_TRUTH_DB_PATH), 'processed_waveforms.pkl')
        return os.path.exists(GROUND_TRUTH_DB_PATH) and os.path.exists(waveforms_pkl_path)
    if step_name == 'transform':
        return os.path.exists(cwt_h5_path)
    if step_name == 'create_tfrecords':
        return os.path.exists(tfrecord_path)
    return False

def run_step(script_path, step_name):
    """
    运行一个处理步骤并打印状态信息。
    """
    print("\n" + "="*80)
    print(f"--- Running Step: {step_name} ---")
    
    # 自动跳过功能
    if check_output_exists(step_name):
        print(f"--- Output file(s) for this step already exist. SKIPPING. ---")
        return

    print(f"Executing: python {script_path}")
    print("="*80 + "\n")
    
    start_time = time.time()
    try:
        process = subprocess.Popen(['python', script_path], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8')
        
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
        
        rc = process.poll()
        if rc != 0:
            raise subprocess.CalledProcessError(rc, script_path)

        end_time = time.time()
        print(f"\n--- Step '{step_name}' completed successfully in {end_time - start_time:.2f} seconds. ---")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"\n--- ERROR: Step '{step_name}' failed. ---")
        print(f"Error details: {e}")
        exit(1)

def main():
    parser = argparse.ArgumentParser(description="Acoustic Virtual-Imaging Project (AVIP) Main Orchestrator")
    parser.add_argument('--array', type=int, default=3, help='The array ID to process (e.g., 3, 7, 11).')
    parser.add_argument('--force_run', action='store_true', help='Force run all steps even if output files exist.')
    
    args = parser.parse_args()

    print(f"Setting environment variable 'ARRAY_ID' to {args.array}")
    os.environ['ARRAY_ID'] = str(args.array)

    workflow_steps = {
        'preprocess': 'src/data_processing/main_preprocess.py',
        'transform': 'src/cwt_transformation/main_transform_translation.py',
        'create_tfrecords': 'src/data_processing/create_tfrecords.py',
        'train': 'src/modeling/train.py'
    }
    
    # 如果是强制运行，则不检查文件存在
    if args.force_run:
        print("\n--- FORCE RUN enabled. All steps will be executed regardless of existing files. ---\n")
        for name, path in workflow_steps.items():
            run_step(path, name)
        return

    # 按顺序执行工作流，并自动跳过
    run_step(workflow_steps['preprocess'], 'preprocess')
    run_step(workflow_steps['transform'], 'transform')
    run_step(workflow_steps['create_tfrecords'], 'create_tfrecords')
    run_step(workflow_steps['train'], 'train')


if __name__ == '__main__':
    # 在config.py中将 ARRAY_ID = 3 修改为
    # ARRAY_ID = int(os.getenv('ARRAY_ID', '3'))
    # 以确保此机制生效
    main()