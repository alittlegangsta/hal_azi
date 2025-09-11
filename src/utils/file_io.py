import pickle
import os

def load_pickle(file_path):
    """
    加载一个pickle文件。
    """
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def save_pickle(data, file_path):
    """
    将数据保存为一个pickle文件。
    """
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

def create_dir(dir_path):
    """
    安全地创建一个目录，如果它不存在的话。
    """
    os.makedirs(dir_path, exist_ok=True)