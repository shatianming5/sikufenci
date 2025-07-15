import os
import glob
from typing import List, Generator


def read_files(raw_path: str) -> Generator[tuple, None, None]:
    """
    读取指定文件夹中的所有txt文件
    
    Args:
        raw_path: 待分词语料的文件夹路径
        
    Yields:
        tuple: (文件名, 文件内容)
    """
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"路径不存在: {raw_path}")
    
    txt_files = glob.glob(os.path.join(raw_path, "*.txt"))
    if not txt_files:
        raise ValueError(f"在路径 {raw_path} 中没有找到txt文件")
    
    for file_path in txt_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                filename = os.path.basename(file_path)
                yield filename, content
        except UnicodeDecodeError:
            print(f"警告: 文件 {file_path} 编码不是UTF-8，跳过处理")
            continue


def write_files(result_path: str, filename: str, content: str) -> None:
    """
    将分词结果写入文件
    
    Args:
        result_path: 结果文件夹路径
        filename: 文件名
        content: 分词后的内容
    """
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    
    output_file = os.path.join(result_path, filename)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(content)


def validate_file_structure(raw_path: str) -> bool:
    """
    验证文件结构是否符合要求
    
    Args:
        raw_path: 待验证的文件夹路径
        
    Returns:
        bool: 是否符合要求
    """
    if not os.path.exists(raw_path):
        return False
    
    txt_files = glob.glob(os.path.join(raw_path, "*.txt"))
    return len(txt_files) > 0