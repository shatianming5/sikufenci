import os
import sys
from typing import List
from tqdm import tqdm

# 添加项目根目录到path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.file_utils import read_files, write_files, validate_file_structure
from utils.text_utils import preprocess_text, postprocess_text
from models.simple_tokenizer import SimpleTokenizer


def TCfenci_all(raw_path: str, resultpath: str, max_seq_length: int = 128, 
                eval_batch_size: int = 3) -> None:
    """
    简化版繁体中文分词主函数
    """
    print("开始初始化分词系统...")
    
    # 验证输入路径
    if not validate_file_structure(raw_path):
        raise ValueError(f"输入路径 {raw_path} 不存在或不包含txt文件")
    
    # 创建结果目录
    if not os.path.exists(resultpath):
        os.makedirs(resultpath)
    
    # 初始化简单分词器
    tokenizer = SimpleTokenizer()
    
    # 处理文件
    print("开始处理文件...")
    processed_files = 0
    
    try:
        for filename, content in read_files(raw_path):
            print(f"正在处理文件: {filename}")
            
            # 预处理文本
            sequences = preprocess_text(content, max_seq_length)
            
            if not sequences:
                print(f"文件 {filename} 为空，跳过处理")
                continue
            
            # 分词处理
            segmented_results = []
            for sequence in tqdm(sequences, desc="分词处理"):
                segmented_text = tokenizer.segment_text(sequence)
                segmented_results.append(segmented_text)
            
            # 后处理
            final_result = postprocess_text(segmented_results)
            
            # 写入结果
            write_files(resultpath, filename, final_result)
            
            processed_files += 1
            print(f"文件 {filename} 处理完成")
    
    except Exception as e:
        print(f"处理文件时发生错误: {e}")
        return
    
    print(f"分词完成！共处理了 {processed_files} 个文件")
    print(f"结果已保存到: {resultpath}")


if __name__ == "__main__":
    # 测试用例
    TCfenci_all("../../datatest", "../../resulttest", 128, 3)