import os
import sys
from typing import List
from tqdm import tqdm

# 添加项目根目录到path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.file_utils import read_files, write_files, validate_file_structure
from utils.text_utils import preprocess_text, postprocess_text
from models.simple_tokenizer import SimpleTokenizer

try:
    from utils.device_utils import get_device, get_device_info
    from models.tokenizer import SikuTokenizer
    from models.model_loader import ModelLoader
    from models.predictor import Predictor
    USE_ADVANCED_MODEL = True
except ImportError:
    USE_ADVANCED_MODEL = False
    print("高级模型依赖未安装，使用简化版分词器")


def TCfenci_all(raw_path: str, resultpath: str, max_seq_length: int = 128, 
                eval_batch_size: int = 3) -> None:
    """
    增强版繁体中文分词主函数
    
    Args:
        raw_path: 待分词语料的文件夹路径
        resultpath: 分词结果的存储路径
        max_seq_length: 最大序列长度，默认128
        eval_batch_size: 批处理大小，默认3
    """
    print("开始初始化分词系统...")
    
    # 验证输入路径
    if not validate_file_structure(raw_path):
        raise ValueError(f"输入路径 {raw_path} 不存在或不包含txt文件")
    
    # 创建结果目录
    if not os.path.exists(resultpath):
        os.makedirs(resultpath)
    
    # 选择分词器
    if USE_ADVANCED_MODEL:
        try:
            # 获取设备信息
            device, num_cores = get_device()
            device_info = get_device_info()
            
            print(f"设备信息: {device_info['device_type']}")
            if device_info['device_type'] == 'GPU':
                print(f"GPU名称: {device_info.get('gpu_name', 'Unknown')}")
            else:
                print(f"CPU核心数: {num_cores}")
            
            # 初始化高级组件
            print("正在加载高级分词器...")
            tokenizer = SikuTokenizer()
            
            print("正在加载模型...")
            model_loader = ModelLoader(device=device)
            
            print("正在初始化预测器...")
            predictor = Predictor(model_loader, tokenizer)
            
            use_advanced = True
            
        except Exception as e:
            print(f"高级模型初始化失败: {e}")
            print("切换到简化版分词器")
            use_advanced = False
    else:
        use_advanced = False
    
    if not use_advanced:
        print("使用简化版分词器...")
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
            if use_advanced:
                # 使用高级分词器
                segmented_results = predictor.process_text_batch(
                    sequences, 
                    max_seq_length, 
                    eval_batch_size
                )
            else:
                # 使用简化分词器
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
    
    finally:
        # 清理资源
        if use_advanced and 'predictor' in locals():
            predictor.model_loader.unload_model()
    
    print(f"分词完成！共处理了 {processed_files} 个文件")
    print(f"结果已保存到: {resultpath}")


if __name__ == "__main__":
    # 测试用例
    TCfenci_all("../../datatest", "../../resulttest", 128, 3)