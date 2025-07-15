import re
from typing import List


def preprocess_text(text: str, max_seq_length: int = 128) -> List[str]:
    """
    预处理文本，切分成指定长度的序列
    
    Args:
        text: 原始文本
        max_seq_length: 最大序列长度
        
    Returns:
        List[str]: 切分后的文本序列
    """
    # 按行切分
    lines = text.strip().split('\n')
    sequences = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # 如果行长度超过max_seq_length，进行切分
        if len(line) > max_seq_length:
            for i in range(0, len(line), max_seq_length):
                sequences.append(line[i:i + max_seq_length])
        else:
            sequences.append(line)
    
    return sequences


def postprocess_text(sequences: List[str]) -> str:
    """
    后处理文本，合并分词结果
    
    Args:
        sequences: 分词后的序列列表
        
    Returns:
        str: 合并后的文本
    """
    return '\n'.join(sequences)


def validate_text_encoding(text: str) -> bool:
    """
    验证文本编码是否为UTF-8
    
    Args:
        text: 待验证的文本
        
    Returns:
        bool: 是否为有效的UTF-8编码
    """
    try:
        text.encode('utf-8')
        return True
    except UnicodeEncodeError:
        return False


def clean_text(text: str) -> str:
    """
    清理文本中的特殊字符
    
    Args:
        text: 原始文本
        
    Returns:
        str: 清理后的文本
    """
    # 移除不可见字符
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    return text