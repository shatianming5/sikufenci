import json
import os
from typing import List, Dict, Generator


def read_json_file(json_path: str) -> List[Dict]:
    """
    读取JSON文件
    
    Args:
        json_path: JSON文件路径
        
    Returns:
        List[Dict]: JSON数据列表
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON文件不存在: {json_path}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data


def extract_text_from_json(json_data: List[Dict], text_field: str = 'paragraphs') -> Generator[tuple, None, None]:
    """
    从JSON数据中提取文本内容
    
    Args:
        json_data: JSON数据列表
        text_field: 文本字段名
        
    Yields:
        tuple: (文件名, 文本内容)
    """
    for i, item in enumerate(json_data):
        if text_field in item:
            # 生成文件名
            author = item.get('author', 'unknown')
            title = item.get('title', 'untitled')
            filename = f"poem_{i+1:05d}_{author}_{title}.txt"
            
            # 清理文件名中的特殊字符
            filename = filename.replace('/', '_').replace('\\', '_').replace(':', '_')
            
            content = item[text_field]
            yield filename, content


def json_to_txt_files(json_path: str, output_dir: str, text_field: str = 'paragraphs', 
                     max_files: int = None) -> int:
    """
    将JSON文件转换为多个txt文件
    
    Args:
        json_path: JSON文件路径
        output_dir: 输出目录
        text_field: 文本字段名
        max_files: 最大文件数量限制
        
    Returns:
        int: 转换的文件数量
    """
    # 读取JSON数据
    json_data = read_json_file(json_path)
    
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 提取文本并写入文件
    count = 0
    for filename, content in extract_text_from_json(json_data, text_field):
        if max_files and count >= max_files:
            break
            
        output_path = os.path.join(output_dir, filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        count += 1
    
    return count


def get_json_stats(json_path: str) -> Dict:
    """
    获取JSON文件统计信息
    
    Args:
        json_path: JSON文件路径
        
    Returns:
        Dict: 统计信息
    """
    json_data = read_json_file(json_path)
    
    stats = {
        'total_items': len(json_data),
        'sample_keys': list(json_data[0].keys()) if json_data else [],
        'authors': set(),
        'total_chars': 0
    }
    
    for item in json_data:
        if 'author' in item:
            stats['authors'].add(item['author'])
        if 'paragraphs' in item:
            stats['total_chars'] += len(item['paragraphs'])
    
    stats['unique_authors'] = len(stats['authors'])
    stats['authors'] = list(stats['authors'])
    
    return stats