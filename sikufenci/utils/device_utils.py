import torch
import multiprocessing
from typing import Tuple


def check_gpu_available() -> bool:
    """
    检查GPU是否可用
    
    Returns:
        bool: GPU是否可用
    """
    return torch.cuda.is_available()


def get_device() -> Tuple[torch.device, int]:
    """
    获取设备信息
    
    Returns:
        Tuple[torch.device, int]: 设备对象和可用核心数
    """
    if check_gpu_available():
        device = torch.device("cuda")
        num_cores = 1  # GPU模式下使用1个核心
    else:
        device = torch.device("cpu")
        num_cores = multiprocessing.cpu_count()  # CPU模式下使用所有核心
    
    return device, num_cores


def get_device_info() -> dict:
    """
    获取详细的设备信息
    
    Returns:
        dict: 设备信息字典
    """
    device, num_cores = get_device()
    
    info = {
        'device': device,
        'num_cores': num_cores,
        'device_type': 'GPU' if device.type == 'cuda' else 'CPU'
    }
    
    if device.type == 'cuda':
        info['gpu_name'] = torch.cuda.get_device_name(0)
        info['gpu_memory'] = torch.cuda.get_device_properties(0).total_memory
    
    return info