import torch
import multiprocessing
from typing import Tuple


def check_gpu_available() -> bool:
    """
    检查GPU是否可用
    
    Returns:
        bool: GPU是否可用
    """
    return False  # 强制使用CPU


def get_device() -> Tuple[torch.device, int]:
    """
    获取设备信息
    
    Returns:
        Tuple[torch.device, int]: 设备对象和可用核心数
    """
    device = torch.device("cpu")  # 强制使用CPU
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
        'device_type': 'CPU'  # 强制CPU类型
    }
    
    return info