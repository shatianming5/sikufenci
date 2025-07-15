import os
import torch
from typing import Optional
from pytorch_pretrained_bert import BertForTokenClassification


class ModelLoader:
    """
    模型加载器
    """
    
    def __init__(self, model_path: str = None, device: torch.device = None):
        """
        初始化模型加载器
        
        Args:
            model_path: 模型文件路径
            device: 设备
        """
        if model_path is None:
            # 使用默认的模型目录
            current_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(
                current_dir, 
                '..', 
                'train_fenci_sikuroberta_vocabtxt'
            )
        
        self.model_path = model_path
        self.device = device or torch.device('cpu')
        self.model = None
    
    def load_model(self, num_labels: int = 9) -> BertForTokenClassification:
        """
        加载预训练模型
        
        Args:
            num_labels: 标签数量
            
        Returns:
            BertForTokenClassification: 加载的模型
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型目录不存在: {self.model_path}")
        
        # 加载模型
        self.model = BertForTokenClassification.from_pretrained(
            self.model_path,
            num_labels=num_labels
        )
        
        # 移动到指定设备
        self.model.to(self.device)
        
        # 设置为评估模式
        self.model.eval()
        
        return self.model
    
    def get_model(self) -> Optional[BertForTokenClassification]:
        """
        获取已加载的模型
        
        Returns:
            Optional[BertForTokenClassification]: 模型对象或None
        """
        return self.model
    
    def is_model_loaded(self) -> bool:
        """
        检查模型是否已加载
        
        Returns:
            bool: 是否已加载
        """
        return self.model is not None
    
    def unload_model(self) -> None:
        """
        卸载模型
        """
        if self.model is not None:
            del self.model
            self.model = None
            
            # 清理GPU内存
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()