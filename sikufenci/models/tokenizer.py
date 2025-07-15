import os
from typing import List, Dict
from pytorch_pretrained_bert import BertTokenizer


class SikuTokenizer:
    """
    SikuBERT分词器
    """
    
    def __init__(self, vocab_file: str = None):
        """
        初始化分词器
        
        Args:
            vocab_file: 词汇表文件路径
        """
        if vocab_file is None:
            # 使用默认的模型目录
            current_dir = os.path.dirname(os.path.abspath(__file__))
            model_dir = os.path.join(
                current_dir, 
                '..', 
                'train_fenci_sikuroberta_vocabtxt'
            )
            vocab_file = model_dir
        
        self.tokenizer = BertTokenizer.from_pretrained(vocab_file)
        self.vocab_size = len(self.tokenizer.vocab)
    
    def tokenize(self, text: str) -> List[str]:
        """
        对文本进行分词
        
        Args:
            text: 输入文本
            
        Returns:
            List[str]: 分词结果
        """
        return self.tokenizer.tokenize(text)
    
    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        """
        将token转换为ID
        
        Args:
            tokens: token列表
            
        Returns:
            List[int]: ID列表
        """
        return self.tokenizer.convert_tokens_to_ids(tokens)
    
    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        """
        将ID转换为token
        
        Args:
            ids: ID列表
            
        Returns:
            List[str]: token列表
        """
        return self.tokenizer.convert_ids_to_tokens(ids)
    
    def encode(self, text: str, max_length: int = 512) -> Dict[str, List[int]]:
        """
        编码文本
        
        Args:
            text: 输入文本
            max_length: 最大长度
            
        Returns:
            Dict[str, List[int]]: 编码结果
        """
        tokens = self.tokenize(text)
        
        # 截断到最大长度
        if len(tokens) > max_length - 2:  # 减2是为了[CLS]和[SEP]
            tokens = tokens[:max_length - 2]
        
        # 添加特殊token
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        
        # 转换为ID
        input_ids = self.convert_tokens_to_ids(tokens)
        
        # 生成attention mask
        attention_mask = [1] * len(input_ids)
        
        # 填充到最大长度
        while len(input_ids) < max_length:
            input_ids.append(0)
            attention_mask.append(0)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }