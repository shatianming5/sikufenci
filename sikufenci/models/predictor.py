import torch
import numpy as np
from typing import List, Tuple
from tqdm import tqdm
from .tokenizer import SikuTokenizer
from .model_loader import ModelLoader


class Predictor:
    """
    分词预测器
    """
    
    def __init__(self, model_loader: ModelLoader, tokenizer: SikuTokenizer):
        """
        初始化预测器
        
        Args:
            model_loader: 模型加载器
            tokenizer: 分词器
        """
        self.model_loader = model_loader
        self.tokenizer = tokenizer
        self.model = None
    
    def load_model(self) -> None:
        """
        加载模型
        """
        if not self.model_loader.is_model_loaded():
            self.model = self.model_loader.load_model()
        else:
            self.model = self.model_loader.get_model()
    
    def predict_batch(self, texts: List[str], max_seq_length: int = 128) -> List[List[str]]:
        """
        批量预测
        
        Args:
            texts: 文本列表
            max_seq_length: 最大序列长度
            
        Returns:
            List[List[str]]: 分词结果列表
        """
        if self.model is None:
            self.load_model()
        
        results = []
        
        for text in tqdm(texts, desc="分词处理"):
            segmented_text = self.predict_single(text, max_seq_length)
            results.append(segmented_text)
        
        return results
    
    def predict_single(self, text: str, max_seq_length: int = 128) -> List[str]:
        """
        单个文本预测
        
        Args:
            text: 输入文本
            max_seq_length: 最大序列长度
            
        Returns:
            List[str]: 分词结果
        """
        # 编码文本
        encoded = self.tokenizer.encode(text, max_seq_length)
        
        # 转换为tensor
        input_ids = torch.tensor([encoded['input_ids']], device=self.model_loader.device)
        attention_mask = torch.tensor([encoded['attention_mask']], device=self.model_loader.device)
        
        # 预测
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            logits = outputs[0]
            predictions = torch.argmax(logits, dim=-1)
        
        # 解码结果
        predicted_labels = predictions[0].cpu().numpy()
        tokens = self.tokenizer.convert_ids_to_tokens(encoded['input_ids'])
        
        # 根据预测结果进行分词
        segmented_tokens = self._segment_tokens(tokens, predicted_labels)
        
        return segmented_tokens
    
    def _segment_tokens(self, tokens: List[str], labels: List[int]) -> List[str]:
        """
        根据标签对token进行分词
        
        Args:
            tokens: token列表
            labels: 标签列表
            
        Returns:
            List[str]: 分词结果
        """
        segments = []
        current_segment = []
        
        for i, (token, label) in enumerate(zip(tokens, labels)):
            # 跳过特殊token
            if token in ['[CLS]', '[SEP]', '[PAD]']:
                continue
            
            # 处理subword token
            if token.startswith('##'):
                token = token[2:]
            
            current_segment.append(token)
            
            # 如果标签为1（分词边界）或者是最后一个token，则结束当前segment
            if label == 1 or i == len(tokens) - 1:
                if current_segment:
                    segments.append(''.join(current_segment))
                    current_segment = []
        
        return segments
    
    def process_text_batch(self, texts: List[str], max_seq_length: int = 128, 
                          eval_batch_size: int = 8) -> List[str]:
        """
        批量处理文本
        
        Args:
            texts: 文本列表
            max_seq_length: 最大序列长度
            eval_batch_size: 批处理大小
            
        Returns:
            List[str]: 分词结果列表
        """
        results = []
        
        # 分批处理
        for i in range(0, len(texts), eval_batch_size):
            batch_texts = texts[i:i + eval_batch_size]
            batch_results = self.predict_batch(batch_texts, max_seq_length)
            
            # 将分词结果转换为字符串
            for segmented_tokens in batch_results:
                result_text = '/'.join(segmented_tokens)
                results.append(result_text)
        
        return results