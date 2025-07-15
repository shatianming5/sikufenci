from typing import List, Dict


class SimpleTokenizer:
    """
    简单的分词器实现，用于在没有预训练模型时进行基本分词
    """
    
    def __init__(self):
        # 基本的繁体中文标点符号
        self.punctuation = set('，。！？：；""''（）【】《》「」『』〈〉')
        
    def tokenize(self, text: str) -> List[str]:
        """
        简单分词
        """
        tokens = []
        current_token = ""
        
        for char in text:
            if char in self.punctuation or char.isspace():
                if current_token:
                    tokens.append(current_token)
                    current_token = ""
                if not char.isspace():
                    tokens.append(char)
            else:
                current_token += char
        
        if current_token:
            tokens.append(current_token)
            
        return tokens
    
    def segment_text(self, text: str) -> str:
        """
        对文本进行分词处理
        """
        tokens = self.tokenize(text)
        return '/'.join(tokens)