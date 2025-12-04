from .base_model import BaseModel
from ..decode import compute_logits
import torch
import math

class GreedyDecodeModel(BaseModel):
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model

    
    def generate_with_perplexity(self, input: str, max_length: int=100) -> tuple[str, float]:
        """
        生成文本并同时计算perplexity
        返回: (generated_text, perplexity)
        """
        generated = input
        total_log_prob = 0.0
        num_tokens = 0
        
        with torch.no_grad():
            for step in range(max_length):
                logits = compute_logits(self.model, self.tokenizer, generated)
                
                # 处理batch维度：logits形状是 [batch_size, vocab_size] 或 [vocab_size]
                if logits.dim() == 2:
                    logits = logits[0]  # 取第一个batch，得到 [vocab_size]
                
                # 计算概率分布
                log_probs = torch.log_softmax(logits, dim=-1)
                
                # 选择下一个token
                next_token_id = torch.argmax(logits, dim=-1).item()
                next_token = self.tokenizer.decode([next_token_id])
                
                # 累积log probability（跳过第一步）
                if step > 0:
                    total_log_prob += log_probs[next_token_id].item()
                    num_tokens += 1
                
                generated += next_token
                
                if next_token_id == self.tokenizer.eos_token_id:
                    break
        
        # 计算perplexity
        if num_tokens > 0:
            avg_log_prob = total_log_prob / num_tokens
            perplexity = math.exp(-avg_log_prob)
        else:
            perplexity = float('inf')
        
        return generated, perplexity
    
    def generate(self, input: str, max_length: int=100) -> str:
        """保持原有接口不变"""
        generated_text, _ = self.generate_with_perplexity(input, max_length)
        return generated_text

