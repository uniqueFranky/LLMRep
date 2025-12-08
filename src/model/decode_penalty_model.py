from .base_model import BaseModel
from ..decode import compute_logits
import torch
import math

class DecodePenaltyModel(BaseModel):
    def __init__(self, tokenizer, model, penalty_func, n=3, lmda=0.1):
        self.tokenizer = tokenizer
        self.model = model
        self.penalty_func = penalty_func
        self.n = n
        self.lmda = lmda
    
    def generate_with_perplexity(self, input: str, max_length: int=100, with_prompt: bool=False) -> tuple[str, float]:
        """
        生成文本并同时计算perplexity
        返回: (generated_text, perplexity)
        """
        generated = input
        total_log_prob = 0.0
        num_tokens = 0
        
        # 获取输入的token数量
        input_tokens = self.tokenizer.encode(input)
        input_length = len(input_tokens)
        
        with torch.no_grad():
            for step in range(max_length):
                # 计算当前序列的logits
                raw_logits = compute_logits(self.model, self.tokenizer, generated)
                
                # 确保logits是2维的，用于penalty函数
                if raw_logits.dim() == 1:
                    logits_2d = raw_logits.unsqueeze(0)  # [vocab_size] -> [1, vocab_size]
                else:
                    logits_2d = raw_logits
                
                # 应用penalty（使用2维logits）
                penalized_logits_2d = self.penalty_func(generated, self.tokenizer, logits_2d, self.n, self.lmda)
                
                # 转换回1维用于后续计算
                if penalized_logits_2d.dim() == 2:
                    penalized_logits = penalized_logits_2d.squeeze(0)
                    original_logits = logits_2d.squeeze(0)
                else:
                    penalized_logits = penalized_logits_2d
                    original_logits = raw_logits
                
                # 计算概率分布（用于perplexity计算，使用原始logits）
                log_probs = torch.log_softmax(original_logits, dim=-1)
                
                # 选择下一个token（使用penalized logits）
                next_token_id = torch.argmax(penalized_logits, dim=-1).item()
                next_token = self.tokenizer.decode([next_token_id])
                
                # 累积log probability
                current_tokens = self.tokenizer.encode(generated)
                if len(current_tokens) > input_length:
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

        generated_text = generated[len(input):] if len(generated) > len(input) else generated        
        return generated_text, perplexity
    
    def generate(self, input: str, max_length: int=100) -> str:
        """保持原有接口不变"""
        generated_text, _ = self.generate_with_perplexity(input, max_length)
        return generated_text
