from .base_model import BaseModel
from ..decode import compute_logits
from .decoder import BaseDecoder
from src.decode import apply_ngram_penalty
import torch
import math

class DecodePenaltyModel(BaseModel):
    def __init__(self, tokenizer, model, penalty_config: dict[int, float], decoder: BaseDecoder):
        self.tokenizer = tokenizer
        self.model = model
        self.penalty_config = penalty_config
        self.decoder = decoder
    
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
                
                penalized_logits = raw_logits.clone()
                if self.penalty_config:
                    for ngram_size, penalty in self.penalty_config.items():
                        penalized_logits = apply_ngram_penalty(generated, self.tokenizer, penalized_logits, ngram_size, penalty)

                # 计算概率分布（用于perplexity计算，使用原始logits）
                log_probs = torch.log_softmax(raw_logits, dim=-1).squeeze(0)

                # 选择下一个token（使用penalized logits）
                next_token_id = self.decoder(penalized_logits.squeeze(0))
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

        if with_prompt is False:
            generated_text = generated[len(input):] if len(generated) > len(input) else generated
        else:
            generated_text = generated

        return generated_text, perplexity
    
    def generate(self, input: str, max_length: int=100) -> str:
        """保持原有接口不变"""
        generated_text, _ = self.generate_with_perplexity(input, max_length)
        return generated_text
