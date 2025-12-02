from .base_model import BaseModel
from ..decode import compute_logits
import torch

class DecodePenaltyModel(BaseModel):
    def __init__(self, tokenizer, model, penalty_func, n=3, lmda=0.1):
        self.tokenizer = tokenizer
        self.model = model
        self.penalty_func = penalty_func # def apply_ngram_penalty(input_text: str, tokenizer, logits: torch.Tensor, n: int, lmbda: float) -> torch.Tensor:
        self.n = n
        self.lmda = lmda
    
    def generate(self, input: str, max_length: int=100) -> str:
        generated = input
    
        for _ in range(max_length):
            logits = compute_logits(self.model, self.tokenizer, generated)
            logits = self.penalty_func(generated, self.tokenizer, logits, self.n, self.lmda)
            
            next_token_id = torch.argmax(logits, dim=-1).item()
            next_token = self.tokenizer.decode([next_token_id])
            
            generated += next_token
            
            if next_token_id == self.tokenizer.eos_token_id:
                break
                
        return generated