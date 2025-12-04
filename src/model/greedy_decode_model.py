from .base_model import BaseModel
from ..decode import compute_logits
import torch

class GreedyDecodeModel(BaseModel):
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model

    
    def generate(self, input: str, max_length: int=100) -> str:
        generated = input
        with torch.no_grad():
            for _ in range(max_length):
                logits = compute_logits(self.model, self.tokenizer, generated)
                next_token_id = torch.argmax(logits, dim=-1).item()
                next_token = self.tokenizer.decode([next_token_id])
                
                generated += next_token
                
                if next_token_id == self.tokenizer.eos_token_id:
                    break
                    
            return generated

