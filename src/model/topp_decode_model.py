from .base_model import BaseModel
from ..decode import compute_logits
import torch
import torch.nn.functional as F
import math


class TopPDecodeModel(BaseModel):
    """
    Top-p (Nucleus) + Temperature sampling 解码器
    - 与 GreedyDecodeModel 完全兼容
    """

    def __init__(self, tokenizer, model, top_p: float = 0.9, temperature: float = 1.0):
        self.tokenizer = tokenizer
        self.model = model
        self.top_p = top_p
        self.temperature = temperature

    # ------------------------------------------------------------
    # nucleus sampling (top-p)
    # ------------------------------------------------------------
    def sample_top_p(self, logits: torch.Tensor, top_p: float, temperature: float):
        """
        从 top-p 分布中按 temperature 采样
        """

        # temperature scaling
        if temperature > 0:
            logits = logits / temperature
        else:
            return torch.argmax(logits).item()

        # softmax
        probs = F.softmax(logits, dim=-1)

        # sort tokens by probability
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)

        # cumulative sum
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # mask tokens to keep only those within top-p threshold
        mask = cumulative_probs <= top_p
        mask[0] = True  # keep at least one token

        filtered_probs = sorted_probs * mask
        filtered_probs = filtered_probs / filtered_probs.sum()

        # multinomial sampling
        sampled_idx = torch.multinomial(filtered_probs, 1).item()

        # map back to original token id
        next_token_id = sorted_indices[sampled_idx].item()

        return next_token_id

    # ------------------------------------------------------------
    # generate_with_perplexity
    # ------------------------------------------------------------
    def generate_with_perplexity(self, input: str, max_length: int = 100, with_prompt: bool = False):
        """
        返回 (generated_text, perplexity)
        """
        generated = input
        total_log_prob = 0.0
        num_tokens = 0

        with torch.no_grad():
            for step in range(max_length):

                logits = compute_logits(self.model, self.tokenizer, generated)
                if logits.dim() == 2:
                    logits = logits[0]

                # nucleus sampling
                next_token_id = self.sample_top_p(
                    logits,
                    top_p=self.top_p,
                    temperature=self.temperature
                )

                next_token = self.tokenizer.decode([next_token_id])

                # perplexity accumulation
                log_probs = F.log_softmax(logits, dim=-1)
                if step > 0:
                    total_log_prob += log_probs[next_token_id].item()
                    num_tokens += 1

                generated += next_token

                # eos
                if next_token_id == self.tokenizer.eos_token_id:
                    break

        # perplexity
        if num_tokens > 0:
            avg_log_prob = total_log_prob / num_tokens
            perplexity = math.exp(-avg_log_prob)
        else:
            perplexity = float("inf")

        # remove prompt
        if with_prompt is False:
            generated_text = generated[len(input):] if len(generated) > len(input) else generated
        else:
            generated_text = generated

        return generated_text, perplexity

    # ------------------------------------------------------------
    # generate
    # ------------------------------------------------------------
    def generate(self, input: str, max_length: int = 100) -> str:
        generated_text, _ = self.generate_with_perplexity(input, max_length)
        return generated_text
