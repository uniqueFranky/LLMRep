from .base_model import BaseModel
from ..decode import compute_logits
import torch
import math
import torch.nn.functional as F


class TopKDecodeModel(BaseModel):
    """
    Top-K + Temperature sampling 解码器
    - 与 GreedyDecodeModel 完全兼容
    - generate_with_perplexity 返回 (generated_text, perplexity)
    """

    def __init__(self, tokenizer, model, top_k: int = 5, temperature: float = 1.0):
        self.tokenizer = tokenizer
        self.model = model
        self.top_k = top_k
        self.temperature = temperature


    def sample_top_k(self, logits: torch.Tensor, top_k: int, temperature: float):
        """从 top-k 分布中按 temperature 采样"""

        # temperature scaling
        if temperature > 0:
            logits = logits / temperature
        else:
            # temperature = 0 → 等价 greedy
            return torch.argmax(logits).item()

        # top-k 保留最高 k 个 logits
        if top_k > 0:
            # 找到 top_k 阈值：超过者保留，不足者设为 -inf
            top_k_values, _ = torch.topk(logits, top_k)
            threshold = top_k_values[-1]
            logits = torch.where(logits >= threshold, logits, torch.tensor(float('-inf')).to(logits.device))

        # softmax
        probs = F.softmax(logits, dim=-1)

        # multinomial sampling
        next_token_id = torch.multinomial(probs, num_samples=1).item()

        return next_token_id


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

                # 用 top-k + temperature 采样
                next_token_id = self.sample_top_k(
                    logits,
                    top_k=self.top_k,
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
            perplexity = float('inf')

        # remove the prompt if needed
        if with_prompt is False:
            generated_text = generated[len(input):] if len(generated) > len(input) else generated
        else:
            generated_text = generated

        return generated_text, perplexity


    def generate(self, input: str, max_length: int = 100) -> str:
        generated_text, _ = self.generate_with_perplexity(input, max_length)
        return generated_text
