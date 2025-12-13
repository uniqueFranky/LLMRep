from .base_model import BaseModel
import torch
import math
import torch.nn.functional as F

from transformers import AutoTokenizer
from sae_lens import SAE, HookedSAETransformer
from src.decode import apply_ngram_penalty
from .decoder import BaseDecoder

class SaeDecodePenaltyModel(BaseModel):

    def __init__(
        self,
        decoder: BaseDecoder,
        tokenizer: AutoTokenizer,
        latent_idxs,
        penalty_config: dict[int, float]=None,
        steering_coefficient: float = -4,
        sae_release: str = "gpt2-small-res-jb",
        sae_id: str = "blocks.9.hook_resid_pre",
        device: str = "cuda",
    ):
        """
        解码风格通过 __init__ 配置，不改变 generate_with_perplexity 的函数签名
        """
        self.decoder = decoder
        self.penalty_config = penalty_config
        self.device = device
        self.latent_idxs = latent_idxs
        self.steering_coefficient = steering_coefficient

        # ===== load SAE =====
        print("Loading SAE...")
        self.sae, cfg_dict, sparsity = SAE.from_pretrained(
            release=sae_release,
            sae_id=sae_id,
            device=device,
        )
        print("SAE loaded.")

        self.hook_name = self.sae.cfg.metadata["hook_name"]
        model_name = self.sae.cfg.metadata.get("model_name", "gpt2-small")
        model_kwargs = self.sae.cfg.metadata.get("model_from_pretrained_kwargs", {}) or {}

        # ===== load transformer =====
        self.model = HookedSAETransformer.from_pretrained(
            model_name,
            device=device,
            **model_kwargs,
        )

        # ===== tokenizer =====

        self.tokenizer = tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token


    # steering hook
    def _steering_hook(self, activations, hook):
        for latent_idx in self.latent_idxs:
            activations += self.steering_coefficient * self.sae.W_dec[latent_idx]

    # ====== your required function signature ======
    @torch.no_grad()
    def generate_with_perplexity(self, input: str, max_length: int, with_prompt: bool=False) -> tuple[str, float]:
        """
        签名不能改变：仅 input, max_length
        解码风格完全依赖 __init__ 中的配置（greedy vs sampling）
        """

        enc = self.tokenizer(input, return_tensors="pt").to(self.device)
        generated_ids = enc["input_ids"]

        total_log_prob = 0.0
        num_tokens = 0

        for step in range(max_length):

            with self.model.hooks(fwd_hooks=[(self.hook_name, self._steering_hook)]):

                logits = self.model(generated_ids)
                next_logits = logits[0, -1, :]
                log_probs = torch.log_softmax(next_logits, dim=-1)

            if self.penalty_config is not None:
                for ngram_size, penalty in self.penalty_config.items():
                    next_logits = apply_ngram_penalty(
                        self.tokenizer.decode(generated_ids[0], skip_special_tokens=True),
                        self.tokenizer,
                        next_logits,
                        ngram_size,
                        penalty,
                    )
            
            # decode next token using mode set in init
            next_token_id = self.decoder(next_logits)

            if step > 0:
                total_log_prob += log_probs[next_token_id].item()
                num_tokens += 1

            next_token_tensor = torch.tensor([[next_token_id]], device=self.device)
            generated_ids = torch.cat([generated_ids, next_token_tensor], dim=1)

            if next_token_id == self.tokenizer.eos_token_id:
                break

        if num_tokens > 0:
            avg_log_prob = total_log_prob / num_tokens
            ppl = math.exp(-avg_log_prob)
        else:
            ppl = float('inf')

        text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        if with_prompt is False:
            text = text[len(input):]
        return text, ppl

    def generate(self, input: str, max_length: int):
        text, _ = self.generate_with_perplexity(input, max_length)
        return text
