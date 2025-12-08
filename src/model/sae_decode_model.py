from .base_model import BaseModel
import torch
import math
import torch.nn.functional as F

from transformers import AutoTokenizer
from transformer_lens import HookedTransformer
from sae_lens import SAE


class SaeGreedyDecodeModel(BaseModel):

    def __init__(
        self,
        latent_idxs,
        steering_coefficient: float = -0.01,
        sae_release: str = "gpt2-small-res-jb",
        sae_id: str = "blocks.9.hook_resid_pre",
        device: str = "cuda",

        # ===== 新增解码配置（保持 generate_with_perplexity 不变） =====
        decoding_mode: str = "greedy",   # "greedy" or "sample"
        temperature: float = 1.0,
        freq_penalty: float = 0.0,
    ):
        """
        解码风格通过 __init__ 配置，不改变 generate_with_perplexity 的函数签名
        """
        self.device = device
        self.latent_idxs = latent_idxs
        self.steering_coefficient = steering_coefficient

        # decoding config
        self.decoding_mode = decoding_mode
        self.temperature = temperature
        self.freq_penalty = freq_penalty

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
        self.model = HookedTransformer.from_pretrained(
            model_name,
            device=device,
            **model_kwargs,
        )

        # ===== tokenizer =====
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # ===== steering vector =====
        with torch.no_grad():
            steer_vec = torch.zeros(self.sae.W_dec.shape[1], device=device)
            for idx in latent_idxs:
                steer_vec += self.sae.W_dec[idx]
            self.steer_vec = self.steering_coefficient * steer_vec

    # steering hook
    def _steering_hook(self, activations, hook):
        activations = activations + self.steer_vec
        return activations

    # decoding
    @torch.no_grad()
    def _decode_token(self, logits, generated_ids):
        vocab_size = logits.shape[-1]

        if self.decoding_mode == "greedy":
            return torch.argmax(logits).item()

        # frequency penalty
        counts = torch.bincount(generated_ids.view(-1), minlength=vocab_size).float()
        logits = logits - self.freq_penalty * counts.to(logits.device)

        # temperature
        logits = logits / max(self.temperature, 1e-6)

        probs = F.softmax(logits, dim=-1)
        next_token_id = torch.multinomial(probs, 1).item()
        return next_token_id

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

        for _ in range(max_length):

            with self.model.hooks(fwd_hooks=[(self.hook_name, self._steering_hook)]):

                logits = self.model(generated_ids)
                next_logits = logits[0, -1, :]
                log_probs = torch.log_softmax(next_logits, dim=-1)

            # decode next token using mode set in init
            next_token_id = self._decode_token(next_logits, generated_ids)

            total_log_prob += log_probs[next_token_id].item()
            num_tokens += 1

            next_token_tensor = torch.tensor([[next_token_id]], device=self.device)
            generated_ids = torch.cat([generated_ids, next_token_tensor], dim=1)

            if next_token_id == self.tokenizer.eos_token_id:
                break

        avg_log_prob = total_log_prob / max(num_tokens, 1)
        ppl = math.exp(-avg_log_prob)

        text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return text, ppl

    def generate(self, input: str, max_length: int):
        text, _ = self.generate_with_perplexity(input, max_length)
        return text
