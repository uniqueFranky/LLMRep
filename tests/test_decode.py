from src.decode import compute_logits, apply_ngram_penalty
from src.metrics import rep_n, rep_r, rep_w
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def test_gpt2_get_logits(gpt2_model, gpt2_tokenizer, input_text):
    from src.decode import compute_logits

    logits = compute_logits(gpt2_model, gpt2_tokenizer, input_text)
    assert logits.dim() == 2
    assert logits.shape[0] == 1
    assert logits.shape[1] == 50257

def test_gpt2_ngram_penalty(gpt2_model, gpt2_tokenizer, input_text_with_3gram_rep):
    # 1. 计算原始 Logits
    logits = compute_logits(gpt2_model, gpt2_tokenizer, input_text_with_3gram_rep)

    logits_before = logits.clone()

    # 2. 应用惩罚
    # 设定 n=3 (3-gram), lambda=0.9 (惩罚力度较轻)
    new_logits = apply_ngram_penalty(input_text_with_3gram_rep, gpt2_tokenizer, logits, n=3, lmbda=0.9)

    # --- 3. 检查逻辑 ---
    
    # A. 手动找出应该被惩罚的 Token
    # n=3, 说明我们要看最后 2 个词 (prefix) 在历史上接了什么
    input_ids = gpt2_tokenizer.encode(input_text_with_3gram_rep, add_special_tokens=False)
    prefix_len = 3 - 1
    prefix = tuple(input_ids[-prefix_len:]) 
    
    target_tokens = set()

    # 扫描历史找出重复模式
    for i in range(len(input_ids) - prefix_len):
        if tuple(input_ids[i : i + prefix_len]) == prefix:
            target_tokens.add(input_ids[i + prefix_len])
            
    assert len(target_tokens) > 0

    
    for tid in target_tokens:
        old_val = logits_before[0, tid].item()
        new_val = new_logits[0, tid].item()
        
        assert new_val < old_val


def test_greedy_decode_with_ngram_penalty(gpt2_model, gpt2_tokenizer):
    from src.decode import greedy_decode_with_ngram_penalty, greedy_decode_without_penalty

    prompt = "Once upon a time"
    max_length = 100
    n = 2
    lmbda = 0.01

    generated_text_with_penalty = greedy_decode_with_ngram_penalty(
        gpt2_model, gpt2_tokenizer, prompt, max_length, n, lmbda
    )

    assert isinstance(generated_text_with_penalty, str)
    assert len(generated_text_with_penalty) > len(prompt)

    generated_text_without_penalty = greedy_decode_without_penalty(
        gpt2_model, gpt2_tokenizer, prompt, max_length
    )

    assert isinstance(generated_text_without_penalty, str)
    assert len(generated_text_without_penalty) > len(prompt)

    tokens_with_penalty = gpt2_tokenizer.encode(generated_text_with_penalty, add_special_tokens=False)
    tokens_without_penalty = gpt2_tokenizer.encode(generated_text_without_penalty, add_special_tokens=False)

    assert rep_w(tokens_with_penalty, 20) < rep_w(tokens_without_penalty, 20)
    assert rep_r(tokens_with_penalty) < rep_r(tokens_without_penalty)
    assert rep_n(tokens_with_penalty, 5) < rep_n(tokens_without_penalty, 5)

