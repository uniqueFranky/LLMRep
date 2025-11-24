import torch
import math
from collections import Counter


def compute_logits(model, tokenizer, text):
    # ⚠️ 兼容性处理：GPT-2 默认没有 pad_token，需要手动指定，否则批处理会报错
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 准备输入
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    # 获取 Logits
    with torch.no_grad():
        outputs = model(**inputs)
        next_token_logits = outputs.logits[:, -1, :]
        return next_token_logits


def apply_ngram_penalty(input_text: str, tokenizer, logits: torch.Tensor, n: int, lmbda: float) -> torch.Tensor:
    """
    计算即将生成的 token 若构成重复 N-gram，则对其概率进行 lambda^k 的惩罚。
    
    Args:
        input_text: 当前已生成的完整历史文本 (str)
        tokenizer:用于将 input_text 转为 input_ids 的 tokenizer 对象
        logits: 模型输出的 logits [1, vocab_size]
        n: n-gram 的大小 (例如 n=3 表示 3-gram)
        lmbda: 衰减系数 (0 < lambda < 1)，越小惩罚越重
        
    Returns:
        修正后的 logits
    """
    
    # --- 1. 文本转 ID (Tokenization) ---
    # 必须使用 add_special_tokens=False，防止在文本中间插入 [BOS]/[CLS] 干扰匹配
    input_ids = tokenizer.encode(input_text, add_special_tokens=False)
    
    # --- 2. 边界检查 ---
    # 如果历史长度甚至不足以构成一个 n-gram 前缀，直接返回
    if len(input_ids) < n - 1:
        return logits

    # --- 3. 确定前缀 (Context Suffix) ---
    # 我们需要看历史文本的最后 n-1 个 token
    # 比如 n=3, text="A B C A B", prefix 就是 "A B"
    prefix_len = n - 1
    prefix = tuple(input_ids[-prefix_len:])
    
    # --- 4. 扫描历史，统计重复次数 k ---
    # 目标：找出历史中所有出现过 prefix 的位置，并记录它们后面接了哪个词
    next_token_counts = Counter()
    
    # 遍历范围说明：
    # 我们只需要遍历到倒数第 (n-1) 个位置之前。
    # 因为最后一次出现肯定是当前结尾本身，不需要统计（那是我们要预测的位置）。
    search_limit = len(input_ids) - prefix_len
    
    for i in range(search_limit):
        # 检查当前片段是否与前缀匹配
        if tuple(input_ids[i : i + prefix_len]) == prefix:
            # 如果匹配，取出紧接着的下一个 token ID
            target_token_id = input_ids[i + prefix_len]
            next_token_counts[target_token_id] += 1
            
    # --- 5. 应用惩罚 (Logits Space) ---
    # 原理推导：
    # P_new = P_old * (lambda^k)
    # log(P_new) = log(P_old) + k * log(lambda)
    # Logits 是 log(P) 的未归一化形式，直接加即可
    
    if not next_token_counts:
        return logits
        
    log_lmbda = math.log(lmbda)
    
    for token_id, k in next_token_counts.items():
        # 注意：logits 通常是 [batch_size, vocab_size]，这里假设 batch=1
        logits[0, token_id] += k * log_lmbda
        
    return logits


