import torch

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


