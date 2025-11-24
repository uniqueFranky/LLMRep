from pytest import fixture


@fixture(scope='session')
def tokens_without_rep():
    return ['I', 'like', 'apple']

@fixture(scope='session')
def tokens_with_3gram_rep():
    return ['I', 'like', 'apple', 'I', 'like', 'apple', 'I', 'like', 'apple']


@fixture(scope='session')
def gpt2_model():
    from transformers import AutoModelForCausalLM
    import torch

    return AutoModelForCausalLM.from_pretrained(
        'gpt2',
        device_map="auto", 
        torch_dtype=torch.bfloat16, # 推荐用于 Llama 3.1 和 Gemma 2
        # token="YOUR_HF_TOKEN"     # 如果未在终端登录，需在此填入 HF Token
    )

@fixture(scope='session')
def gpt2_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained('gpt2')

@fixture(scope='session')
def input_text():
    return 'I like apples'