from pytest import fixture
import dotenv
import os

dotenv.load_dotenv()

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
def gemma2_model():
    from transformers import AutoModelForCausalLM
    import torch

    return AutoModelForCausalLM.from_pretrained(
        'google/gemma-2-2b',
        device_map="auto", 
        torch_dtype=torch.bfloat16, # 推荐用于 Llama 3.1 和 Gemma 2
        token=os.getenv('HF_TOKEN')
    )

@fixture(scope='session')
def gemma2_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained("google/gemma-2-2b")


@fixture(scope='session')
def llama3_model():
    from transformers import AutoModelForCausalLM
    import torch

    return AutoModelForCausalLM.from_pretrained(
        'meta-llama/Llama-3.1-8B',
        device_map="auto", 
        torch_dtype=torch.bfloat16, # 推荐用于 Llama 3.1 和 Gemma 2
        token=os.getenv('HF_TOKEN')
    )

@fixture(scope='session')
def llama3_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")



@fixture(scope='session')
def input_text():
    return 'I like apples'

@fixture(scope='session')
def input_text_with_3gram_rep():
    return 'I like apples I like apples I like apples I like apples I like apples'