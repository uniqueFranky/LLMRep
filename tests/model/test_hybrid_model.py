from src.model.hybrid_model import HybridModel
from src.model.decoder import TopKDecoder
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def test_hybrid_model_generate():
    model = AutoModelForCausalLM.from_pretrained(
        'gpt2',
        device_map='cuda:4', 
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    decoder = TopKDecoder(k=3)
    hybrid_model = HybridModel(
        model_name='gpt2',
        model=model,
        tokenizer=tokenizer,
        decoder=decoder,
        use_decode_penalty=True,
        use_sae_depression=True,
        use_neuron_depression=True,
        device='cuda:4'
    )
    logger.info(hybrid_model.generate("Once upon a time", max_length=200))


def test_hybrid_decode_penalty_and_sae_depression():
    model = AutoModelForCausalLM.from_pretrained(
        'gpt2',
        device_map='cuda:4', 
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    decoder = TopKDecoder(k=3)
    hybrid_model = HybridModel(
        model_name='gpt2',
        model=model,
        tokenizer=tokenizer,
        decoder=decoder,
        use_decode_penalty=True,
        use_sae_depression=True,
        use_neuron_depression=False,
        device='cuda:4'
    )
    logger.info(hybrid_model.generate("Once upon a time", max_length=200))


def test_hybrid_decode_penalty_and_neuron_depression():
    model = AutoModelForCausalLM.from_pretrained(
        'gpt2',
        device_map='cuda:4', 
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    decoder = TopKDecoder(k=3)
    hybrid_model = HybridModel(
        model_name='gpt2',
        model=model,
        tokenizer=tokenizer,
        decoder=decoder,
        use_decode_penalty=True,
        use_sae_depression=False,
        use_neuron_depression=True,
        device='cuda:4'
    )
    logger.info(hybrid_model.generate("Once upon a time", max_length=200))


def test_hybrid_sae_depression_and_neuron_depression():
    model = AutoModelForCausalLM.from_pretrained(
        'gpt2',
        device_map='cuda:4', 
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    decoder = TopKDecoder(k=3)
    hybrid_model = HybridModel(
        model_name='gpt2',
        model=model,
        tokenizer=tokenizer,
        decoder=decoder,
        use_decode_penalty=False,
        use_sae_depression=True,
        use_neuron_depression=True,
        device='cuda:4'
    )
    logger.info(hybrid_model.generate("Once upon a time", max_length=200))