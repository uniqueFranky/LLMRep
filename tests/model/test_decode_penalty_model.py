from src.model.decode_penalty_model import DecodePenaltyModel
from src.decode import apply_ngram_penalty
import pytest

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

@pytest.mark.parametrize("model_name, tokenizer_name", [
    ('gpt2_model', 'gpt2_tokenizer'),
    ('gemma2_model', 'gemma2_tokenizer'),
    # ('llama3_model', 'llama3_tokenizer') # TODO: llama is now unavailable due to restriction to China
])
def test_decode_penalty_model(model_name, tokenizer_name, request):
    model = request.getfixturevalue(model_name)
    tokenizer = request.getfixturevalue(tokenizer_name)

    decode_penalty_model = DecodePenaltyModel(tokenizer, model, penalty_func=apply_ngram_penalty)
    generated = decode_penalty_model.generate('Once upon a time')
    logger.info(generated)

