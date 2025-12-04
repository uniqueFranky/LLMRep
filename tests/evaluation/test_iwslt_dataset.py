from src.evaluation.iwslt import IWSLTDataset
from src.model.greedy_decode_model import GreedyDecodeModel
from src.model.decode_penalty_model import DecodePenaltyModel
from src.decode import apply_ngram_penalty
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def test_iteration():
    dataset = IWSLTDataset()
    for (i, o) in dataset:
        assert len(i) > 0
        assert len(o) > 0
        

def test_evaluate(gemma2_model, gemma2_tokenizer):
    dataset = IWSLTDataset()
    model = DecodePenaltyModel(gemma2_tokenizer, gemma2_model, penalty_func=apply_ngram_penalty)
    dataset.evaluate(model, result_path='/data/data_public/yrb/LLMRep/results/iwsl-decode.jsonl', max_length=200)

