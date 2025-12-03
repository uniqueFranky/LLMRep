from src.evaluation.iwslt import IWSLTDataset
from src.model.greedy_decode_model import GreedyDecodeModel
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
    model = GreedyDecodeModel(gemma2_tokenizer, gemma2_model)
    dataset.evaluate(model, result_path='/data/data_public/yrb/LLMRep/results/iwsl.json', max_length=200)

