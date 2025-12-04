from src.evaluation.iwslt import IWSLTDataset
from src.model.greedy_decode_model import GreedyDecodeModel
import logging
from datasets import load_dataset

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def test_evaluate(gemma2_model, gemma2_tokenizer):
    dataset = "YokyYao/Diversity_Challenge"
    ds = load_dataset(dataset)
    question_list = ds['train']['question']
    model = GreedyDecodeModel(gemma2_tokenizer, gemma2_model)
    for q in question_list:
        a = model.generate(q, max_length=200)
        logger.info(f"Q: {q}\nA: {a}\n")