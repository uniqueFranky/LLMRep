from src.model.decoder import GreedyDecoder, TopKDecoder, TopPDecoder
import torch


def test_greedy_decoder():
    decoder = GreedyDecoder()
    logits = torch.tensor([0.1, 0.5, 0.2, 0.9])
    predicted_index = decoder(logits)
    assert predicted_index == 3  # The index of the highest logit


def test_topk_decoder():
    decoder = TopKDecoder(k=2)
    logits = torch.tensor([0.1, 0.5, 0.2, 0.9])
    predicted_index = decoder(logits)
    assert predicted_index in [1, 3]  # Should be one of the top-2 indices


def test_topp_decoder():
    decoder = TopPDecoder(p=0.9)
    logits = torch.tensor([0.1, 0.5, 0.2, 0.9])
    predicted_index = decoder(logits)
    assert predicted_index == 3  # Should be one of the indices with cumulative probability <= 0.9