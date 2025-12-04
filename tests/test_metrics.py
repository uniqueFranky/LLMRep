import logging
import pytest

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def test_rep_w(tokens_without_rep, tokens_with_3gram_rep):
    from src.metrics import rep_w

    assert rep_w(tokens_without_rep, 3) == 0

    assert rep_w(tokens_with_3gram_rep, 3) == 6 / 9


def test_rep_n(tokens_without_rep, tokens_with_3gram_rep):
    from src.metrics import rep_n

    assert rep_n(tokens_without_rep, 3) == 0

    assert rep_n(tokens_with_3gram_rep, 3) == 1 - (3 / 7)


def test_rep_r(tokens_without_rep, tokens_with_3gram_rep):
    from src.metrics import rep_r

    assert rep_r(tokens_without_rep) == 0

    assert rep_r(tokens_with_3gram_rep) == 1


def test_perplexity():
    from src.metrics import perplexity
    assert perplexity("The cat is on the mat.", "gpt2") < perplexity("hello world", "gpt2")


def test_bleu():
    from src.metrics import bleu

    pred = "The cat is on the mat."
    gold = "The cat is on the mat."

    score = bleu(pred, gold)
    assert score == pytest.approx(100.0)

    pred2 = "The cat is on mat."
    score2 = bleu(pred2, gold)
    assert score2 < 100.0


def test_meteor():
    from src.metrics import meteor

    pred = "The cat is on the mat."
    gold = "The cat is on the mat."

    score = meteor(pred, gold)
    assert score >= 0.99

    pred2 = "The cat is on mat."
    score2 = meteor(pred2, gold)
    assert score2 < 0.9