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