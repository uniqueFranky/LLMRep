from typing import List, Union

def rep_w(tokens: Union[List[int], List[str]], w: int) -> float:
    """
    tokens: token_id 或 token本身
    w: 窗口长度
    """
    rep = 0
    for i in range(len(tokens)):
        if tokens[i] in tokens[max(0, i - w): i]:
            rep += 1
    return rep / len(tokens)


def rep_n(tokens: Union[List[int], List[str]], n: int) -> float:
    """
    tokens: token_id 或 token本身
    n: n-gram
    """
    if len(tokens) < n:
        return 0.0
    grams = [tuple(tokens[i: i + n]) for i in range(len(tokens) - n + 1)]
    unique_grams = set(grams)
    return 1 - len(unique_grams) / (len(tokens) - n + 1)


def rep_r(tokens: Union[List[int], List[str]]) -> float:
    rep = 0
    for i in range(len(tokens)):
        for j in range(len(tokens)):
            if i == j:
                continue
            if tokens[i] == tokens[j] and ((i > 0 and j > 0 and tokens[i - 1] == tokens[j - 1]) or (j + 1 < len(tokens) and i + 1 < len(tokens) and tokens[i + 1] == tokens[j + 1])):
                rep += 1
                break
    return rep / len(tokens)


