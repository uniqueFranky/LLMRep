from src.evaluation.wikitext import WikitextDataset


def test_iteration():
    dataset = WikitextDataset(input_window=100, output_window=100)
    for (i, o) in dataset:
        assert len(i.split(' ')) == 100
        assert len(o.split(' ')) == 100
        

