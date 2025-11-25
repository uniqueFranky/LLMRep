from src.evaluation.iwslt import IWSLTDataset


def test_iteration():
    dataset = IWSLTDataset()
    for (i, o) in dataset:
        assert len(i) > 0
        assert len(o) > 0
        

