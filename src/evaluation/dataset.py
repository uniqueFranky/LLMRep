from typing import Protocol

class Dataset(Protocol):
    def __iter__(self):
        pass

    def __next__(self):
        pass


