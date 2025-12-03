from typing import Protocol
from src.model.base_model import BaseModel

class Dataset(Protocol):
    def __iter__(self):
        pass

    def __next__(self):
        pass

    def evaluate(self, model: BaseModel, result_path: str, max_length):
        pass

