import dotenv
import os
from .dataset import Dataset
from ..model.base_model import BaseModel
import json
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

dotenv.load_dotenv()


class IWSLTDataset(Dataset):
    def __init__(self, split='test', dir=None):
        if dir is None:
            dir = os.path.join(os.getenv('DATASET_DIR'), 'iwslt')
        
        en_file_path = os.path.join(dir, f'{split}.en')
        with open(en_file_path, 'r', encoding='utf-8') as file:
            self.en_texts = file.readlines()

        de_file_path = os.path.join(dir, f'{split}.de')
        with open(de_file_path, 'r', encoding='utf-8') as file:
            self.de_texts = file.readlines()
        
        self.offset = 0
        self.max_len = len(self.en_texts)

    def __iter__(self):
        self.offset = 0
        return self
    
    def __next__(self) -> tuple[str, str]:
        if self.offset >= self.max_len:
            raise StopIteration
        self.offset += 1
        return (self.de_texts[self.offset - 1], self.en_texts[self.offset - 1])
    
    def evaluate(self, model: BaseModel, result_path: str, max_length: int=500):
        result = []
        for (i, o) in iter(self):
            input = "Translate the following texts in Germany into English: " + i
            generated = model.generate(input, max_length=max_length)
            result.append({
                'input': input,
                'expected': o,
                'generated': generated[len(input): ]
            })
            logger.info(f'{self.offset} / {self.max_len}')
            logger.info(generated[len(input): ])

        with open(result_path, 'w') as file:
            json.dump(result, file)



