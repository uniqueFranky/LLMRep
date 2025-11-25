import dotenv
import os
from .dataset import Dataset

dotenv.load_dotenv()

class WikitextDataset(Dataset):
    def __init__(self, input_window=100, output_window=100, split='test', dir=None):
        if dir is None:
            dir = os.path.join(os.getenv('DATASET_DIR'), 'wikitext-103')
        self.input_window = input_window
        self.output_window = output_window
        
        file_path = os.path.join(dir, f'wiki.{split}.tokens')
        with open(file_path, 'r', encoding='utf-8') as file:
            self.texts = file.read()
        self.tokens = self.texts.split(' ')
        self.offset = 0
        self.max_len = len(self.tokens)

    def __iter__(self):
        return self
    
    def __next__(self) -> tuple[str, str]:

        if self.offset + self.input_window + self.output_window > self.max_len:
            raise StopIteration
        text_in = ' '.join(self.tokens[self.offset: self.offset + self.input_window])
        text_out = ' '.join(self.tokens[self.offset + self.input_window: self.offset + self.input_window + self.output_window])
        self.offset += 1
        return (text_in, text_out)


