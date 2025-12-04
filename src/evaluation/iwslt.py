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
        # 1. 在循环开始前，先清空或创建文件（如果需要续写，可以改用 'a' 模式）
        # 使用 'w' 模式会覆盖旧文件，确保从头开始
        with open(result_path, 'w', encoding='utf-8') as f:
            pass 

        for (i, o) in iter(self):
            input_text = "Translate the following texts in Germany into English: " + i
            
            # 推理
            generated = model.generate(input_text, max_length=max_length)
            
            # 处理生成结果（去掉 prompt 部分）
            # 注意：建议加个判断，防止切片报错
            generated_text = generated[len(input_text):] if len(generated) > len(input_text) else generated
            
            current_result = {
                'input': input_text,
                'expected': o,
                'generated': generated_text
            }

            # 2. 核心修改：以追加模式 ('a') 打开文件，写入一条后立即关闭
            # 或者保持文件打开，但使用 flush()
            with open(result_path, 'a', encoding='utf-8') as f:
                # ensure_ascii=False 保证德语字符（如 ä, ö, ü）正常显示，而不是乱码
                f.write(json.dumps(current_result, ensure_ascii=False) + '\n')
                
            # 日志打印
            logger.info(f'{self.offset} / {self.max_len}')
            logger.info(generated_text)


