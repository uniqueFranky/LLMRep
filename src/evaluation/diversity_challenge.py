from datasets import load_dataset
from .dataset import Dataset
from src.model.base_model import BaseModel
import json
import logging
import argparse

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)



class DiversityChallengeDataset(Dataset):
    def __init__(self, split: str = "train"):
        self.dataset = load_dataset("YokyYao/Diversity_Challenge", split=split)
        self.offset = 0
        self.max_len = len(self.dataset)

    def __iter__(self):
        self.offset = 0
        return self
    
    def __next__(self) -> tuple[str, str]:
        if self.offset >= self.max_len:
            raise StopIteration
        
        item = self.dataset[self.offset]
        self.offset += 1
        
        return (item['question'], "")
    
    def evaluate(self, model: BaseModel, result_path: str, max_length: int=500, max_samples: int=-1):
        # 1. 在循环开始前，先清空或创建文件（如果需要续写，可以改用 'a' 模式）
        # 使用 'w' 模式会覆盖旧文件，确保从头开始
        with open(result_path, 'w', encoding='utf-8') as f:
            pass 
        cnt = 0
        for (i, o) in iter(self):
            cnt += 1
            if max_samples > 0 and cnt > max_samples:
                break
            input_text = i
            
            # 推理
            generated, ppl = model.generate_with_perplexity(input_text, max_length=max_length, with_prompt=True)
        
            current_result = {
                'input': input_text,
                'expected': o,
                'generated': generated,
                'perplexity': ppl
            }

            # 2. 核心修改：以追加模式 ('a') 打开文件，写入一条后立即关闭
            # 或者保持文件打开，但使用 flush()
            with open(result_path, 'a', encoding='utf-8') as f:
                # ensure_ascii=False 保证德语字符（如 ä, ö, ü）正常显示，而不是乱码
                f.write(json.dumps(current_result, ensure_ascii=False) + '\n')
                
            # 日志打印
            logger.info(f'{self.offset} / {min(self.max_len, max_samples if max_samples > 0 else self.max_len)}')
            logger.info(generated)
