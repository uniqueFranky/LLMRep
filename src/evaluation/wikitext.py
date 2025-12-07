import dotenv
import os
from .dataset import Dataset
from ..model.base_model import BaseModel

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
        self.offset = 0
        return self
    
    def __next__(self) -> tuple[str, str]:

        if self.offset + self.input_window + self.output_window > self.max_len:
            raise StopIteration
        text_in = ' '.join(self.tokens[self.offset: self.offset + self.input_window])
        text_out = ' '.join(self.tokens[self.offset + self.input_window: self.offset + self.input_window + self.output_window])
        self.offset += 1
        return (text_in, text_out)

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
            generated, ppl = model.generate_with_perplexity(input_text, max_length=max_length)
            
            # 处理生成结果（去掉 prompt 部分）
            # 注意：建议加个判断，防止切片报错
            generated_text = generated[len(input_text):] if len(generated) > len(input_text) else generated
            
            current_result = {
                'input': input_text,
                'expected': o,
                'generated': generated_text,
                'perplexity': ppl
            }

            # 2. 核心修改：以追加模式 ('a') 打开文件，写入一条后立即关闭
            # 或者保持文件打开，但使用 flush()
            with open(result_path, 'a', encoding='utf-8') as f:
                # ensure_ascii=False 保证德语字符（如 ä, ö, ü）正常显示，而不是乱码
                f.write(json.dumps(current_result, ensure_ascii=False) + '\n')
                
            # 日志打印
            logger.info(f'{self.offset} / {min(self.max_len, max_samples if max_samples > 0 else self.max_len)}')
            logger.info(generated_text)

