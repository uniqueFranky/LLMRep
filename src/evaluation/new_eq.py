import json
import logging
from typing import List, Dict

from .dataset import Dataset
from src.model.base_model import BaseModel

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class NewEQDataset(Dataset):
    def __init__(self, jsonl_path: str):
        """
        jsonl_path: rewritten_eq_dataset.jsonl
        Each line:
        {
          "id": int,
          "question": str,
          "rewritten_question": str
        }
        """
        self.data: List[Dict] = []

        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                self.data.append(obj)

        # ✅ 关键点：按 id 排序，保证与原 EQ 顺序一致
        self.data.sort(key=lambda x: x["id"])

        self.offset = 0
        self.max_len = len(self.data)

    def __iter__(self):
        self.offset = 0
        return self

    def __next__(self) -> tuple[str, str]:
        if self.offset >= self.max_len:
            raise StopIteration

        item = self.data[self.offset]
        self.offset += 1

        # 使用 rewritten_question 作为 input
        return (item["rewritten_question"], "")

    def evaluate(
        self,
        model: BaseModel,
        result_path: str,
        max_length: int = 500,
        max_samples: int = -1
    ):
        # 1. 清空结果文件
        with open(result_path, 'w', encoding='utf-8'):
            pass

        cnt = 0
        for (i, o) in iter(self):
            cnt += 1
            if max_samples > 0 and cnt > max_samples:
                break

            input_text = i

            # 推理
            generated, ppl = model.generate_with_perplexity(
                input_text,
                max_length=max_length,
                with_prompt=True
            )

            current_result = {
                "input": input_text,
                "expected": o,
                "generated": generated,
                "perplexity": ppl
            }

            # 2. 追加写入
            with open(result_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(current_result, ensure_ascii=False) + '\n')

            logger.info(
                f"{self.offset} / {min(self.max_len, max_samples if max_samples > 0 else self.max_len)}"
            )
            logger.info(generated)
