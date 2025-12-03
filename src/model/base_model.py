from typing import Union, List

class BaseModel:
    def generate(input: Union[str, List[str]], max_length: int) -> str:
        pass