from typing import Union, List

class BaseModel:
    def generate(input: Union[str, List[str]], max_length: int) -> str:
        pass

    def generate_with_perplexity(input: str, max_length: int, with_prompt: bool) -> tuple[str, float]:
        pass