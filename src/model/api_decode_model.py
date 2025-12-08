from .base_model import BaseModel
import requests
import json
import time
import math


class APIDecodeModel(BaseModel):
    """
    最简 API 模型调用类
    - 仅实现 BaseModel 要求的两个方法
    - 适配硅基流动（SiliconFlow）或任何兼容 Chat Completions 格式的 API
    """

    def __init__(
        self,
        api_url: str,
        api_key: str,
        model_name: str,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        max_retries: int = 3,
        timeout: int = 30,
    ):
        self.api_url = api_url
        self.api_key = api_key
        self.model_name = model_name

        # 解码参数
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p

        # 请求控制
        self.max_retries = max_retries
        self.timeout = timeout

    # ==============================================
    # BaseModel 接口 1: generate_with_perplexity
    # ==============================================
    def generate_with_perplexity(self, input: str, max_length: int = 100):

        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": input}],
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "max_tokens": max_length,
            "enable_thinking": False, 
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        # --- 简单 retry 机制 ---
        for attempt in range(self.max_retries):
            try:
                resp = requests.post(
                    self.api_url,
                    json=payload,
                    headers=headers,
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                break
            except Exception as e:
                print(f"[APIDecodeModel] Error: {e} (retry {attempt+1}/{self.max_retries})")
                time.sleep(1)
        else:
            return "[API failed]", float("nan")

        data = resp.json()

        # --- 尝试解析 message.content ---
        try:
            output_text = data["choices"][0]["message"]["content"]
        except:
            output_text = json.dumps(data, ensure_ascii=False)

        # API 无 token-level logprobs → 无法算 perplexity
        return output_text, float("nan")

    # ==============================================
    # BaseModel 接口 2: generate
    # ==============================================
    def generate(self, input: str, max_length: int = 100) -> str:
        text, _ = self.generate_with_perplexity(input, max_length)
        return text
