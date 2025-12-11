from typing import Protocol, Dict, Type
import torch

class BaseDecoder(Protocol):
    def __call__(self, logits: torch.Tensor) -> int:
        pass


# 创建注册表
class DecoderRegistry:
    _registry: Dict[str, Type[BaseDecoder]] = {}
    
    @classmethod
    def register(cls, name: str):
        """装饰器：注册 decoder"""
        def decorator(decoder_cls):
            cls._registry[name] = decoder_cls
            return decoder_cls
        return decorator
    
    @classmethod
    def get(cls, name: str, **kwargs) -> BaseDecoder:
        """根据名称获取 decoder 实例"""
        if name not in cls._registry:
            raise ValueError(f"Decoder '{name}' not found. Available: {list(cls._registry.keys())}")
        return cls._registry[name](**kwargs)
    
    @classmethod
    def list_decoders(cls):
        """列出所有已注册的 decoder"""
        return list(cls._registry.keys())


@DecoderRegistry.register("greedy")
class GreedyDecoder(BaseDecoder):
    def __init__(self, **kwargs):
        pass
    def __call__(self, logits: torch.Tensor) -> int:
        return torch.argmax(logits, dim=-1).item()


@DecoderRegistry.register("top_k")
class TopKDecoder(BaseDecoder):
    def __init__(self, **kwargs):
        self.k = kwargs.get("k", 3)

    def __call__(self, logits: torch.Tensor) -> int:
        topk_values, topk_indices = torch.topk(logits, self.k, dim=-1)
        topk_probs = torch.softmax(topk_values, dim=-1)
        sampled_index = torch.multinomial(topk_probs, num_samples=1)
        return topk_indices[sampled_index].item()


@DecoderRegistry.register("top_p")
class TopPDecoder(BaseDecoder):
    def __init__(self, **kwargs):
        self.p = kwargs.get("p", 0.9)

    def __call__(self, logits: torch.Tensor) -> int:
        probs = torch.softmax(logits, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # 找到累积概率超过 p 的位置
        mask = cumulative_probs <= self.p
        # 确保至少包含一个 token
        if not mask.any():
            mask[0] = True
        else:
            # 包含第一个使累积概率超过 p 的 token
            last_included = mask.sum().item()
            if last_included < len(mask):
                mask[last_included] = True
        
        filtered_indices = sorted_indices[mask]
        filtered_probs = sorted_probs[mask]
        filtered_probs = filtered_probs / filtered_probs.sum()
        sampled_index = torch.multinomial(filtered_probs, num_samples=1)
        return filtered_indices[sampled_index].item()



# 使用示例
if __name__ == "__main__":
    logits = torch.tensor([1.0, 3.0, 2.0, 0.5, 2.5])
    
    # 查看所有可用的 decoder
    print("Available decoders:", DecoderRegistry.list_decoders())
    
    # 通过名称创建 decoder
    greedy = DecoderRegistry.get("greedy")
    print(f"Greedy: {greedy(logits)}")
    
    topk = DecoderRegistry.get("top_k", k=3)
    print(f"Top-K: {topk(logits)}")
    
    topp = DecoderRegistry.get("top_p", p=0.8)
    print(f"Top-P: {topp(logits)}")
