from typing import Protocol
import torch

class BaseDecoder(Protocol):
    def __call__(self, logits: torch.Tensor) -> int:
        pass


class GreedyDecoder(BaseDecoder):
    def __call__(self, logits: torch.Tensor) -> int:
        return torch.argmax(logits, dim=-1).item()


class TopKDecoder(BaseDecoder):
    def __init__(self, k: int):
        self.k = k

    def __call__(self, logits: torch.Tensor) -> int:
        # 获取 top-k 的索引和值
        topk_values, topk_indices = torch.topk(logits, self.k, dim=-1)
        
        # 对 top-k 的 logits 应用 softmax 得到概率分布
        topk_probs = torch.softmax(topk_values, dim=-1)
        
        # 从概率分布中采样
        sampled_index = torch.multinomial(topk_probs, num_samples=1)
        
        # 返回原始词汇表中的索引
        return topk_indices[sampled_index].item()


class TopPDecoder(BaseDecoder):
    def __init__(self, p: float):
        self.p = p

    def __call__(self, logits: torch.Tensor) -> int:
        # 计算 softmax 概率分布
        probs = torch.softmax(logits, dim=-1)
        
        # 对概率进行降序排序
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        
        # 计算累计概率
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # 找到累积概率超过 p 的位置（保留累积概率 <= p 的 token）
        # 使用布尔索引找到需要保留的 token
        mask = cumulative_probs <= self.p
        
        # 至少保留一个 token（概率最高的）
        if not mask.any():
            mask[0] = True
        else:
            # 包含第一个使累积概率超过 p 的 token
            # 找到第一个 False 的位置
            first_false = (~mask).nonzero(as_tuple=True)[0]
            if len(first_false) > 0:
                mask[first_false[0]] = True
        
        # 获取保留的索引和概率
        filtered_indices = sorted_indices[mask]
        filtered_probs = sorted_probs[mask]
        
        # 重新归一化保留的概率
        filtered_probs = filtered_probs / filtered_probs.sum()
        
        # 从过滤后的概率分布中采样
        sampled_index = torch.multinomial(filtered_probs, num_samples=1)
        
        # 返回原始词汇表中的索引
        return filtered_indices[sampled_index].item()
