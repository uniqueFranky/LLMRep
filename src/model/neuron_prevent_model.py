from .base_model import BaseModel
from ..decode import compute_logits
from ..find_neurons import findNeurons
from ..neuron_intervention import *
import torch
import math


class NeuronPreventModel(BaseModel):
    def __init__(self, tokenizer, model, dataset, K=1500, device=None):
        self.tokenizer = tokenizer
        self.model = model
        self.device = device or model.device

        repetitionDataset = torch.load(dataset, map_location=self.device)
        self.sortedNeurons = findNeurons(repetitionDataset, model, tokenizer, maxRange=30, device=self.device)
        self.targetNeurons = [neuron['neuron'] for neuron in self.sortedNeurons[:K]]


    def generate_with_perplexity(self, input: str, max_length: int = 100) -> tuple[str, float]:
        """
        生成文本并同时计算perplexity
        返回: (generated_text, perplexity)
        """
        model = self.model
        model.eval()
        mode = "deactive"

        if mode == 'activate':
            INTERV = Activator
        else:
            INTERV = Deactivator

        layer2neurons = convertNeuronsToDict(self.targetNeurons)

        acts =[]
        if 'GemmaForCausalLM' in str(type(model)) or 'Gemma2ForCausalLM' in str(type(model)) or 'LlamaForCausalLM' in str(type(model)):
            acts = [INTERV(layer.mlp.act_fn, layer2neurons[i], 'last') for i, layer in enumerate(model.model.layers) if
                    i in layer2neurons]
        elif 'GPTNeoXForCausalLM' in str(type(model)):
            acts = [INTERV(layer.mlp.act, layer2neurons[i], 'last') for i, layer in enumerate(model.gpt_neox.layers) if
                    i in layer2neurons]
        elif 'PhiForCausalLM' in str(type(model)):
            acts = [INTERV(layer.mlp.activation_fn, layer2neurons[i], 'last') for i, layer in
                    enumerate(model.model.layers) if i in layer2neurons]
        elif 'GPT2LMHeadModel' in str(type(model)):
            acts = [INTERV(layer.mlp.act, layer2neurons[i], 'last') for i, layer in enumerate(model.transformer.h) if
                    i in layer2neurons]
        else:
            print('model is not supported!')


        generated = input
        total_log_prob = 0.0
        num_tokens = 0

        # 获取输入的token数量
        input_tokens = self.tokenizer.encode(input)
        input_length = len(input_tokens)

        with torch.no_grad():
            for step in range(max_length):
                # 计算当前序列的logits
                raw_logits = compute_logits(self.model, self.tokenizer, generated)

                if raw_logits.dim() == 1:
                    logits_2d = raw_logits.unsqueeze(0)  # [vocab_size] -> [1, vocab_size]
                else:
                    logits_2d = raw_logits

                penalized_logits_2d = raw_logits.clone()

                # 转换回1维用于后续计算
                if penalized_logits_2d.dim() == 2:
                    penalized_logits = penalized_logits_2d.squeeze(0)
                    original_logits = logits_2d.squeeze(0)
                else:
                    penalized_logits = penalized_logits_2d
                    original_logits = raw_logits

                # 计算概率分布（用于perplexity计算，使用原始logits）
                log_probs = torch.log_softmax(original_logits, dim=-1)

                # 选择下一个token（使用penalized logits）
                next_token_id = torch.argmax(penalized_logits, dim=-1).item()
                next_token = self.tokenizer.decode([next_token_id])

                # 累积log probability
                current_tokens = self.tokenizer.encode(generated)
                if len(current_tokens) > input_length:
                    total_log_prob += log_probs[next_token_id].item()
                    num_tokens += 1

                generated += next_token

                if next_token_id == self.tokenizer.eos_token_id:
                    break

        for a in acts:
            a.release()

        # 计算perplexity
        if num_tokens > 0:
            avg_log_prob = total_log_prob / num_tokens
            perplexity = math.exp(-avg_log_prob)
        else:
            perplexity = float('inf')

        return generated, perplexity


    def generate(self, input: str, max_length: int = 100) -> str:
        """保持原有接口不变"""
        generated_text, _ = self.generate_with_perplexity(input, max_length)
        return generated_text
