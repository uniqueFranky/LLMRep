from .base_model import BaseModel
from .decoder import BaseDecoder
from dataclasses import dataclass
import logging
from sae_lens import SAE
import torch
import math
from src.find_neurons import findNeurons
from src.neuron_intervention import *
from contextlib import nullcontext
from src.decode import apply_ngram_penalty

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class DecodePenaltyConfig:
    penalty_dict: dict[int, float] = None
    
    def __post_init__(self):
        if self.penalty_dict is None:
            self.penalty_dict = {
                3: 0.9,
                4: 0.8,
                5: 0.7,
            }


@dataclass
class SAEDepressionConfig:
    latent_idxs: list[int] = [22275, 6972, 8357, 3615, 13944, 7798, 10178, 22317, 18380, 16631, 3661, 16888, 3164, 6371, 17597, 16894, 12873, 7083, 5295, 8848, 17443, 23990, 18929, 21963, 15147, 10931, 4051, 4025, 20200, 186, 19336, 15875, 7699, 5051, 7770, 24312]
    steering_coefficient: float = -0.5
    sae_release: str = "gpt2-small-res-jb"
    sae_id: str = "blocks.9.hook_resid_pre"
    device: str = "cuda:4"


@dataclass
class NeuronDepressionConfig:
    repetition_dataset_path: str = '/data/data_public/lixutian/repetition_neuron/datasets/repetitionDIY'
    k: int = 1500


class HybridModel(BaseModel):
    model_name: str
    use_decode_penalty: bool = False
    use_sae_depression: bool = False
    use_neuron_depression: bool = False
    decode_penalty_config: DecodePenaltyConfig = DecodePenaltyConfig()
    sae_depression_config: SAEDepressionConfig = SAEDepressionConfig()
    neuron_depression_config: NeuronDepressionConfig = NeuronDepressionConfig()
    decoder: BaseDecoder

    def __init__(self, model_name, model, tokenizer, decoder: BaseDecoder,
                 use_decode_penalty=False, decode_penalty_config=None,
                 use_sae_depression=False, sae_depression_config=None,
                 use_neuron_depression=False, neuron_depression_config=None,
                 device='cuda:4'
                ):
        self.model_name = model_name
        self.model = model
        self.tokenizer = tokenizer
        self.decoder = decoder
        self.device = device
        self.use_decode_penalty = use_decode_penalty
        self.decode_penalty_config = decode_penalty_config or DecodePenaltyConfig()
        self.use_sae_depression = use_sae_depression
        self.sae_depression_config = sae_depression_config or SAEDepressionConfig()
        self.use_neuron_depression = use_neuron_depression
        self.neuron_depression_config = neuron_depression_config or NeuronDepressionConfig()
        
        # SAE相关初始化
        self.sae = None
        self.steer_vec = None
        self.hook_handle = None
        self.target_module = None
        
        if use_sae_depression:
            self.init_sae_depression()
        
        if use_neuron_depression:
            self.init_neuron_depression()

    def init_sae_depression(self):
        logger.info("Initializing SAE Depression with config: %s", self.sae_depression_config)
        logger.info("Loading SAE...")
        
        # 加载SAE
        self.sae, cfg_dict, sparsity = SAE.from_pretrained(
            release=self.sae_depression_config.sae_release,
            sae_id=self.sae_depression_config.sae_id,
            device=self.device,
        )
        logger.info("SAE loaded.")

        # 获取hook信息
        hook_name = self.sae.cfg.metadata["hook_name"]
        
        # 解析hook_name来找到对应的层
        # 例如: "blocks.9.hook_resid_pre" -> layer 9
        if "blocks." in hook_name:
            layer_idx = int(hook_name.split(".")[1])
            
            # 根据模型类型找到对应的层
            model_type = str(type(self.model))
            if 'GPT2LMHeadModel' in model_type:
                self.target_module = self.model.transformer.h[layer_idx]
            elif 'GemmaForCausalLM' in model_type or 'Gemma2ForCausalLM' in model_type or 'LlamaForCausalLM' in model_type:
                self.target_module = self.model.model.layers[layer_idx]
            elif 'GPTNeoXForCausalLM' in model_type:
                self.target_module = self.model.gpt_neox.layers[layer_idx]
            elif 'PhiForCausalLM' in model_type:
                self.target_module = self.model.model.layers[layer_idx]
            else:
                raise ValueError(f"Model type {model_type} not supported for SAE depression")

        # 设置tokenizer
        if not hasattr(self.tokenizer, 'pad_token') or self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 计算steering vector，确保数据类型匹配
        with torch.no_grad():
            # 获取模型的数据类型
            model_dtype = next(self.model.parameters()).dtype
            
            steer_vec = torch.zeros(self.sae.W_dec.shape[1], device=self.device, dtype=model_dtype)
            if self.sae_depression_config.latent_idxs:
                for idx in self.sae_depression_config.latent_idxs:
                    steer_vec += self.sae.W_dec[idx].to(dtype=model_dtype)
            self.steer_vec = self.sae_depression_config.steering_coefficient * steer_vec

    def _steering_hook_fn(self, module, input, output):
        """SAE steering hook function"""
        # 确保steering vector的数据类型与输出匹配
        if isinstance(output, tuple):
            # 如果输出是tuple，通常第一个元素是主要的激活
            modified_output = list(output)
            steer_vec = self.steer_vec.to(dtype=modified_output[0].dtype, device=modified_output[0].device)
            modified_output[0] = modified_output[0] + steer_vec
            return tuple(modified_output)
        else:
            # 如果输出是tensor
            steer_vec = self.steer_vec.to(dtype=output.dtype, device=output.device)
            return output + steer_vec
    
    def init_neuron_depression(self):
        logger.info("Initializing Neuron Depression with config: %s", self.neuron_depression_config)
        repetition_dataset = torch.load(f'{self.neuron_depression_config.repetition_dataset_path}/{self.model_name}.pt', map_location=self.device)
        self.sortedNeurons = findNeurons(repetition_dataset, self.model, self.tokenizer, maxRange=30, device=self.device)
        self.targetNeurons = [neuron['neuron'] for neuron in self.sortedNeurons[:self.neuron_depression_config.k]]

    def _register_sae_hook(self):
        """注册SAE hook"""
        if self.use_sae_depression and self.target_module is not None:
            self.hook_handle = self.target_module.register_forward_hook(self._steering_hook_fn)

    def _remove_sae_hook(self):
        """移除SAE hook"""
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None

    def generate_with_perplexity(self, input: str, max_length: int = 150) -> tuple[str, float]:
        """
        生成文本并同时计算perplexity
        返回: (generated_text, perplexity)
        """
        generated = input
        total_log_prob = 0.0
        num_tokens = 0
        
        with torch.no_grad():
            for step in range(max_length):
                acts = []
                
                # 在每个步骤中设置神经元干预
                if self.use_neuron_depression:
                    INTERV = Deactivator
                    layer2neurons = convertNeuronsToDict(self.targetNeurons)

                    model_type = str(type(self.model))
                    if 'GemmaForCausalLM' in model_type or 'Gemma2ForCausalLM' in model_type or 'LlamaForCausalLM' in model_type:
                        acts = [INTERV(layer.mlp.act_fn, layer2neurons[i], 'last') for i, layer in enumerate(self.model.model.layers) if i in layer2neurons]
                    elif 'GPTNeoXForCausalLM' in model_type:
                        acts = [INTERV(layer.mlp.act, layer2neurons[i], 'last') for i, layer in enumerate(self.model.gpt_neox.layers) if i in layer2neurons]
                    elif 'PhiForCausalLM' in model_type:
                        acts = [INTERV(layer.mlp.activation_fn, layer2neurons[i], 'last') for i, layer in enumerate(self.model.model.layers) if i in layer2neurons]
                    elif 'GPT2LMHeadModel' in model_type:
                        acts = [INTERV(layer.mlp.act, layer2neurons[i], 'last') for i, layer in enumerate(self.model.transformer.h) if i in layer2neurons]
                    else:
                        raise ValueError(f'model {model_type} is not supported!')

                # 注册SAE hook
                if self.use_sae_depression:
                    self._register_sae_hook()

                try:
                    enc = self.tokenizer(generated, return_tensors="pt").to(self.device)
                    generated_ids = enc["input_ids"]
                    
                    # 获取模型输出
                    outputs = self.model(generated_ids)
                    
                    # 处理不同的输出格式
                    if hasattr(outputs, 'logits'):
                        logits = outputs.logits
                    else:
                        logits = outputs
                    
                    next_logits = logits[0, -1, :]
                    
                    # 计算概率分布（在应用惩罚之前）
                    log_probs = torch.log_softmax(next_logits, dim=-1)

                    # 应用解码惩罚
                    if self.use_decode_penalty:
                        penalty_dict = self.decode_penalty_config.penalty_dict
                        for ngram_size, penalty in penalty_dict.items():
                            next_logits = apply_ngram_penalty(generated, self.tokenizer, next_logits, ngram_size, penalty)

                    next_token_id = self.decoder(next_logits)
                    next_token = self.tokenizer.decode([next_token_id])
                    
                    # 累积log probability（跳过第一步）
                    if step > 0:
                        total_log_prob += log_probs[next_token_id].item()
                        num_tokens += 1
                    
                    generated += next_token
                    
                    # 检查是否生成结束token
                    if next_token_id == self.tokenizer.eos_token_id:
                        break
                        
                finally:
                    # 清理SAE hook
                    if self.use_sae_depression:
                        self._remove_sae_hook()
                    
                    # 在每个步骤结束时清理神经元干预
                    if self.use_neuron_depression:
                        for act in acts:
                            act.release()
        
        # 计算perplexity
        if num_tokens > 0:
            avg_log_prob = total_log_prob / num_tokens
            perplexity = math.exp(-avg_log_prob)
        else:
            perplexity = float('inf')
        
        return generated, perplexity

    def generate(self, input: str, max_length: int = 150) -> str:
        """保持原有接口不变"""
        generated_text, _ = self.generate_with_perplexity(input, max_length)
        return generated_text

    def __del__(self):
        """析构函数，确保hook被正确清理"""
        if hasattr(self, 'hook_handle') and self.hook_handle is not None:
            self.hook_handle.remove()
