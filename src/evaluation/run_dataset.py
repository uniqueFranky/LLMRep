import dotenv
import argparse
import os
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from pathlib import Path

dotenv.load_dotenv()
# 如果有本地模型，也可以导入
# from src.model.greedy_decode_model import GreedyDecodeModel

# ====== Dataset Registry ======
# 你需要在这里注册你的所有数据集
def load_dataset(name: str):
    if name == "iwslt":
        from src.evaluation.iwslt import IWSLTDataset
        return IWSLTDataset(split='test')
    elif name == "nq":
        from src.evaluation.natural_questions import NaturalQuestionsDataset
        return NaturalQuestionsDataset(split='train')
    elif name == "eq":
        from src.evaluation.diversity_challenge import DiversityChallengeDataset
        return DiversityChallengeDataset(split='train')
    else:
        raise ValueError(f'Unsupported dataset: {name}')

def load_api_model(model_name: str, temperature: float, top_k: int, top_p: float):
    from src.model.api_decode_model import APIDecodeModel
    decode_model = APIDecodeModel(
        api_url="https://api.siliconflow.cn/v1/chat/completions",
        api_key=os.getenv('LLM_API_KEY'),
        model_name=model_name,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )
    return decode_model

def load_model(args):
    if args.model in ["gpt2", "google/gemma-2-2b"]:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            device_map=args.device, 
            torch_dtype=torch.bfloat16,
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    if args.decode_mode == "greedy":
        if args.model in ["gpt2", "google/gemma-2-2b"]:
            from src.model.greedy_decode_model import GreedyDecodeModel
            decode_model = GreedyDecodeModel(tokenizer, model)
        else:
            decode_model = load_api_model(args.model, temperature=0.0, top_k=1, top_p=1.0)
    elif args.decode_mode == "topk":
        if args.model in ["gpt2", "google/gemma-2-2b"]:
            from src.model.topk_decode_model import TopKDecodeModel
            decode_model = TopKDecodeModel(tokenizer, model, top_k=30, temperature=0.5)
        else:
            decode_model = load_api_model(args.model, temperature=0.5, top_k=30, top_p=1.0)
    elif args.decode_mode == "topp":
        if args.model in ["gpt2", "google/gemma-2-2b"]:
            from src.model.topp_decode_model import TopPDecodeModel
            decode_model = TopPDecodeModel(tokenizer, model, top_p=0.9, temperature=0.5)
        else:
            decode_model = load_api_model(args.model, temperature=0.5, top_k=-1, top_p=0.9)
    elif args.decode_mode == "penalty":
        from src.model.decode_penalty_model import DecodePenaltyModel
        from src.decode import apply_ngram_penalty
        decode_model = DecodePenaltyModel(tokenizer, model, penalty_func=apply_ngram_penalty)
    elif args.decode_mode == "sae":
        from src.model.sae_decode_model import SaeGreedyDecodeModel
        if args.model == "gpt2":
            latent_idxs = [22275, 6972, 8357, 3615, 13944, 7798, 10178, 22317, 18380, 16631, 3661, 16888, 3164, 6371, 17597, 16894, 12873, 7083, 5295, 8848, 17443, 23990, 18929, 21963, 15147, 10931, 4051, 4025, 20200, 186, 19336, 15875, 7699, 5051, 7770, 24312]
            decode_model = SaeGreedyDecodeModel (
                latent_idxs=latent_idxs,
                steering_coefficient=-4,
                sae_release="gpt2-small-res-jb",
                sae_id="blocks.9.hook_resid_pre",
                device="cuda",
                # decoding_mode="greedy"
                decoding_mode="sample",
                temperature=0.5,
                # freq_penalty=0.5,
            )
        elif args.model == "google/gemma-2-2b":
            latent_idxs = [3350, 2424, 2752, 11566, 11653, 3050, 11018, 1563, 3996, 13589, 7644, 15662, 13000, 13032, 2210, 15312, 12056, 2205, 13513, 94, 421, 3858, 4884, 12653, 10243, 5263, 6608, 9423, 10860, 11592, 7637, 7618, 14613, 8065, 8509, 7341, 2645, 15954, 1988, 5490, 11985, 16300, 4017, 11076, 5425, 11049, 5429, 9227, 9795, 10178, 10566, 11073, 13907, 16094, 4946, 6129, 630, 7543, 1883, 8280, 14727, 12656, 12493, 7704, 13775, 1008, 6206, 7624, 11423, 14848, 14950, 2678, 3440, 4051, 7827, 8575, 13593, 16186, 13586, 16105, 6789, 147, 514, 1000, 7470, 15037, 577, 1447, 1007, 2632, 4071, 4807, 5964, 6954, 10744, 12099, 12827, 14148, 1365, 6023]
            decode_model = SaeGreedyDecodeModel (
                latent_idxs=latent_idxs,
                steering_coefficient=-0.8,
                sae_release="gemma-scope-2b-pt-res-canonical",
                sae_id="layer_25/width_16k/canonical",
                device="cuda",
                decoding_mode="sample",
                temperature=0.5,
                # freq_penalty=0.5,
            )
        else:
            raise ValueError(f'Unsupported sae mode for model : {args.model}')
    else:
        raise ValueError(f'Unsupported decode mode: {args.decode_mode}')

    return decode_model

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, required=True, choices=["iwslt", "nq", "eq"], 
                        help="Dataset name. nq is Natural Questions, eq is Diversity_Challenge.")
    parser.add_argument('--model', type=str, required=True, 
                        choices=["gpt2", "google/gemma-2-2b", "Qwen/Qwen3-8B", "Qwen/Qwen3-32B", "tencent/Hunyuan-A13B-Instruct", "deepseek-ai/DeepSeek-V3.2", "Qwen/Qwen3-235B-A22B-Instruct-2507"], 
                        help='Model name')
    parser.add_argument("--result_dir", type=str, required=True, help="dir to save evaluation results, filename is result_dir/dataset/{{decode_mode}}_{{model}}.jsonl")
    parser.add_argument('--max-length', type=int, default=150, help='Maximum length for generation')
    parser.add_argument('--decode-mode', type=str, required=True, help='Decoding mode: greedy, penalty.')
    parser.add_argument('--max-samples', type=int, default=-1, help='Maximum number of samples to evaluate, -1 for all.')
    parser.add_argument('--device', type=str, default='auto', help='Device to use for model inference.')

    args = parser.parse_args()

    # 加载 Dataset
    dataset = load_dataset(args.dataset)
    print(f"[INFO] Loaded dataset: {args.dataset}")

    # 加载 decode_model
    decode_model = load_model(args)
    print(f"[INFO] Using API model: {args.model}")

    model_name = args.model.split("/")[-1]
    project_root = Path(__file__).resolve().parent.parent.parent
    result_dir = project_root / args.result_dir / args.dataset
    result_dir.mkdir(parents=True, exist_ok=True)
    result_path = result_dir / f"{args.decode_mode}_{model_name}.jsonl"
    print(f"[INFO] Writing results to: {result_path}")
    dataset.evaluate(decode_model, result_path, max_length=args.max_length, max_samples=args.max_samples)

    print("[INFO] Evaluation complete.")


if __name__ == "__main__":
    main()
    
