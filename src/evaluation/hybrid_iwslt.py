from .iwslt import IWSLTDataset
from src.model.hybrid_model import HybridModel
from src.model.decoder import DecoderRegistry
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='model name', choices=['gpt2', 'google/gemma-2-2b'])
    parser.add_argument('--dataset', type=str, default='iwslt', choices=['iwslt'], help='dataset name')
    parser.add_argument('--use-penalty', action='store_true', help='whether to use decoding penalty')
    parser.add_argument('--use-sae', action='store_true', help='whether to use sae depression')
    parser.add_argument('--use-neuron', action='store_true', help='whether to use neuron depression')
    parser.add_argument('--device', type=str, default='cuda:4', help='device to use')
    parser.add_argument('--max-samples', type=int, default=1000, help='maximum number of samples to evaluate')
    parser.add_argument('--output-file', type=str, default='results.jsonl', help='file to save results')
    parser.add_argument('--decode-strategy', type=str, default='greedy', choices=['greedy', 'top_k', 'top_p'], help='decoding strategy')
    parser.add_argument('--topk', type=int, default=3, help='top-k value for top-k decoding')
    parser.add_argument('--topp', type=float, default=0.9, help='top-p value for top-p decoding')
    parser.add_argument('--max-length', type=int, default=150, help='maximum generation length')
    return parser.parse_args()


def main():
    args = parse_args()

    model = AutoModelForCausalLM.from_pretrained(args.model).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    dataset = None
    if args.dataset == 'iwslt':
        dataset = IWSLTDataset(split='test')
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    
    decoder = DecoderRegistry.get(
        args.decode_strategy,
        kwargs={
            'k': args.topk,
            'p': args.topp
        })

    hybrid_model = HybridModel(
        model_name=args.model,
        model=model,
        tokenizer=tokenizer,
        decoder=decoder,
        use_decode_penalty=args.use_penalty,
        use_sae_depression=args.use_sae,
        use_neuron_depression=args.use_neuron,
        device=args.device,
    )

    dataset.evaluate(hybrid_model, result_path=args.output_file, max_length=args.max_length, max_samples=args.max_samples)


if __name__ == '__main__':
    main()