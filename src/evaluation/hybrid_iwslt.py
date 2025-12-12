from .iwslt import IWSLTDataset
from .wikitext import WikitextDataset
from src.model.hybrid_model import HybridModel, SAEDepressionConfig, NeuronDepressionConfig
from src.model.decoder import DecoderRegistry
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='model name', choices=['gpt2', 'google/gemma-2-2b'])
    parser.add_argument('--dataset', type=str, default='iwslt', choices=['iwslt', 'wiki'], help='dataset name')
    parser.add_argument('--use-penalty', action='store_true', help='whether to use decoding penalty')
    parser.add_argument('--use-sae', action='store_true', help='whether to use sae depression')
    parser.add_argument('--use-neuron', action='store_true', help='whether to use neuron depression')
    parser.add_argument('--device', type=str, default='cuda:4', help='device to use')
    parser.add_argument('--max-samples', type=int, default=1000, help='maximum number of samples to evaluate')
    parser.add_argument('--output-file', type=str, default='results.jsonl', help='file to save results')
    parser.add_argument('--decode-strategy', type=str, default='greedy', choices=['greedy', 'top_k', 'top_p'], help='decoding strategy')
    parser.add_argument('--topk', type=int, default=30, help='top-k value for top-k decoding')
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

    sae_config = SAEDepressionConfig()
    neuron_config = NeuronDepressionConfig()

    if args.model == 'gpt2':
        sae_config.latent_idxs = [22275, 6972, 8357, 3615, 13944, 7798, 10178, 22317, 18380, 16631, 3661, 16888, 3164, 6371, 17597, 16894, 12873, 7083, 5295, 8848, 17443, 23990, 18929, 21963, 15147, 10931, 4051, 4025, 20200, 186, 19336, 15875, 7699, 5051, 7770, 24312]
        sae_config.sae_release = 'gpt2-small-res-jb'
        sae_config.sae_id = 'blocks.9.hook_resid_pre'
    elif args.model == 'google/gemma-2-2b':
        sae_config.latent_idxs = [3350, 2424, 2752, 11566, 11653, 3050, 11018, 1563, 3996, 13589, 7644, 15662, 13000, 13032, 2210, 15312, 12056, 2205, 13513, 94, 421, 3858, 4884, 12653, 10243, 5263, 6608, 9423, 10860, 11592, 7637, 7618, 14613, 8065, 8509, 7341, 2645, 15954, 1988, 5490, 11985, 16300, 4017, 11076, 5425, 11049, 5429, 9227, 9795, 10178, 10566, 11073, 13907, 16094, 4946, 6129, 630, 7543, 1883, 8280, 14727, 12656, 12493, 7704, 13775, 1008, 6206, 7624, 11423, 14848, 14950, 2678, 3440, 4051, 7827, 8575, 13593, 16186, 13586, 16105, 6789, 147, 514, 1000, 7470, 15037, 577, 1447, 1007, 2632, 4071, 4807, 5964, 6954, 10744, 12099, 12827, 14148, 1365, 6023]
        sae_config.sae_release = 'gemma-scope-2b-pt-res-canonical'
        sae_config.sae_id = 'layer_25/width_16k/canonical'

        

    hybrid_model = HybridModel(
        model_name=args.model,
        model=model,
        tokenizer=tokenizer,
        decoder=decoder,
        use_decode_penalty=args.use_penalty,
        use_sae_depression=args.use_sae,
        sae_depression_config=sae_config,
        use_neuron_depression=args.use_neuron,
        neuron_depression_config=neuron_config,
        device=args.device,
    )

    dataset.evaluate(hybrid_model, result_path=args.output_file, max_length=args.max_length, max_samples=args.max_samples)


if __name__ == '__main__':
    main()