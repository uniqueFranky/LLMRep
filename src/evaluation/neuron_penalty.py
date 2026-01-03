import argparse
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.model.neuron_prevent_penalty_model import NeuronPreventPenaltyModel
from src.model.decoder import DecoderRegistry
from src.evaluation.iwslt import IWSLTDataset

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='model name', choices=['gpt2', 'google/gemma-2-2b'])
    parser.add_argument('--use-penalty', action='store_true', help='whether to use decoding penalty')
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
    if args.use_penalty:
        penalty_config = {
            3: 0.8,
            4: 0.7,
            5: 0.6,
            6: 0.5,
            7: 0.4,
            8: 0.3,
            9: 0.2,
            10: 0.1,
        }
    else:
        penalty_config = None

    
    decoder = DecoderRegistry.get(
        args.decode_strategy,
        kwargs={
            'k': args.topk,
            'p': args.topp
        })

    hybrid_model = NeuronPreventPenaltyModel(tokenizer, model, f'/data/data_public/lixutian/repetition_neuron/datasets/repetitionDIY/{args.model}.pt', decoder, penalty_config=penalty_config, device=args.device)
    dataset = IWSLTDataset(split='test')
    dataset.evaluate(hybrid_model, args.output_file, max_samples=args.max_samples, max_length=args.max_length)


if __name__ == '__main__':
    main()