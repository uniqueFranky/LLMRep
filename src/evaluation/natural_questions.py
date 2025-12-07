from datasets import load_dataset
from .dataset import Dataset
from src.model.base_model import BaseModel
import json
import logging
import argparse

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)



class NaturalQuestionsDataset(Dataset):
    def __init__(self, split: str = "train"):
        self.dataset = load_dataset("sentence-transformers/natural-questions", split=split)
        self.offset = 0
        self.max_len = len(self.dataset)


    def __iter__(self):
        self.offset = 0
        return self
    
    def __next__(self) -> tuple[str, str]:
        if self.offset >= self.max_len:
            raise StopIteration
        
        item = self.dataset[self.offset]
        self.offset += 1
        
        return (item['query'], item['answer'])
    
    def evaluate(self, model: BaseModel, result_path: str, max_length: int=500, max_samples: int=-1):
        # 1. 在循环开始前，先清空或创建文件（如果需要续写，可以改用 'a' 模式）
        # 使用 'w' 模式会覆盖旧文件，确保从头开始
        with open(result_path, 'w', encoding='utf-8') as f:
            pass 
        cnt = 0
        for (i, o) in iter(self):
            cnt += 1
            if max_samples > 0 and cnt > max_samples:
                break
            input_text = i
            
            # 推理
            generated, ppl = model.generate_with_perplexity(input_text, max_length=max_length)
            
            # 处理生成结果（去掉 prompt 部分）
            # 注意：建议加个判断，防止切片报错
            generated_text = generated[len(input_text):] if len(generated) > len(input_text) else generated
            
            current_result = {
                'input': input_text,
                'expected': o,
                'generated': generated_text,
                'perplexity': ppl
            }

            # 2. 核心修改：以追加模式 ('a') 打开文件，写入一条后立即关闭
            # 或者保持文件打开，但使用 flush()
            with open(result_path, 'a', encoding='utf-8') as f:
                # ensure_ascii=False 保证德语字符（如 ä, ö, ü）正常显示，而不是乱码
                f.write(json.dumps(current_result, ensure_ascii=False) + '\n')
                
            # 日志打印
            logger.info(f'{self.offset} / {min(self.max_len, max_samples if max_samples > 0 else self.max_len)}')
            logger.info(generated_text)


# python -m src.evaluation.natural_questions --model=gpt2 --result-path=results/nq_greedy_gpt2.jsonl --max-length=150 --decode-mode=greedy --max-samples=1000
# python -m src.evaluation.natural_questions --model=gpt2 --result-path=results/nq_penlaty_gpt2.jsonl --max-length=150 --decode-mode=penalty --max-samples=1000
# python -m src.evaluation.natural_questions --model=gpt2 --result-path=results/nq_greedy_gpt2.jsonl --compute-metrics
# python -m src.evaluation.natural_questions --model=gpt2 --result-path=results/nq_penlaty_gpt2.jsonl --compute-metrics
# python -m src.evaluation.natural_questions --model=gpt2 --result-path=results/nq_penlaty_gpt2.jsonl.metrics.jsonl --result-path2=results/nq_greedy_gpt2.jsonl.metrics.jsonl --compare-metrics


# python -m src.evaluation.natural_questions --model=google/gemma-2-2b --result-path=results/nq_greedy_gemma2.jsonl --max-length=150 --decode-mode=greedy --max-samples=1000 --device=cuda:4
# python -m src.evaluation.natural_questions --model=google/gemma-2-2b --result-path=results/nq_penlaty_gemma2.jsonl --max-length=150 --decode-mode=penalty --max-samples=1000 --device=cuda:5
# python -m src.evaluation.natural_questions --model=google/gemma-2-2b --result-path=results/nq_greedy_gemma2.jsonl --compute-metrics
# python -m src.evaluation.natural_questions --model=google/gemma-2-2b --result-path=results/nq_penlaty_gemma2.jsonl --compute-metrics
# python -m src.evaluation.natural_questions --model=google/gemma-2-2b --result-path=results/nq_penlaty_gemma2.jsonl.metrics.jsonl --result-path2=results/nq_greedy_gemma2.jsonl.metrics.jsonl --compare-metrics


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--model', type=str, required=True, help='Model name, e.g., gpt2')
    argparser.add_argument('--result-path', type=str, required=True, help='Path to save evaluation results')
    argparser.add_argument('--result-path2', type=str, default=None, help='')
    argparser.add_argument('--max-length', type=int, default=150, help='Maximum length for generation')
    argparser.add_argument('--decode-mode', type=str, default='greedy', help='Decoding mode: greedy, penalty.')
    argparser.add_argument('--max-samples', type=int, default=-1, help='Maximum number of samples to evaluate, -1 for all.')
    argparser.add_argument('--compute-metrics', action='store_true', help='Only compute metrics from existing results if set.')
    argparser.add_argument('--compare-metrics', action='store_true', help='Only compare metrics from existing results if set.')
    argparser.add_argument('--device', type=str, default='auto', help='Device to use for model inference.')
    
    args = argparser.parse_args()
    model_name = args.model
    result_path = args.result_path
    max_length = args.max_length
    decode_mode = args.decode_mode

    if args.compare_metrics:
        path1 = args.result_path
        path2 = args.result_path2
        bleu1 = 0
        bleu2 = 0
        meteor1 = 0
        meteor2 = 0
        ppl1 = 0
        ppl2 = 0
        repw1 = 0
        repw2 = 0
        repr1 = 0
        repr2 = 0
        repn1 = 0
        repn2 = 0
        with open(path1, 'r', encoding='utf-8') as f:
            results1 = [json.loads(line) for line in f.readlines()]
            with open(path2, 'r', encoding='utf-8') as f2:
                results2 = [json.loads(line) for line in f2.readlines()]

                num = min(len(results1), len(results2))
                for i in range(num):
                    bleu1 += results1[i]['bleu']
                    bleu2 += results2[i]['bleu']
                    meteor1 += results1[i]['meteor']
                    meteor2 += results2[i]['meteor']
                    ppl1 += results1[i]['perplexity']
                    ppl2 += results2[i]['perplexity']
                    repw1 += results1[i]['rep_w']
                    repw2 += results2[i]['rep_w']
                    repr1 += results1[i]['rep_r']
                    repr2 += results2[i]['rep_r']
                    repn1 += results1[i]['rep_n_5']
                    repn2 += results2[i]['rep_n_5']
                
                bleu1 /= num
                bleu2 /= num
                meteor1 /= num
                meteor2 /= num
                ppl1 /= num
                ppl2 /= num
                repw1 /= num
                repw2 /= num
                repr1 /= num
                repr2 /= num
                repn1 /= num
                repn2 /= num
                logger.info(f'Comparison of metrics between {path1} and {path2}:')
                logger.info(f'BLEU: {bleu1:.4f} vs {bleu2:.4f}')
                logger.info(f'METEOR: {meteor1:.4f} vs {meteor2:.4f}')
                logger.info(f'Perplexity: {ppl1:.4f} vs {ppl2:.4f}')
                logger.info(f'Rep_w: {repw1:.4f} vs {repw2:.4f}')
                logger.info(f'Rep_r: {repr1:.4f} vs {repr2:.4f}')
                logger.info(f'Rep_n_5: {repn1:.4f} vs {repn2:.4f}')

        exit(0)

    if args.compute_metrics:
        from src.metrics import bleu, meteor, perplexity, rep_w, rep_n, rep_r
        
        # 输出路径
        output_metrics_path = result_path + '.metrics.jsonl'
        
        # 读取结果
        with open(result_path, 'r', encoding='utf-8') as f:
            results = [json.loads(line) for line in f.readlines()]
        
        logger.info(f'Computing metrics for {len(results)} samples...')
        logger.info(f'Writing metrics to {output_metrics_path}')
        
        # 逐条计算并写入
        with open(output_metrics_path, 'w', encoding='utf-8') as out_f:
            for i, item in enumerate(results):
                pred = item['generated']
                gold = item['expected']
                
                # 计算指标
                metric = {
                    'index': i,
                    'input': item.get('input', ''),
                    'generated': pred,
                    'expected': gold,
                    'bleu': bleu(pred, gold),
                    'meteor': meteor(pred, gold),
                    'perplexity': item.get('perplexity', float('inf')),
                    'rep_w': rep_w(pred.split(), w=10),
                    'rep_n_1': rep_n(pred.split(), n=1),
                    'rep_n_2': rep_n(pred.split(), n=2),
                    'rep_n_3': rep_n(pred.split(), n=3),
                    'rep_n_4': rep_n(pred.split(), n=4),
                    'rep_n_5': rep_n(pred.split(), n=5),
                    'rep_r': rep_r(pred.split())
                }
                
                # 立即写入文件
                out_f.write(json.dumps(metric, ensure_ascii=False) + '\n')
                out_f.flush()  # 确保立即写入磁盘
                
                # 定期打印进度
                if (i + 1) % 100 == 0:
                    logger.info(f'Computed metrics for {i + 1}/{len(results)} samples')
        
        logger.info(f'All metrics written to {output_metrics_path}')
        
        # 计算平均值
        logger.info('Computing average metrics...')
        with open(output_metrics_path, 'r', encoding='utf-8') as f:
            all_metrics = [json.loads(line) for line in f.readlines()]
        
        # 提取数值指标的键
        metric_keys = ['bleu', 'meteor', 'perplexity', 'rep_w', 
                       'rep_n_1', 'rep_n_2', 'rep_n_3', 'rep_n_4', 'rep_n_5', 'rep_r']
        
        # 计算平均值
        avg_metrics = {}
        for key in metric_keys:
            values = [m[key] for m in all_metrics]
            avg_metrics[key] = sum(values) / len(values)
        
        # 保存汇总结果
        summary_path = result_path + '.metrics.summary.json'
        summary = {
            'model': model_name,
            'decode_mode': decode_mode,
            'num_samples': len(all_metrics),
            'avg_metrics': avg_metrics,
            'result_path': result_path,
            'metrics_path': output_metrics_path
        }
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=4)
        
        logger.info(f'Summary saved to {summary_path}')
        logger.info('Average Metrics:')
        for key, value in avg_metrics.items():
            logger.info(f'  {key}: {value:.4f}')
        
        exit(0)


    model = None
    tokenizer = None

    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=args.device, 
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    

    dataset = NaturalQuestionsDataset(split='train')

    decode_model = None
    if decode_mode == 'greedy':
        from src.model.greedy_decode_model import GreedyDecodeModel
        decode_model = GreedyDecodeModel(tokenizer, model)
    elif decode_mode == 'penalty':
        from src.model.decode_penalty_model import DecodePenaltyModel
        from src.decode import apply_ngram_penalty
        decode_model = DecodePenaltyModel(tokenizer, model, penalty_func=apply_ngram_penalty)
    else:
        raise ValueError(f'Unsupported decode mode: {decode_mode}')

    dataset.evaluate(decode_model, result_path, max_length=max_length, max_samples=args.max_samples)
    

