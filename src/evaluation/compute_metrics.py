#!/usr/bin/env python3
import argparse
import os
import json
from pathlib import Path

from src.metrics import bleu, meteor, rep_w, rep_n, rep_r

def strip_jsonl_suffix(path: Path) -> Path:
    """移除 .jsonl 后缀"""
    if path.name.endswith(".jsonl"):
        return path.with_suffix("")  # 去掉 .jsonl
    return path


def extract_model_and_decode(file_path: Path):
    """
    从文件名 {decode_mode}_{model}.jsonl 中提取 decode_mode 和 model
    忽略目录，只看文件名
    """
    name = file_path.stem  # 不带后缀

    # 示例：topp09_Qwen2-7B  → ["topp09", "Qwen2-7B"]
    if "_" in name:
        parts = name.split("_", 1)
        decode_mode = parts[0]
        model_name = parts[1]
    else:
        decode_mode = "unknown"
        model_name = "unknown"

    return decode_mode, model_name


def compute_one_file(result_path: Path):
    """
    对单个 results.jsonl 文件计算 metrics
    """

    base_path = strip_jsonl_suffix(result_path)

    metrics_path = Path(str(base_path) + "_metrics.jsonl")
    summary_path = Path(str(base_path) + "_summary.jsonl")

    decode_mode, model_name = extract_model_and_decode(result_path)

    print(f"\n=== Processing {result_path} ===")

    # ------------------------
    # 读取结果
    # ------------------------
    with open(result_path, "r", encoding="utf-8") as f:
        results = [json.loads(line) for line in f]

    print(f"Loaded {len(results)} samples")

    # ------------------------
    # 写 per-sample metrics
    # ------------------------
    with open(metrics_path, "w", encoding="utf-8") as out_f:

        for i, item in enumerate(results):
            pred = item["generated"]
            gold = item["expected"]

            pred_tokens = pred.split()

            metric = {
                "index": i,
                "input": item.get("input", ""),
                "generated": pred,
                "expected": gold,

                # 核心指标
                "bleu": bleu(pred, gold),
                "meteor": meteor(pred, gold),
                "perplexity": item.get("perplexity", float("nan")),

                # 复读指标
                "rep_w": rep_w(pred_tokens, w=10),
                "rep_n_1": rep_n(pred_tokens, n=1),
                "rep_n_2": rep_n(pred_tokens, n=2),
                "rep_n_3": rep_n(pred_tokens, n=3),
                "rep_n_4": rep_n(pred_tokens, n=4),
                "rep_n_5": rep_n(pred_tokens, n=5),
                "rep_r": rep_r(pred_tokens),
            }

            out_f.write(json.dumps(metric, ensure_ascii=False) + "\n")

    print(f"metrics written to {metrics_path}")

    # ------------------------
    # 计算平均值
    # ------------------------
    with open(metrics_path, "r", encoding="utf-8") as f:
        all_metrics = [json.loads(line) for line in f]

    metric_keys = [
        "bleu", "meteor", "perplexity", "rep_w",
        "rep_n_1", "rep_n_2", "rep_n_3", "rep_n_4", "rep_n_5", "rep_r"
    ]

    avg_metrics = {}
    for key in metric_keys:
        values = [m[key] for m in all_metrics]
        avg_metrics[key] = sum(values) / len(values)

    summary = {
        "model": model_name,
        "decode_mode": decode_mode,
        "num_samples": len(all_metrics),
        "avg_metrics": avg_metrics,
        "result_path": str(result_path),
        "metrics_path": str(metrics_path),
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=4)

    print(f"summary written to {summary_path}")
    print("Average Metrics:")
    for k, v in avg_metrics.items():
        print(f"  {k}: {v:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result_dir",
        type=str,
        required=True,
        help="compute metrics for all file in result_dir."
    )
    args = parser.parse_args()

    root = Path(args.result_dir)

    # 遍历所有子目录，寻找 .jsonl 文件
    jsonl_files = list(root.rglob("*.jsonl"))

    print(f"Found {len(jsonl_files)} result files under {root}")

    for file in jsonl_files:
        # 跳过 metrics 和 summary 文件
        if file.name.endswith("_metrics.jsonl"):
            continue
        if file.name.endswith("_summary.jsonl"):
            continue

        compute_one_file(file)


if __name__ == "__main__":
    main()
