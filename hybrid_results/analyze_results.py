import json
import os
import pandas as pd
import numpy as np
from pathlib import Path
from src.metrics import rep_w, rep_n, rep_r, bleu, meteor
import re

def parse_filename(filename):
    """从文件名解析实验设置"""
    # 移除扩展名
    name = filename.replace('.jsonl', '')
    
    # 初始化设置
    settings = {
        'use_penalty': False,
        'use_sae': False,
        'use_neuron': False,
        'decode_strategy': ''
    }
    
    # 检查penalty
    if 'no_penalty' in name:
        settings['use_penalty'] = False
    elif 'penalty' in name:
        settings['use_penalty'] = True
    
    # 检查sae
    if 'no_sae' in name:
        settings['use_sae'] = False
    elif 'sae' in name:
        settings['use_sae'] = True
    
    # 检查neuron
    if 'no_neuron' in name:
        settings['use_neuron'] = False
    elif 'neuron' in name:
        settings['use_neuron'] = True
    
    # 检查解码策略
    for strategy in ['greedy', 'top_k', 'top_p']:
        if strategy in name:
            settings['decode_strategy'] = strategy
            break
    
    return settings


def calculate_metrics(data, tokenizer=None):
    """计算所有指标"""
    metrics = {
        'avg_perplexity': [],
        'bleu_scores': [],
        'meteor_scores': [],
        'rep_w_scores': [],
        'rep_n_scores': [],
        'rep_r_scores': []
    }
    
    for item in data:
        generated = item['generated']
        expected = item['expected']
        
        # 添加perplexity（如果文件中已有）
        if 'perplexity' in item:
            metrics['avg_perplexity'].append(item['perplexity'])
        
        # 计算BLEU
        try:
            bleu_score = bleu(generated, expected)
            metrics['bleu_scores'].append(bleu_score)
        except:
            metrics['bleu_scores'].append(0.0)
        
        # 计算METEOR
        try:
            meteor_score = meteor(generated, expected)
            metrics['meteor_scores'].append(meteor_score)
        except:
            metrics['meteor_scores'].append(0.0)
        
        # 计算重复指标（使用生成的文本的token）
        # 简单按空格分词，实际使用中可能需要更精确的tokenizer
        tokens = generated.split()
        
        if len(tokens) > 0:
            # rep_w (窗口大小设为10)
            rep_w_score = rep_w(tokens, 10)
            metrics['rep_w_scores'].append(rep_w_score)
            
            # rep_n (n-gram设为3)
            rep_n_score = rep_n(tokens, 3)
            metrics['rep_n_scores'].append(rep_n_score)
            
            # rep_r
            rep_r_score = rep_r(tokens)
            metrics['rep_r_scores'].append(rep_r_score)
        else:
            metrics['rep_w_scores'].append(0.0)
            metrics['rep_n_scores'].append(0.0)
            metrics['rep_r_scores'].append(0.0)
    
    # 计算平均值
    result = {}
    for key, values in metrics.items():
        if values:
            result[key.replace('_scores', '').replace('avg_', '')] = np.mean(values)
        else:
            result[key.replace('_scores', '').replace('avg_', '')] = 0.0
    
    return result

def analyze_results(results_dir='hybrid_results'):
    """分析所有结果文件"""
    results_path = Path(results_dir)
    
    if not results_path.exists():
        print(f"结果目录 {results_dir} 不存在")
        return
    
    all_results = []
    
    # 遍历所有jsonl文件
    for file_path in results_path.glob('*.jsonl'):
        print(f"处理文件: {file_path.name}")
        
        # 解析文件名获取实验设置
        settings = parse_filename(file_path.name)
        
        # 读取数据
        data = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data.append(json.loads(line.strip()))
        except Exception as e:
            print(f"读取文件 {file_path.name} 时出错: {e}")
            continue
        
        if not data:
            print(f"文件 {file_path.name} 为空，跳过")
            continue
        
        # 计算指标
        metrics = calculate_metrics(data)
        
        # 合并设置和指标
        result = {
            'filename': file_path.name,
            **settings,
            **metrics,
            'sample_count': len(data)
        }
        
        all_results.append(result)
    
    if not all_results:
        print("没有找到有效的结果文件")
        return
    
    # 创建DataFrame
    df = pd.DataFrame(all_results)
    
    # 重新排列列的顺序
    columns_order = [
        'filename',
        'use_penalty', 'use_sae', 'use_neuron', 'decode_strategy',
        'perplexity', 'bleu', 'meteor',
        'rep_w', 'rep_n', 'rep_r',
        'sample_count'
    ]
    
    # 确保所有列都存在
    for col in columns_order:
        if col not in df.columns:
            df[col] = 0.0 if col.startswith(('perplexity', 'bleu', 'meteor', 'rep_')) else ''
    
    df = df[columns_order]
    
    # 格式化布尔值为勾叉
    for col in ['use_penalty', 'use_sae', 'use_neuron']:
        df[col] = df[col].apply(lambda x: '✓' if x else '✗')
    
    # 格式化数值列，保留4位小数
    numeric_cols = ['perplexity', 'bleu', 'meteor', 'rep_w', 'rep_n', 'rep_r']
    for col in numeric_cols:
        df[col] = df[col].apply(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x)
    
    # 按照实验设置排序
    df = df.sort_values(['use_penalty', 'use_sae', 'use_neuron', 'decode_strategy'])
    
    return df

def save_analysis(df, output_file='analysis_results.csv'):
    """保存分析结果"""
    df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"分析结果已保存到: {output_file}")

def print_summary_table(df):
    """打印格式化的汇总表格"""
    print("\n" + "="*120)
    print("实验结果汇总表")
    print("="*120)
    
    # 设置pandas显示选项
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 20)
    
    print(df.to_string(index=False))
    
    print("\n" + "="*120)
    print("指标说明:")
    print("- Perplexity: 困惑度（越低越好）")
    print("- BLEU: 翻译质量评估（0-100，越高越好）")
    print("- METEOR: 翻译质量评估（0-1，越高越好）")
    print("- Rep_w: 窗口重复率（越低越好）")
    print("- Rep_n: N-gram重复率（越低越好）")
    print("- Rep_r: 相邻重复率（越低越好）")
    print("="*120)

def analyze_by_factors(df):
    """按因子分析结果"""
    print("\n按因子分析:")
    print("-"*60)
    
    # 转换回布尔值进行分析
    df_analysis = df.copy()
    for col in ['use_penalty', 'use_sae', 'use_neuron']:
        df_analysis[col] = df_analysis[col].apply(lambda x: True if x == '✓' else False)
    
    # 转换数值列
    numeric_cols = ['perplexity', 'bleu', 'meteor', 'rep_w', 'rep_n', 'rep_r']
    for col in numeric_cols:
        df_analysis[col] = pd.to_numeric(df_analysis[col], errors='coerce')
    
    factors = ['use_penalty', 'use_sae', 'use_neuron', 'decode_strategy']
    
    for factor in factors:
        print(f"\n{factor.upper()} 影响分析:")
        if factor == 'decode_strategy':
            grouped = df_analysis.groupby(factor)[numeric_cols].mean()
        else:
            grouped = df_analysis.groupby(factor)[numeric_cols].mean()
        
        print(grouped.round(4))

def main():
    """主函数"""
    print("开始分析实验结果...")
    
    # 分析结果
    df = analyze_results()
    
    if df is None or df.empty:
        print("没有找到有效的结果数据")
        return
    
    # 打印汇总表格
    print_summary_table(df)
    
    # 保存结果
    save_analysis(df)
    
    # 按因子分析
    analyze_by_factors(df)
    
    # 找出最佳配置
    print("\n最佳配置分析:")
    print("-"*60)
    
    df_numeric = df.copy()
    numeric_cols = ['perplexity', 'bleu', 'meteor', 'rep_w', 'rep_n', 'rep_r']
    for col in numeric_cols:
        df_numeric[col] = pd.to_numeric(df_numeric[col], errors='coerce')
    
    # 最低困惑度
    best_perplexity = df_numeric.loc[df_numeric['perplexity'].idxmin()]
    print(f"最低困惑度: {best_perplexity['filename']} (Perplexity: {best_perplexity['perplexity']:.4f})")
    
    # 最高BLEU
    best_bleu = df_numeric.loc[df_numeric['bleu'].idxmax()]
    print(f"最高BLEU: {best_bleu['filename']} (BLEU: {best_bleu['bleu']:.4f})")
    
    # 最高METEOR
    best_meteor = df_numeric.loc[df_numeric['meteor'].idxmax()]
    print(f"最高METEOR: {best_meteor['filename']} (METEOR: {best_meteor['meteor']:.4f})")
    
    # 最低重复率（综合rep_w, rep_n, rep_r）
    df_numeric['avg_rep'] = (df_numeric['rep_w'] + df_numeric['rep_n'] + df_numeric['rep_r']) / 3
    best_rep = df_numeric.loc[df_numeric['avg_rep'].idxmin()]
    print(f"最低重复率: {best_rep['filename']} (平均重复率: {best_rep['avg_rep']:.4f})")

if __name__ == '__main__':
    main()
