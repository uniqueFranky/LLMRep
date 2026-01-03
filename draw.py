import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import defaultdict
import math

# Set matplotlib to use English
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def read_all_summaries(result_dir):
    """Read all summary files"""
    root = Path(result_dir)
    summary_files = list(root.rglob("*_summary.jsonl"))
    
    data = []
    for file in summary_files:
        with open(file, 'r', encoding='utf-8') as f:
            summary = json.load(f)
        
        # Extract correct decode_mode from file path
        file_name = file.stem.replace('_summary', '')  # Remove _summary suffix
        if 'greedy' in file_name:
            continue
        # Find model name and extract decode_mode
        if 'gpt2' in file_name:
            model_name = 'GPT-2'
            decode_mode = file_name.replace('_gpt2', '')
        elif 'gemma-2-2b' in file_name:
            model_name = 'Gemma-2-2B'
            decode_mode = file_name.replace('_gemma-2-2b', '')
        else:
            continue
        
        # Extract dataset name from path
        dataset = file.parent.name.upper()
        
        # Update summary information
        summary['model'] = model_name
        summary['decode_mode'] = decode_mode
        summary['dataset'] = dataset
        
        data.append(summary)
    
    return data

def get_decode_modes_and_colors(data):
    """Get all unique decode modes and assign highly distinguishable colors"""
    all_decode_modes_in_data = set([d['decode_mode'] for d in data])
    
    # Define order: Baseline -> Single methods -> Composite methods
    ordered_decode_modes = []
    
    # 1. Baseline methods first
    baseline_methods = ['topp', 'greedy']
    for method in baseline_methods:
        if method in all_decode_modes_in_data:
            ordered_decode_modes.append(method)
    
    # 2. Single methods in the middle
    single_methods = ['sae', 'neuron', 'penalty']
    for method in single_methods:
        if method in all_decode_modes_in_data:
            ordered_decode_modes.append(method)
    
    # 3. Composite methods at the end
    composite_methods = ['sae_penalty', 'neuron_penalty']
    for method in composite_methods:
        if method in all_decode_modes_in_data:
            ordered_decode_modes.append(method)
    
    # Add any remaining methods not in the predefined lists
    for method in sorted(all_decode_modes_in_data):
        if method not in ordered_decode_modes:
            ordered_decode_modes.append(method)
    
    # Highly distinguishable color palette with clear visual separation
    color_scheme = {
        'topp': '#808080',          # Gray - Baseline (Top-p)
        'greedy': '#C0C0C0',        # Silver - Baseline (Greedy)
        'sae': '#1E90FF',           # Dodger Blue - SAE (明亮的蓝色)
        'neuron': '#FF6B35',        # Orange Red - Neuron (鲜艳的橙色)
        'penalty': '#32CD32',       # Lime Green - Penalty (鲜绿色)
        'sae_penalty': '#9370DB',   # Medium Purple - SAE+Penalty (紫色)
        'neuron_penalty': '#DC143C' # Crimson - Neuron+Penalty (深红色)
    }
    
    # Get colors in order
    colors = [color_scheme.get(mode, '#000000') for mode in ordered_decode_modes]
    
    decode_mode_names = {
        'penalty': 'Penalty',
        'neuron': 'Neuron',
        'sae_penalty': 'SAE+Penalty',
        'neuron_penalty': 'Neuron+Penalty',
        'greedy': 'Baseline (Greedy)',
        'sae': 'SAE',
        'topp': 'Baseline'
    }
    
    # Print color legend for presentation reference
    print("\n" + "="*60)
    print("COLOR LEGEND FOR PRESENTATION:")
    print("="*60)
    for mode in ordered_decode_modes:
        display_name = decode_mode_names.get(mode, mode.title())
        color = color_scheme.get(mode, '#000000')
        color_description = {
            '#808080': 'Gray',
            '#C0C0C0': 'Silver',
            '#1E90FF': 'Bright Blue',
            '#FF6B35': 'Orange',
            '#32CD32': 'Green',
            '#9370DB': 'Purple',
            '#DC143C': 'Red'
        }
        print(f"  {display_name:25s} → {color_description.get(color, 'Unknown'):15s} ({color})")
    print("="*60 + "\n")
    
    return ordered_decode_modes, colors, decode_mode_names

def create_comparison_plots(data):
    """Create comparison plots with selected metrics including PPL with adjusted scale"""
    # Get all decode modes and colors
    all_decode_modes, colors, decode_mode_names = get_decode_modes_and_colors(data)
    
    # Organize data
    datasets = sorted(list(set([d['dataset'] for d in data])))
    models = sorted(list(set([d['model'] for d in data])))
    
    # Select only 5 metrics to display
    metrics_to_plot = ['bleu', 'perplexity', 'rep_w', 'rep_r', 'rep_n_avg']
    metric_names = {
        'bleu': 'BLEU',
        'perplexity': 'PPL',
        'rep_w': 'Rep-W',
        'rep_r': 'Rep-R',
        'rep_n_avg': 'Rep-N-Avg'
    }
    
    # Create plots for datasets
    fig, axes = plt.subplots(len(datasets), len(models), figsize=(15*len(models), 6*len(datasets)))
    if len(datasets) == 1:
        axes = axes.reshape(1, -1)
    if len(models) == 1:
        axes = axes.reshape(-1, 1)
    
    fig.suptitle('Model Performance Comparison: Key Metrics (BLEU, PPL, Rep-W, Rep-R, Rep-N-Avg)', 
                 fontsize=18, fontweight='bold')
    
    for dataset_idx, dataset in enumerate(datasets):
        for model_idx, model in enumerate(models):
            ax = axes[dataset_idx, model_idx]
            
            # Filter data for current dataset and model
            filtered_data = [d for d in data if d['dataset'] == dataset and d['model'] == model]
            
            # Organize data for plotting - maintain order from all_decode_modes
            metrics_data = defaultdict(list)
            decode_mode_labels = []
            used_colors = []
            
            for decode_mode in all_decode_modes:
                mode_data = [d for d in filtered_data if d['decode_mode'] == decode_mode]
                if mode_data:
                    display_name = decode_mode_names.get(decode_mode, decode_mode.title())
                    decode_mode_labels.append(display_name)
                    color_idx = all_decode_modes.index(decode_mode)
                    used_colors.append(colors[color_idx])
                    
                    for metric in metrics_to_plot:
                        if metric == 'rep_n_avg':
                            # 计算 Rep-N-1 到 Rep-N-5 的平均值
                            rep_n_values = []
                            for i in range(1, 6):
                                val = mode_data[0]['avg_metrics'].get(f'rep_n_{i}', 0)
                                if val != float('inf') and val == val:  # Not inf and not NaN
                                    rep_n_values.append(val)
                            value = np.mean(rep_n_values) if rep_n_values else 0
                        else:
                            value = mode_data[0]['avg_metrics'].get(metric, 0)
                            # Handle infinity and NaN values
                            if value == float('inf') or value != value or math.isnan(value):
                                value = 0
                        
                        # 特殊处理PPL：缩放到合适范围以便可视化
                        if metric == 'perplexity' and value > 0:
                            # 假设PPL的合理范围是0-6，将其缩放到0-2范围
                            value = min(value / 3.0, 2.0)
                        
                        metrics_data[metric].append(value)
            
            if not decode_mode_labels:  # No data for this combination
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{dataset} Dataset - {model} Model', fontsize=14, fontweight='bold')
                continue
            
            # Create grouped bar chart
            x = np.arange(len(metrics_to_plot))
            width = 0.15  # Adjusted for fewer methods
            
            for i, decode_mode in enumerate(decode_mode_labels):
                values = [metrics_data[metric][i] for metric in metrics_to_plot]
                offset = (i - len(decode_mode_labels)/2 + 0.5) * width
                bars = ax.bar(x + offset, values, width, label=decode_mode, color=used_colors[i], 
                             alpha=0.85, edgecolor='black', linewidth=0.8)
                
                # 为PPL柱子添加实际数值标签
                for j, (bar, metric) in enumerate(zip(bars, metrics_to_plot)):
                    if metric == 'perplexity':
                        # 显示原始PPL值
                        original_ppl = metrics_data[metric][i] * 3.0  # 还原原始值
                        if original_ppl > 0:
                            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05,
                                   f'{original_ppl:.1f}',
                                   ha='center', va='bottom', fontsize=8, fontweight='bold',
                                   color='red')
            
            ax.set_xlabel('Metrics', fontsize=12)
            ax.set_ylabel('Score', fontsize=12)
            ax.set_title(f'{dataset} Dataset - {model} Model', fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels([metric_names[m] for m in metrics_to_plot], rotation=0, ha='center')
            ax.legend(fontsize=10, loc='upper right', framealpha=0.9)
            ax.grid(True, alpha=0.3, axis='y')
            
            # 设置y轴范围为0-2，给图表更多空间
            ax.set_ylim(0, 2.7)
            
            # 添加PPL说明文本
            ax.text(0.02, 0.98, 'PPL values shown in red\n(scaled: PPL/3)', 
                   transform=ax.transAxes, fontsize=8, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('model_comparison_complete_all_metrics_with_ppl.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_quality_metrics_detailed_plot(data):
    """创建质量指标（BLEU, METEOR, PPL）的详细对比图"""
    all_decode_modes, colors, decode_mode_names = get_decode_modes_and_colors(data)
    datasets = sorted(list(set([d['dataset'] for d in data])))
    models = sorted(list(set([d['model'] for d in data])))
    
    fig, axes = plt.subplots(len(datasets), len(models), figsize=(12*len(models), 6*len(datasets)))
    if len(datasets) == 1:
        axes = axes.reshape(1, -1)
    if len(models) == 1:
        axes = axes.reshape(-1, 1)
    
    fig.suptitle('Quality Metrics Detailed Comparison (BLEU, METEOR, PPL)', fontsize=18, fontweight='bold')
    
    for dataset_idx, dataset in enumerate(datasets):
        for model_idx, model in enumerate(models):
            ax = axes[dataset_idx, model_idx]
            
            filtered_data = [d for d in data if d['dataset'] == dataset and d['model'] == model]
            
            # 准备数据
            quality_metrics = ['bleu', 'meteor', 'perplexity']
            x_labels = ['BLEU', 'METEOR', 'PPL']
            x = np.arange(len(quality_metrics))
            width = 0.12
            
            # 存储原始PPL值用于显示
            original_ppl_values = {}
            
            # Maintain order from all_decode_modes
            for i, decode_mode in enumerate(all_decode_modes):
                mode_data = [d for d in filtered_data if d['decode_mode'] == decode_mode]
                if mode_data:
                    display_name = decode_mode_names.get(decode_mode, decode_mode.title())
                    color_idx = all_decode_modes.index(decode_mode)
                    
                    values = []
                    for metric in quality_metrics:
                        val = mode_data[0]['avg_metrics'].get(metric, 0)
                        if val == float('inf') or val != val or math.isnan(val):
                            val = 0
                        
                        # 特殊处理PPL：缩放到0-3范围
                        if metric == 'perplexity' and val > 0:
                            original_ppl_values[display_name] = val  # 保存原始值
                            # 假设PPL的合理范围是0-6，将其缩放到0-2
                            val = min(val / 3.0, 2.0)
                        
                        values.append(val)
                    
                    offset = (i - len(all_decode_modes)/2 + 0.5) * width
                    bars = ax.bar(x + offset, values, width, label=display_name, 
                                 color=colors[color_idx], alpha=0.85, edgecolor='black', linewidth=0.5)
                    
                    # 为PPL柱子添加原始数值标签
                    for j, (bar, metric) in enumerate(zip(bars, quality_metrics)):
                        if metric == 'perplexity' and display_name in original_ppl_values:
                            original_val = original_ppl_values[display_name]
                            if original_val > 0:
                                ax.text(bar.get_x() + bar.get_width()/3., bar.get_height() + 0.05,
                                       f'{original_val:.1f}',
                                       ha='center', va='bottom', fontsize=8, fontweight='bold',
                                       color='red')
            
            ax.set_xlabel('Quality Metrics', fontsize=12)
            ax.set_ylabel('Score', fontsize=12)
            ax.set_title(f'{dataset} Dataset - {model} Model', fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(x_labels)
            ax.legend(fontsize=9, loc='upper right', framealpha=0.9)
            ax.grid(True, alpha=0.3, axis='y')
            
            # 设置y轴范围，给PPL足够空间
            ax.set_ylim(0, 3)
            
            # 添加PPL说明文本
            ax.text(0.02, 0.98, 'PPL values shown in red\n(scaled: PPL/3)', 
                   transform=ax.transAxes, fontsize=8, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('quality_metrics_detailed_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


def create_repn_detailed_plot(data):
    """创建 Rep-N 指标的详细对比图"""
    all_decode_modes, colors, decode_mode_names = get_decode_modes_and_colors(data)
    datasets = sorted(list(set([d['dataset'] for d in data])))
    models = sorted(list(set([d['model'] for d in data])))
    
    fig, axes = plt.subplots(len(datasets), len(models), figsize=(10*len(models), 6*len(datasets)))
    if len(datasets) == 1:
        axes = axes.reshape(1, -1)
    if len(models) == 1:
        axes = axes.reshape(-1, 1)
    
    fig.suptitle('Rep-N Metrics Detailed Comparison (Rep-N-1 to Rep-N-5)', fontsize=18, fontweight='bold')
    
    for dataset_idx, dataset in enumerate(datasets):
        for model_idx, model in enumerate(models):
            ax = axes[dataset_idx, model_idx]
            
            filtered_data = [d for d in data if d['dataset'] == dataset and d['model'] == model]
            
            # 为每个 decode mode 绘制折线图 - maintain order
            for decode_mode in all_decode_modes:
                mode_data = [d for d in filtered_data if d['decode_mode'] == decode_mode]
                if mode_data:
                    display_name = decode_mode_names.get(decode_mode, decode_mode.title())
                    color_idx = all_decode_modes.index(decode_mode)
                    
                    rep_n_values = []
                    for i in range(1, 6):
                        val = mode_data[0]['avg_metrics'].get(f'rep_n_{i}', 0)
                        if val == float('inf') or val != val:
                            val = 0
                        rep_n_values.append(val)
                    
                    x = [1, 2, 3, 4, 5]
                    ax.plot(x, rep_n_values, marker='o', label=display_name, 
                           color=colors[color_idx], linewidth=2.5, markersize=8, alpha=0.85)
            
            ax.set_xlabel('N-gram Size', fontsize=12)
            ax.set_ylabel('Repetition Score', fontsize=12)
            ax.set_title(f'{dataset} Dataset - {model} Model', fontsize=14, fontweight='bold')
            ax.set_xticks([1, 2, 3, 4, 5])
            ax.legend(fontsize=9, loc='best', framealpha=0.9)
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('repn_detailed_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_all_repetition_metrics_plot(data):
    """创建所有重复度指标的综合对比图"""
    all_decode_modes, colors, decode_mode_names = get_decode_modes_and_colors(data)
    datasets = sorted(list(set([d['dataset'] for d in data])))
    models = sorted(list(set([d['model'] for d in data])))
    
    fig, axes = plt.subplots(len(datasets), len(models), figsize=(12*len(models), 6*len(datasets)))
    if len(datasets) == 1:
        axes = axes.reshape(1, -1)
    if len(models) == 1:
        axes = axes.reshape(-1, 1)
    
    fig.suptitle('All Repetition Metrics Comparison (Rep-W, Rep-R, Rep-N-1 to Rep-N-5)', fontsize=18, fontweight='bold')
    
    for dataset_idx, dataset in enumerate(datasets):
        for model_idx, model in enumerate(models):
            ax = axes[dataset_idx, model_idx]
            
            filtered_data = [d for d in data if d['dataset'] == dataset and d['model'] == model]
            
            # 准备数据
            rep_metrics = ['rep_w', 'rep_r', 'rep_n_1', 'rep_n_2', 'rep_n_3', 'rep_n_4', 'rep_n_5']
            x_labels = ['Rep-W', 'Rep-R', 'N-1', 'N-2', 'N-3', 'N-4', 'N-5']
            x = np.arange(len(rep_metrics))
            width = 0.12
            
            # Maintain order from all_decode_modes
            for i, decode_mode in enumerate(all_decode_modes):
                mode_data = [d for d in filtered_data if d['decode_mode'] == decode_mode]
                if mode_data:
                    display_name = decode_mode_names.get(decode_mode, decode_mode.title())
                    color_idx = all_decode_modes.index(decode_mode)
                    
                    values = []
                    for metric in rep_metrics:
                        val = mode_data[0]['avg_metrics'].get(metric, 0)
                        if val == float('inf') or val != val:
                            val = 0
                        values.append(val)
                    
                    offset = (i - len(all_decode_modes)/2 + 0.5) * width
                    ax.bar(x + offset, values, width, label=display_name, 
                          color=colors[color_idx], alpha=0.85, edgecolor='black', linewidth=0.5)
            
            ax.set_xlabel('Repetition Metrics', fontsize=12)
            ax.set_ylabel('Score (Lower is Better)', fontsize=12)
            ax.set_title(f'{dataset} Dataset - {model} Model', fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(x_labels, rotation=45, ha='right')
            ax.legend(fontsize=9, loc='upper right', framealpha=0.9)
            ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('all_repetition_metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_method_ranking_analysis(df):
    """Create method ranking analysis - 包含PPL和所有重复度指标"""
    # Define method order: Baseline -> Single -> Composite
    method_order = ['Baseline', 'Baseline (Greedy)', 'SAE', 'Neuron', 'Penalty', 'SAE+Penalty', 'Neuron+Penalty']
    
    # Map decode_mode to display names for matching
    decode_mode_to_display = {
        'topp': 'Baseline',
        'greedy': 'Baseline (Greedy)',
        'sae': 'SAE',
        'neuron': 'Neuron',
        'penalty': 'Penalty',
        'sae_penalty': 'SAE+Penalty',
        'neuron_penalty': 'Neuron+Penalty'
    }
    
    # Add display name column to df
    df['Display_Name'] = df['Decode_Mode'].map(decode_mode_to_display)
    
    # Filter method_order to only include methods present in data
    available_methods = [m for m in method_order if m in df['Display_Name'].values]
    
    fig, axes = plt.subplots(3, 3, figsize=(20, 16))
    fig.suptitle('Comprehensive Decode Method Performance Analysis (with PPL)', fontsize=18, fontweight='bold')
    
    # Define colors matching the order - highly distinguishable
    method_colors = {
        'Baseline': '#808080',      # Gray
        'Baseline (Greedy)': '#C0C0C0',     # Silver
        'SAE': '#1E90FF',                   # Bright Blue
        'Neuron': '#FF6B35',                # Orange
        'Penalty': '#32CD32',               # Green
        'SAE+Penalty': '#9370DB',           # Purple
        'Neuron+Penalty': '#DC143C'         # Red
    }
    
    # Plot 1: Quality metrics by method - BLEU, METEOR, PPL
    ax1 = axes[0, 0]
    x = np.arange(len(available_methods))
    width = 0.25
    
    bleu_scores = []
    meteor_scores = []
    ppl_scores = []
    
    for method in available_methods:
        method_data = df[df['Display_Name'] == method]
        if not method_data.empty:
            bleu_scores.append(method_data['BLEU'].mean())
            meteor_scores.append(method_data['METEOR'].mean())
            # PPL处理：过滤掉inf和nan值
            ppl_values = method_data['PERPLEXITY'].replace([np.inf, -np.inf], np.nan).dropna()
            ppl_scores.append(ppl_values.mean() if not ppl_values.empty else 0)
        else:
            bleu_scores.append(0)
            meteor_scores.append(0)
            ppl_scores.append(0)
    
    # 绘制三个质量指标
    ax1.bar(x - width, bleu_scores, width, label='BLEU', alpha=0.85, color='#4169E1', edgecolor='black', linewidth=0.8)
    ax1.bar(x, meteor_scores, width, label='METEOR', alpha=0.85, color='#32CD32', edgecolor='black', linewidth=0.8)
    
    # PPL使用右轴（因为数值范围可能差异很大）
    ax1_twin = ax1.twinx()
    ax1_twin.bar(x + width, ppl_scores, width, label='PPL', alpha=0.85, color='#DC143C', edgecolor='black', linewidth=0.8)
    
    ax1.set_xlabel('Decode Method', fontsize=11)
    ax1.set_ylabel('BLEU/METEOR Score', color='#4169E1', fontsize=11, fontweight='bold')
    ax1_twin.set_ylabel('PPL Score (Lower is Better)', color='#DC143C', fontsize=11, fontweight='bold')
    ax1.set_title('Quality Metrics by Method', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(available_methods, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.legend(loc='upper left')
    ax1_twin.legend(loc='upper right')
    
    # Plot 2: Rep-W and Rep-R comparison
    ax2 = axes[0, 1]
    rep_w_scores = []
    rep_r_scores = []
    
    for method in available_methods:
        method_data = df[df['Display_Name'] == method]
        if not method_data.empty:
            rep_w_scores.append(method_data['REP_W'].mean())
            rep_r_scores.append(method_data['REP_R'].mean())
        else:
            rep_w_scores.append(0)
            rep_r_scores.append(0)
    
    ax2.bar(x - width/2, rep_w_scores, width, label='Rep-W', alpha=0.85, color='#FF8C00',
           edgecolor='black', linewidth=0.8)
    ax2.bar(x + width/2, rep_r_scores, width, label='Rep-R', alpha=0.85, color='#FF4500',
           edgecolor='black', linewidth=0.8)
    ax2.set_xlabel('Decode Method', fontsize=11)
    ax2.set_ylabel('Repetition Score', fontsize=11)
    ax2.set_title('Rep-W and Rep-R by Method (Lower is Better)', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(available_methods, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Rep-N metrics line plot
    ax3 = axes[0, 2]
    rep_n_metrics = ['REP_N_1', 'REP_N_2', 'REP_N_3', 'REP_N_4', 'REP_N_5']
    
    for method in available_methods:
        method_data = df[df['Display_Name'] == method]
        if not method_data.empty:
            rep_n_values = [method_data[metric].mean() for metric in rep_n_metrics]
            ax3.plot([1, 2, 3, 4, 5], rep_n_values, marker='o', label=method, 
                    linewidth=2.5, markersize=8, color=method_colors.get(method, '#cccccc'))
    
    ax3.set_xlabel('N-gram Size', fontsize=11)
    ax3.set_ylabel('Average Repetition Score', fontsize=11)
    ax3.set_title('Rep-N Metrics by Method (Lower is Better)', fontsize=12, fontweight='bold')
    ax3.set_xticks([1, 2, 3, 4, 5])
    ax3.legend(fontsize=9, framealpha=0.9)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: All repetition metrics average
    ax4 = axes[1, 0]
    all_rep_metrics = ['REP_W', 'REP_R'] + rep_n_metrics
    rep_avg_scores = []
    
    for method in available_methods:
        method_data = df[df['Display_Name'] == method]
        if not method_data.empty:
            avg_rep = method_data[all_rep_metrics].mean().mean()
            rep_avg_scores.append(avg_rep)
        else:
            rep_avg_scores.append(0)
    
    ax4.bar(x, rep_avg_scores, color=[method_colors.get(m, '#cccccc') for m in available_methods], 
           alpha=0.85, edgecolor='black', linewidth=0.8)
    ax4.set_xlabel('Decode Method', fontsize=11)
    ax4.set_ylabel('Average Repetition Score', fontsize=11)
    ax4.set_title('Overall Repetition Score by Method', fontsize=12, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(available_methods, rotation=45, ha='right')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Plot 5: Model comparison - 包含PPL
    ax5 = axes[1, 1]
    models = sorted(df['Model'].unique())
    model_bleu = df.groupby('Model')['BLEU'].mean()
    model_rep_avg = df.groupby('Model')[all_rep_metrics].mean().mean(axis=1)
    # PPL处理
    model_ppl = df.groupby('Model')['PERPLEXITY'].apply(lambda x: x.replace([np.inf, -np.inf], np.nan).dropna().mean())
    
    x_models = np.arange(len(models))
    ax5_twin = ax5.twinx()
    
    bars1 = ax5.bar(x_models - 0.25, model_bleu.values, 0.25, label='BLEU', alpha=0.85, 
                   color='#4169E1', edgecolor='black', linewidth=0.8)
    bars2 = ax5.bar(x_models, model_rep_avg.values, 0.25, label='Rep-Avg', alpha=0.85, 
                   color='#32CD32', edgecolor='black', linewidth=0.8)
    bars3 = ax5_twin.bar(x_models + 0.25, model_ppl.values, 0.25, label='PPL', 
                        alpha=0.85, color='#DC143C', edgecolor='black', linewidth=0.8)
    
    ax5.set_xlabel('Model', fontsize=11)
    ax5.set_ylabel('BLEU/Rep-Avg Score', color='#4169E1', fontsize=11, fontweight='bold')
    ax5_twin.set_ylabel('PPL Score', color='#DC143C', fontsize=11, fontweight='bold')
    ax5.set_title('Model Performance Comparison (with PPL)', fontsize=12, fontweight='bold')
    ax5.set_xticks(x_models)
    ax5.set_xticklabels(models)
    ax5.grid(True, alpha=0.3, axis='y')
    ax5.legend(loc='upper left')
    ax5_twin.legend(loc='upper right')
    
    # Plot 6: Dataset comparison - Rep metrics
    ax6 = axes[1, 2]
    datasets = sorted(df['Dataset'].unique())
    dataset_rep_stats = df.groupby('Dataset')[['REP_W', 'REP_R']].mean()
    
    x_datasets = np.arange(len(datasets))
    width_dataset = 0.35
    
    ax6.bar(x_datasets - width_dataset/2, dataset_rep_stats['REP_W'], width_dataset, 
           label='Rep-W', alpha=0.85, color='#FF8C00', edgecolor='black', linewidth=0.8)
    ax6.bar(x_datasets + width_dataset/2, dataset_rep_stats['REP_R'], width_dataset, 
           label='Rep-R', alpha=0.85, color='#FF4500', edgecolor='black', linewidth=0.8)
    
    ax6.set_xlabel('Dataset', fontsize=11)
    ax6.set_ylabel('Score', fontsize=11)
    ax6.set_title('Rep-W and Rep-R by Dataset', fontsize=12, fontweight='bold')
    ax6.set_xticks(x_datasets)
    ax6.set_xticklabels(datasets)
    ax6.legend(fontsize=10)
    ax6.grid(True, alpha=0.3, axis='y')
    
    # Plot 7: Heatmap of all repetition metrics by method
    ax7 = axes[2, 0]
    heatmap_data = []
    heatmap_labels = ['W', 'R', 'N-1', 'N-2', 'N-3', 'N-4', 'N-5']
    
    for method in available_methods:
        method_data = df[df['Display_Name'] == method]
        if not method_data.empty:
            row = [method_data['REP_W'].mean(), method_data['REP_R'].mean()] + \
                  [method_data[metric].mean() for metric in rep_n_metrics]
            heatmap_data.append(row)
        else:
            heatmap_data.append([0] * 7)
    
    im = ax7.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
    ax7.set_xticks(np.arange(7))
    ax7.set_yticks(np.arange(len(available_methods)))
    ax7.set_xticklabels(heatmap_labels)
    ax7.set_yticklabels(available_methods)
    ax7.set_title('Repetition Metrics Heatmap by Method', fontsize=12, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax7)
    cbar.set_label('Score', rotation=270, labelpad=15)
    
    # Add text annotations
    for i in range(len(available_methods)):
        for j in range(7):
            text = ax7.text(j, i, f'{heatmap_data[i][j]:.2f}',
                          ha="center", va="center", color="black", fontsize=8)
    
    # Plot 8: PPL comparison by method
    ax8 = axes[2, 1]
    ppl_scores_clean = []
    
    for method in available_methods:
        method_data = df[df['Display_Name'] == method]
        if not method_data.empty:
            ppl_values = method_data['PERPLEXITY'].replace([np.inf, -np.inf], np.nan).dropna()
            ppl_scores_clean.append(ppl_values.mean() if not ppl_values.empty else 0)
        else:
            ppl_scores_clean.append(0)
    
    bars = ax8.bar(x, ppl_scores_clean, color=[method_colors.get(m, '#cccccc') for m in available_methods], 
                   alpha=0.85, edgecolor='black', linewidth=0.8)
    
    # 添加数值标签
    for i, (bar, score) in enumerate(zip(bars, ppl_scores_clean)):
        height = bar.get_height()
        ax8.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.2f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax8.set_xlabel('Decode Method', fontsize=11)
    ax8.set_ylabel('PPL Score (Lower is Better)', fontsize=11)
    ax8.set_title('Perplexity by Method', fontsize=12, fontweight='bold')
    ax8.set_xticks(x)
    ax8.set_xticklabels(available_methods, rotation=45, ha='right')
    ax8.grid(True, alpha=0.3, axis='y')
    ax8.set_ylim(0, max(ppl_scores_clean) * 1.15 if max(ppl_scores_clean) > 0 else 1)
    
    # Plot 9: Quality vs Repetition scatter plot (包含PPL信息)
    ax9 = axes[2, 2]
    
    for method in available_methods:
        method_data = df[df['Display_Name'] == method]
        if not method_data.empty:
            bleu_val = method_data['BLEU'].mean()
            rep_avg_val = method_data[all_rep_metrics].mean().mean()
            ppl_values = method_data['PERPLEXITY'].replace([np.inf, -np.inf], np.nan).dropna()
            ppl_val = ppl_values.mean() if not ppl_values.empty else 0
            
            # 使用PPL值来调整点的大小（PPL越低，点越大）
            size = max(50, 500 - ppl_val * 50) if ppl_val > 0 else 250
            
            ax9.scatter(bleu_val, rep_avg_val, s=size, alpha=0.8, 
                       color=method_colors.get(method, '#cccccc'), 
                       label=method, edgecolors='black', linewidth=1.5)
            ax9.annotate(f'{method}\n(PPL:{ppl_val:.1f})', (bleu_val, rep_avg_val), fontsize=8, 
                        xytext=(5, 5), textcoords='offset points', fontweight='bold')
    
    ax9.set_xlabel('BLEU Score (Higher is Better)', fontsize=11)
    ax9.set_ylabel('Avg Repetition Score (Lower is Better)', fontsize=11)
    ax9.set_title('Quality vs Repetition Trade-off\n(Point size ∝ 1/PPL)', fontsize=12, fontweight='bold')
    ax9.grid(True, alpha=0.3)
    ax9.legend(fontsize=8, loc='best', framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig('comprehensive_analysis_all_metrics_with_ppl.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_detailed_analysis(data):
    """Create detailed analysis table - 包含PPL处理和baseline提升计算"""
    df_list = []
    
    for item in data:
        row = {
            'Dataset': item['dataset'],
            'Model': item['model'],
            'Decode_Mode': item['decode_mode'],
            'Num_Samples': item['num_samples']
        }
        
        # Add all metrics
        for metric, value in item['avg_metrics'].items():
            if metric.lower() == 'perplexity':
                # 特殊处理PPL：将inf和nan设为0
                if value == float('inf') or value != value or math.isnan(value):
                    value = 0
            elif value == float('inf') or value != value:
                value = 0
            row[metric.upper()] = value
        
        df_list.append(row)
    
    df = pd.DataFrame(df_list)
    
    # Save detailed results
    df.to_csv('detailed_results_complete_with_ppl.csv', index=False)
    
    # Print summary statistics
    print("=== Complete Experiment Results Summary (with PPL) ===")
    print(f"Total experimental configurations processed: {len(df)}")
    print(f"Datasets: {sorted(df['Dataset'].unique())}")
    print(f"Models: {sorted(df['Model'].unique())}")
    print(f"Decode Modes: {sorted(df['Decode_Mode'].unique())}")
    
    # Method performance ranking - 包含PPL
    print("\n=== Method Performance Ranking (by BLEU, including PPL) ===")
    method_performance = df.groupby('Decode_Mode').agg({
        'BLEU': 'mean',
        'METEOR': 'mean',
        'PERPLEXITY': lambda x: x.replace([np.inf, -np.inf], np.nan).dropna().mean(),  # PPL特殊处理
        'REP_W': 'mean',
        'REP_R': 'mean',
        'REP_N_1': 'mean',
        'REP_N_2': 'mean',
        'REP_N_3': 'mean',
        'REP_N_4': 'mean',
        'REP_N_5': 'mean'
    }).round(4)
    method_performance = method_performance.sort_values('BLEU', ascending=False)
    print(method_performance)
    
    # Rep-N Average
    print("\n=== Rep-N Average by Method ===")
    rep_n_cols = ['REP_N_1', 'REP_N_2', 'REP_N_3', 'REP_N_4', 'REP_N_5']
    df['REP_N_AVG'] = df[rep_n_cols].mean(axis=1)
    rep_n_summary = df.groupby('Decode_Mode')['REP_N_AVG'].mean().round(4).sort_values()
    print(rep_n_summary)
    
    # All repetition metrics average
    print("\n=== All Repetition Metrics Average by Method ===")
    all_rep_cols = ['REP_W', 'REP_R'] + rep_n_cols
    df['REP_ALL_AVG'] = df[all_rep_cols].mean(axis=1)
    rep_all_summary = df.groupby('Decode_Mode')['REP_ALL_AVG'].mean().round(4).sort_values()
    print(rep_all_summary)
    
    # PPL summary
    print("\n=== Perplexity Summary by Method ===")
    ppl_summary = df.groupby('Decode_Mode')['PERPLEXITY'].apply(
        lambda x: x.replace([np.inf, -np.inf], np.nan).dropna().mean()
    ).round(4).sort_values()
    print(ppl_summary)
    
    # ===== 新增：相对于baseline的提升计算并写入文件 =====
    improvement_output = []
    improvement_output.append("="*80)
    improvement_output.append("IMPROVEMENT ANALYSIS: Performance vs Baseline (topp)")
    improvement_output.append("="*80)
    
    # 计算每个方法相对于baseline的提升
    baseline_method = 'topp'
    
    if baseline_method in df['Decode_Mode'].values:
        # 获取baseline性能
        baseline_stats = df[df['Decode_Mode'] == baseline_method].groupby(['Dataset', 'Model']).agg({
            'BLEU': 'mean',
            'METEOR': 'mean',
            'PERPLEXITY': lambda x: x.replace([np.inf, -np.inf], np.nan).dropna().mean(),
            'REP_W': 'mean',
            'REP_R': 'mean',
            'REP_N_AVG': 'mean',
            'REP_ALL_AVG': 'mean'
        })
        
        # 计算所有方法的性能
        all_methods_stats = df.groupby(['Dataset', 'Model', 'Decode_Mode']).agg({
            'BLEU': 'mean',
            'METEOR': 'mean',
            'PERPLEXITY': lambda x: x.replace([np.inf, -np.inf], np.nan).dropna().mean(),
            'REP_W': 'mean',
            'REP_R': 'mean',
            'REP_N_AVG': 'mean',
            'REP_ALL_AVG': 'mean'
        })
        
        # 计算改进百分比
        improvements = {}
        
        for (dataset, model, method), method_stats in all_methods_stats.iterrows():
            if method == baseline_method:
                continue
                
            if (dataset, model) in baseline_stats.index:
                baseline_perf = baseline_stats.loc[(dataset, model)]
                
                method_key = f"{method}_{dataset}_{model}"
                improvements[method_key] = {
                    'method': method,
                    'dataset': dataset,
                    'model': model,
                    'bleu_improve': ((method_stats['BLEU'] - baseline_perf['BLEU']) / baseline_perf['BLEU'] * 100) if baseline_perf['BLEU'] > 0 else 0,
                    'meteor_improve': ((method_stats['METEOR'] - baseline_perf['METEOR']) / baseline_perf['METEOR'] * 100) if baseline_perf['METEOR'] > 0 else 0,
                    'ppl_improve': ((baseline_perf['PERPLEXITY'] - method_stats['PERPLEXITY']) / baseline_perf['PERPLEXITY'] * 100) if baseline_perf['PERPLEXITY'] > 0 and not math.isnan(baseline_perf['PERPLEXITY']) and not math.isnan(method_stats['PERPLEXITY']) else 0,
                    'rep_w_improve': ((baseline_perf['REP_W'] - method_stats['REP_W']) / baseline_perf['REP_W'] * 100) if baseline_perf['REP_W'] > 0 else 0,
                    'rep_r_improve': ((baseline_perf['REP_R'] - method_stats['REP_R']) / baseline_perf['REP_R'] * 100) if baseline_perf['REP_R'] > 0 else 0,
                    'rep_n_improve': ((baseline_perf['REP_N_AVG'] - method_stats['REP_N_AVG']) / baseline_perf['REP_N_AVG'] * 100) if baseline_perf['REP_N_AVG'] > 0 else 0,
                    'rep_all_improve': ((baseline_perf['REP_ALL_AVG'] - method_stats['REP_ALL_AVG']) / baseline_perf['REP_ALL_AVG'] * 100) if baseline_perf['REP_ALL_AVG'] > 0 else 0,
                }
        
        # 按方法分组并计算平均改进
        method_improvements = defaultdict(list)
        for imp_data in improvements.values():
            method_improvements[imp_data['method']].append(imp_data)
        
        # 添加详细结果到输出
        improvement_output.append(f"\nBaseline Method: {baseline_method.upper()}")
        improvement_output.append("-" * 80)
        
        for method, imp_list in method_improvements.items():
            improvement_output.append(f"\n### {method.upper()} vs {baseline_method.upper()} ###")
            
            # 计算平均改进
            avg_improvements = {
                'bleu': np.mean([imp['bleu_improve'] for imp in imp_list]),
                'meteor': np.mean([imp['meteor_improve'] for imp in imp_list]),
                'ppl': np.mean([imp['ppl_improve'] for imp in imp_list]),
                'rep_w': np.mean([imp['rep_w_improve'] for imp in imp_list]),
                'rep_r': np.mean([imp['rep_r_improve'] for imp in imp_list]),
                'rep_n': np.mean([imp['rep_n_improve'] for imp in imp_list]),
                'rep_all': np.mean([imp['rep_all_improve'] for imp in imp_list])
            }
            
            improvement_output.append(f"Average Improvements:")
            improvement_output.append(f"  BLEU:        {avg_improvements['bleu']:+7.2f}%")
            improvement_output.append(f"  METEOR:      {avg_improvements['meteor']:+7.2f}%")
            improvement_output.append(f"  PPL:         {avg_improvements['ppl']:+7.2f}% (reduction)")
            improvement_output.append(f"  Rep-W:       {avg_improvements['rep_w']:+7.2f}% (reduction)")
            improvement_output.append(f"  Rep-R:       {avg_improvements['rep_r']:+7.2f}% (reduction)")
            improvement_output.append(f"  Rep-N-Avg:   {avg_improvements['rep_n']:+7.2f}% (reduction)")
            improvement_output.append(f"  Rep-All-Avg: {avg_improvements['rep_all']:+7.2f}% (reduction)")
            
            # 添加每个数据集-模型组合的详细结果
            improvement_output.append(f"\nDetailed Results by Dataset-Model:")
            for imp in imp_list:
                improvement_output.append(f"  {imp['dataset']}-{imp['model']}:")
                improvement_output.append(f"    BLEU: {imp['bleu_improve']:+6.2f}%, METEOR: {imp['meteor_improve']:+6.2f}%, PPL: {imp['ppl_improve']:+6.2f}%")
                improvement_output.append(f"    Rep-W: {imp['rep_w_improve']:+6.2f}%, Rep-R: {imp['rep_r_improve']:+6.2f}%, Rep-All: {imp['rep_all_improve']:+6.2f}%")
        
        # 总结最佳方法
        improvement_output.append("\n" + "="*80)
        improvement_output.append("SUMMARY: Best Methods by Metric")
        improvement_output.append("="*80)
        
        best_methods = {}
        for metric in ['bleu', 'meteor', 'ppl', 'rep_w', 'rep_r', 'rep_n', 'rep_all']:
            best_method = None
            best_score = float('-inf')
            
            for method, imp_list in method_improvements.items():
                avg_score = np.mean([imp[f'{metric}_improve'] for imp in imp_list])
                if avg_score > best_score:
                    best_score = avg_score
                    best_method = method
            
            best_methods[metric] = (best_method, best_score)
        
        improvement_output.append(f"Best BLEU improvement:        {best_methods['bleu'][0]} ({best_methods['bleu'][1]:+.2f}%)")
        improvement_output.append(f"Best METEOR improvement:      {best_methods['meteor'][0]} ({best_methods['meteor'][1]:+.2f}%)")
        improvement_output.append(f"Best PPL reduction:           {best_methods['ppl'][0]} ({best_methods['ppl'][1]:+.2f}%)")
        improvement_output.append(f"Best Rep-W reduction:         {best_methods['rep_w'][0]} ({best_methods['rep_w'][1]:+.2f}%)")
        improvement_output.append(f"Best Rep-R reduction:         {best_methods['rep_r'][0]} ({best_methods['rep_r'][1]:+.2f}%)")
        improvement_output.append(f"Best Rep-N-Avg reduction:     {best_methods['rep_n'][0]} ({best_methods['rep_n'][1]:+.2f}%)")
        improvement_output.append(f"Best Rep-All-Avg reduction:   {best_methods['rep_all'][0]} ({best_methods['rep_all'][1]:+.2f}%)")
        
        # 添加时间戳
        from datetime import datetime
        improvement_output.append(f"\nGenerated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    else:
        improvement_output.append(f"\nWarning: Baseline method '{baseline_method}' not found in data!")
    
    improvement_output.append("\n" + "="*80)
    
    # 写入文件
    with open('improvements.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(improvement_output))
    
    # 同时在控制台打印
    print("\n" + "\n".join(improvement_output))
    print(f"\nImprovement analysis saved to: improvements.txt")
    
    return df


def create_complementarity_analysis(df):
    """分析复合方法如何结合单体方法的优点 - 包含PPL"""
    
    # 定义方法组
    method_groups = {
        'SAE Series': {
            'single': ['SAE'],
            'composite': ['SAE+Penalty'],
            'baseline': ['Penalty']
        },
        'Neuron Series': {
            'single': ['Neuron'],
            'composite': ['Neuron+Penalty'],
            'baseline': ['Penalty']
        }
    }
    
    # 映射 decode_mode 到 display name
    decode_mode_to_display = {
        'sae': 'SAE',
        'neuron': 'Neuron',
        'penalty': 'Penalty',
        'sae_penalty': 'SAE+Penalty',
        'neuron_penalty': 'Neuron+Penalty',
        'topp': 'Baseline'
    }
    df['Display_Name'] = df['Decode_Mode'].map(decode_mode_to_display)
    
    fig = plt.figure(figsize=(24, 12))
    fig.suptitle('Complementarity Analysis: How Composite Methods Combine Single Method Advantages (with PPL)', 
                 fontsize=18, fontweight='bold')
    
    colors = {
        'SAE': '#1E90FF',
        'Neuron': '#FF6B35',
        'Penalty': '#32CD32',
        'SAE+Penalty': '#9370DB',
        'Neuron+Penalty': '#DC143C',
        'Baseline': '#808080'
    }
    
    # 分析每个系列
    for idx, (series_name, methods) in enumerate(method_groups.items()):
        # 提取相关方法的数据
        relevant_methods = methods['single'] + methods['composite'] + methods['baseline']
        series_df = df[df['Display_Name'].isin(relevant_methods)]
        
        if series_df.empty:
            continue
        
        # 计算平均指标 - 包含PPL处理
        metrics_summary = series_df.groupby('Display_Name').agg({
            'BLEU': 'mean',
            'METEOR': 'mean',
            'PERPLEXITY': lambda x: x.replace([np.inf, -np.inf], np.nan).dropna().mean(),
            'REP_W': 'mean',
            'REP_R': 'mean',
            'REP_N_1': 'mean',
            'REP_N_2': 'mean',
            'REP_N_3': 'mean',
            'REP_N_4': 'mean',
            'REP_N_5': 'mean'
        })
        
        # 计算综合重复度指标
        rep_cols = ['REP_W', 'REP_R', 'REP_N_1', 'REP_N_2', 'REP_N_3', 'REP_N_4', 'REP_N_5']
        metrics_summary['REP_AVG'] = metrics_summary[rep_cols].mean(axis=1)
        
        # 子图1: 质量指标对比 (BLEU)
        ax1 = plt.subplot(2, 4, idx * 4 + 1)
        methods_order = methods['baseline'] + methods['single'] + methods['composite']
        methods_order = [m for m in methods_order if m in metrics_summary.index]
        
        bleu_values = [metrics_summary.loc[m, 'BLEU'] for m in methods_order]
        bars = ax1.bar(range(len(methods_order)), bleu_values, 
                      color=[colors[m] for m in methods_order],
                      alpha=0.85, edgecolor='black', linewidth=1.5)
        
        # 添加数值标签
        for i, (bar, val) in enumerate(zip(bars, bleu_values)):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.4f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax1.set_ylabel('BLEU Score', fontsize=12, fontweight='bold')
        ax1.set_title(f'{series_name}: Quality (BLEU)\n(Higher is Better)', 
                     fontsize=12, fontweight='bold')
        ax1.set_xticks(range(len(methods_order)))
        ax1.set_xticklabels(methods_order, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.set_ylim(0, max(bleu_values) * 1.15)
        
        # 子图2: PPL对比
        ax2 = plt.subplot(2, 4, idx * 4 + 2)
        ppl_values = [metrics_summary.loc[m, 'PERPLEXITY'] for m in methods_order]
        bars = ax2.bar(range(len(methods_order)), ppl_values,
                      color=[colors[m] for m in methods_order],
                      alpha=0.85, edgecolor='black', linewidth=1.5)
        
        for i, (bar, val) in enumerate(zip(bars, ppl_values)):
            height = bar.get_height()
            if not math.isnan(val):
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.2f}',
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax2.set_ylabel('PPL Score', fontsize=12, fontweight='bold')
        ax2.set_title(f'{series_name}: Perplexity\n(Lower is Better)', 
                     fontsize=12, fontweight='bold')
        ax2.set_xticks(range(len(methods_order)))
        ax2.set_xticklabels(methods_order, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3, axis='y')
        valid_ppl = [v for v in ppl_values if not math.isnan(v)]
        if valid_ppl:
            ax2.set_ylim(0, max(valid_ppl) * 1.15)
        
        # 子图3: 重复度指标对比 (平均)
        ax3 = plt.subplot(2, 4, idx * 4 + 3)
        rep_values = [metrics_summary.loc[m, 'REP_AVG'] for m in methods_order]
        bars = ax3.bar(range(len(methods_order)), rep_values,
                      color=[colors[m] for m in methods_order],
                      alpha=0.85, edgecolor='black', linewidth=1.5)
        
        for i, (bar, val) in enumerate(zip(bars, rep_values)):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.4f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax3.set_ylabel('Average Repetition Score', fontsize=12, fontweight='bold')
        ax3.set_title(f'{series_name}: Repetition (Avg)\n(Lower is Better)', 
                     fontsize=12, fontweight='bold')
        ax3.set_xticks(range(len(methods_order)))
        ax3.set_xticklabels(methods_order, rotation=45, ha='right')
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.set_ylim(0, max(rep_values) * 1.15)
        
        # 子图4: 互补性雷达图（包含PPL）
        ax4 = plt.subplot(2, 4, idx * 4 + 4, projection='polar')
        
        # 准备雷达图数据（归一化）- 包含PPL
        categories = ['BLEU', 'METEOR', 'PPL\n(inv)', 'Rep-W\n(inv)', 'Rep-R\n(inv)', 'Rep-N\n(inv)']
        
        # 计算反向值（越低越好 -> 越高越好）
        max_ppl = metrics_summary['PERPLEXITY'].replace([np.inf, -np.inf], np.nan).dropna().max()
        max_rep_w = metrics_summary['REP_W'].max()
        max_rep_r = metrics_summary['REP_R'].max()
        max_rep_n = metrics_summary[['REP_N_1', 'REP_N_2', 'REP_N_3', 'REP_N_4', 'REP_N_5']].mean(axis=1).max()
        
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]
        
        for method in methods_order:
            ppl_val = metrics_summary.loc[method, 'PERPLEXITY']
            values = [
                metrics_summary.loc[method, 'BLEU'] / metrics_summary['BLEU'].max(),
                metrics_summary.loc[method, 'METEOR'] / metrics_summary['METEOR'].max(),
                1 - (ppl_val / max_ppl) if not math.isnan(ppl_val) and max_ppl > 0 else 0,
                1 - (metrics_summary.loc[method, 'REP_W'] / max_rep_w) if max_rep_w > 0 else 0,
                1 - (metrics_summary.loc[method, 'REP_R'] / max_rep_r) if max_rep_r > 0 else 0,
                1 - (metrics_summary.loc[method, ['REP_N_1', 'REP_N_2', 'REP_N_3', 'REP_N_4', 'REP_N_5']].mean() / max_rep_n) if max_rep_n > 0 else 0
            ]
            values += values[:1]
            
            ax4.plot(angles, values, 'o-', linewidth=2.5, label=method, 
                    color=colors[method], markersize=8)
            ax4.fill(angles, values, alpha=0.15, color=colors[method])
        
        # 设置类别标签
        ax4.set_xticks(angles[:-1])
        ax4.set_xticklabels(categories, fontsize=10, fontweight='bold')
        
        # 移除径向坐标轴标签和刻度
        ax4.set_yticks([])
        ax4.set_yticklabels([])
        
        # 设置范围
        ax4.set_ylim(0, 1)
        
        # 标题
        ax4.set_title(f'{series_name}: Complementarity\n(Normalized, Outer=Better)', 
                     fontsize=12, fontweight='bold', pad=20)
        
        # 图例
        ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9, framealpha=0.9)
        
        # 保留网格线但使其更淡
        ax4.grid(True, alpha=0.25, linestyle='--', linewidth=0.8)
        
        # 设置背景颜色为白色
        ax4.set_facecolor('white')
    
    plt.tight_layout()
    plt.savefig('complementarity_analysis_with_ppl.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_improvement_analysis(df):
    """量化复合方法相对于单体方法的改进"""
    
    decode_mode_to_display = {
        'sae': 'SAE',
        'neuron': 'Neuron',
        'penalty': 'Penalty',
        'sae_penalty': 'SAE+Penalty',
        'neuron_penalty': 'Neuron+Penalty',
        'topp': 'Baseline'
    }
    df['Display_Name'] = df['Decode_Mode'].map(decode_mode_to_display)
    
    # 定义对比组
    comparisons = [
        {
            'composite': 'SAE+Penalty',
            'components': ['SAE', 'Penalty'],
            'name': 'SAE+Penalty vs Components'
        },
        {
            'composite': 'Neuron+Penalty',
            'components': ['Neuron', 'Penalty'],
            'name': 'Neuron+Penalty vs Components'
        }
    ]
    
    fig, axes = plt.subplots(len(comparisons), 2, figsize=(16, 6 * len(comparisons)))
    if len(comparisons) == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle('Improvement Analysis: Composite vs Single Methods', 
                 fontsize=18, fontweight='bold')
    
    for idx, comparison in enumerate(comparisons):
        composite = comparison['composite']
        components = comparison['components']
        
        # 提取数据
        methods = [composite] + components
        method_df = df[df['Display_Name'].isin(methods)]
        
        if method_df.empty:
            continue
        
        metrics_avg = method_df.groupby('Display_Name').agg({
            'BLEU': 'mean',
            'METEOR': 'mean',
            'REP_W': 'mean',
            'REP_R': 'mean',
            'REP_N_1': 'mean',
            'REP_N_2': 'mean',
            'REP_N_3': 'mean',
            'REP_N_4': 'mean',
            'REP_N_5': 'mean'
        })
        
        rep_cols = ['REP_W', 'REP_R', 'REP_N_1', 'REP_N_2', 'REP_N_3', 'REP_N_4', 'REP_N_5']
        metrics_avg['REP_AVG'] = metrics_avg[rep_cols].mean(axis=1)
        
        # 子图1: 相对改进百分比 (BLEU)
        ax1 = axes[idx, 0]
        improvements_bleu = []
        labels = []
        
        for component in components:
            if component in metrics_avg.index and composite in metrics_avg.index:
                improve = ((metrics_avg.loc[composite, 'BLEU'] - 
                          metrics_avg.loc[component, 'BLEU']) / 
                         metrics_avg.loc[component, 'BLEU'] * 100)
                improvements_bleu.append(improve)
                labels.append(f'{composite}\nvs\n{component}')
        
        colors_improve = ['#2ECC71' if x > 0 else '#E74C3C' for x in improvements_bleu]
        bars = ax1.barh(range(len(labels)), improvements_bleu, color=colors_improve,
                       alpha=0.85, edgecolor='black', linewidth=1.5)
        
        for i, (bar, val) in enumerate(zip(bars, improvements_bleu)):
            width = bar.get_width()
            ax1.text(width, bar.get_y() + bar.get_height()/2.,
                    f'{val:+.2f}%',
                    ha='left' if val > 0 else 'right', va='center', 
                    fontsize=11, fontweight='bold')
        
        ax1.set_xlabel('BLEU Improvement (%)', fontsize=12, fontweight='bold')
        ax1.set_title(f'{comparison["name"]}: BLEU Quality Improvement', 
                     fontsize=12, fontweight='bold')
        ax1.set_yticks(range(len(labels)))
        ax1.set_yticklabels(labels)
        ax1.axvline(x=0, color='black', linestyle='--', linewidth=1.5)
        ax1.grid(True, alpha=0.3, axis='x')
        
        # 子图2: 相对改进百分比 (REP_AVG - 注意是降低)
        ax2 = axes[idx, 1]
        improvements_rep = []
        
        for component in components:
            if component in metrics_avg.index and composite in metrics_avg.index:
                # 重复度降低是好的，所以计算 (component - composite) / component
                improve = ((metrics_avg.loc[component, 'REP_AVG'] - 
                          metrics_avg.loc[composite, 'REP_AVG']) / 
                         metrics_avg.loc[component, 'REP_AVG'] * 100)
                improvements_rep.append(improve)
        
        colors_improve = ['#2ECC71' if x > 0 else '#E74C3C' for x in improvements_rep]
        bars = ax2.barh(range(len(labels)), improvements_rep, color=colors_improve,
                       alpha=0.85, edgecolor='black', linewidth=1.5)
        
        for i, (bar, val) in enumerate(zip(bars, improvements_rep)):
            width = bar.get_width()
            ax2.text(width, bar.get_y() + bar.get_height()/2.,
                    f'{val:+.2f}%',
                    ha='left' if val > 0 else 'right', va='center', 
                    fontsize=11, fontweight='bold')
        
        ax2.set_xlabel('Repetition Reduction (%)', fontsize=12, fontweight='bold')
        ax2.set_title(f'{comparison["name"]}: Repetition Reduction', 
                     fontsize=12, fontweight='bold')
        ax2.set_yticks(range(len(labels)))
        ax2.set_yticklabels(labels)
        ax2.axvline(x=0, color='black', linestyle='--', linewidth=1.5)
        ax2.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig('improvement_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    result_dir = "./test_results_new/"
    
    print("Reading experimental results...")
    data = read_all_summaries(result_dir)
    
    print(f"Successfully read {len(data)} experimental results")
    
    # Print found decode modes for debugging
    decode_modes = sorted(list(set([d['decode_mode'] for d in data])))
    print(f"Found decode modes: {decode_modes}")
    
    # Create comparison plots with all metrics including PPL
    print("Creating complete comparison plots with all metrics including PPL...")
    create_comparison_plots(data)
    
    # Create quality metrics detailed plot
    print("Creating quality metrics detailed comparison (BLEU, METEOR, PPL)...")
    create_quality_metrics_detailed_plot(data)
    
    # Create Rep-N detailed plot
    print("Creating Rep-N detailed comparison...")
    create_repn_detailed_plot(data)
    
    # Create all repetition metrics plot
    print("Creating all repetition metrics comparison...")
    create_all_repetition_metrics_plot(data)
    
    # Create detailed analysis
    print("Creating detailed analysis...")
    df = create_detailed_analysis(data)
    
    # Create comprehensive method ranking analysis
    print("Creating comprehensive method ranking analysis with PPL...")
    create_method_ranking_analysis(df)

    # Create complementarity analysis
    print("Creating complementarity analysis with PPL...")
    create_complementarity_analysis(df)
    
    print("Creating improvement analysis...")
    create_improvement_analysis(df)
    
    print("\nAnalysis completed! Generated files:")
    print("- model_comparison_complete_all_metrics_with_ppl.png: Complete comparison including PPL")
    print("- quality_metrics_detailed_comparison.png: Detailed quality metrics (BLEU, METEOR, PPL)")
    print("- repn_detailed_comparison.png: Line plots showing Rep-N-1 to Rep-N-5 trends")
    print("- all_repetition_metrics_comparison.png: Bar chart comparing all repetition metrics")
    print("- comprehensive_analysis_all_metrics_with_ppl.png: 9-panel comprehensive analysis with PPL")
    print("- detailed_results_complete_with_ppl.csv: Complete detailed results table with PPL")
    print("- complementarity_analysis_with_ppl.png: 复合方法分析（包含PPL）")
    print("- improvement_analysis.png: 复合方法相对于单体方法的改进百分比")

if __name__ == "__main__":
    main()
