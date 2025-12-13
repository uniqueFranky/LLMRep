import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import defaultdict

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
    """Get all unique decode modes and assign colors"""
    all_decode_modes = sorted(list(set([d['decode_mode'] for d in data])))
    
    # Extended color palette
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    
    # If we need more colors, generate them
    while len(colors) < len(all_decode_modes):
        colors.extend(['#bcbd22', '#17becf', '#aec7e8', '#ffbb78', '#98df8a'])
    
    decode_mode_names = {
        'penalty': 'Penalty',
        'neuron': 'Neuron',
        'sae_penalty': 'SAE Penalty',
        'neuron_penalty': 'Neuron Penalty',
        'greedy': 'Greedy'
    }
    
    return all_decode_modes, colors[:len(all_decode_modes)], decode_mode_names

def create_comparison_plots(data):
    """Create comparison plots"""
    # Get all decode modes and colors
    all_decode_modes, colors, decode_mode_names = get_decode_modes_and_colors(data)
    
    # Organize data
    datasets = sorted(list(set([d['dataset'] for d in data])))
    models = sorted(list(set([d['model'] for d in data])))
    
    # Select metrics to display
    metrics_to_plot = ['bleu', 'meteor', 'rep_w', 'rep_n_2', 'rep_r']
    metric_names = {
        'bleu': 'BLEU',
        'meteor': 'METEOR',
        'rep_w': 'Rep-W',
        'rep_n_2': 'Rep-N-2',
        'rep_r': 'Rep-R'
    }
    
    # Create plots for datasets
    fig, axes = plt.subplots(len(datasets), len(models), figsize=(6*len(models), 6*len(datasets)))
    if len(datasets) == 1:
        axes = axes.reshape(1, -1)
    if len(models) == 1:
        axes = axes.reshape(-1, 1)
    
    fig.suptitle('Model Performance Comparison Across Datasets and Decode Modes', fontsize=16, fontweight='bold')
    
    for dataset_idx, dataset in enumerate(datasets):
        for model_idx, model in enumerate(models):
            ax = axes[dataset_idx, model_idx]
            
            # Filter data for current dataset and model
            filtered_data = [d for d in data if d['dataset'] == dataset and d['model'] == model]
            
            # Organize data for plotting
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
                        value = mode_data[0]['avg_metrics'].get(metric, 0)
                        # Handle infinity values
                        if value == float('inf') or value != value:  # NaN check
                            value = 0
                        metrics_data[metric].append(value)
            
            if not decode_mode_labels:  # No data for this combination
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{dataset} Dataset - {model} Model', fontsize=14, fontweight='bold')
                continue
            
            # Create grouped bar chart
            x = np.arange(len(metrics_to_plot))
            width = 0.8 / len(decode_mode_labels)
            
            for i, decode_mode in enumerate(decode_mode_labels):
                values = [metrics_data[metric][i] for metric in metrics_to_plot]
                ax.bar(x + i * width - width * (len(decode_mode_labels) - 1) / 2, 
                      values, width, label=decode_mode, color=used_colors[i], alpha=0.8)
            
            ax.set_xlabel('Metrics', fontsize=12)
            ax.set_ylabel('Score', fontsize=12)
            ax.set_title(f'{dataset} Dataset - {model} Model', fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels([metric_names[m] for m in metrics_to_plot], rotation=45)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # Set y-axis limits for better visualization
            if metrics_data:
                max_val = max([max(metrics_data[m]) for m in metrics_to_plot if metrics_data[m]])
                ax.set_ylim(0, max_val * 1.1 + 0.1)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_detailed_analysis(data):
    """Create detailed analysis table"""
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
            if value == float('inf') or value != value:
                value = 0
            row[metric.upper()] = value
        
        df_list.append(row)
    
    df = pd.DataFrame(df_list)
    
    # Save detailed results
    df.to_csv('detailed_results.csv', index=False)
    
    # Print summary statistics
    print("=== Experiment Results Summary ===")
    print(f"Total experimental configurations processed: {len(df)}")
    print(f"Datasets: {sorted(df['Dataset'].unique())}")
    print(f"Models: {sorted(df['Model'].unique())}")
    print(f"Decode Modes: {sorted(df['Decode_Mode'].unique())}")
    
    print("\n=== Average Metrics ===")
    metrics = ['BLEU', 'METEOR', 'REP_W', 'REP_N_1', 'REP_N_2', 'REP_N_3', 'REP_N_4', 'REP_N_5', 'REP_R']
    for metric in metrics:
        if metric in df.columns:
            print(f"{metric}: {df[metric].mean():.4f}")
    
    return df

def create_heatmap_analysis(df):
    """Create heatmap analysis"""
    metrics = [col for col in df.columns if col.upper() in ['BLEU', 'METEOR', 'REP_W', 'REP_N_2', 'REP_R', 'PERPLEXITY']]
    
    if not metrics:
        print("No suitable metrics found for heatmap analysis")
        return
    
    # Limit to 6 metrics maximum
    metrics = metrics[:6]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Performance Heatmap Analysis Across Different Configurations', fontsize=16, fontweight='bold')
    
    for i, metric in enumerate(metrics):
        if i >= 6:
            break
        row = i // 3
        col = i % 3
        ax = axes[row, col]
        
        # Create pivot table
        try:
            pivot_data = df.pivot_table(
                values=metric, 
                index=['Dataset', 'Model'], 
                columns='Decode_Mode', 
                aggfunc='mean'
            )
            
            # Handle empty pivot table
            if pivot_data.empty or pivot_data.isna().all().all():
                ax.text(0.5, 0.5, f'No data for {metric}', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{metric.upper()}')
                continue
            
            # Create heatmap
            im = ax.imshow(pivot_data.values, cmap='viridis', aspect='auto')
            
            # Set labels
            ax.set_xticks(range(len(pivot_data.columns)))
            ax.set_xticklabels(pivot_data.columns, rotation=45, ha='right')
            ax.set_yticks(range(len(pivot_data.index)))
            ax.set_yticklabels([f"{idx[0]}-{idx[1]}" for idx in pivot_data.index])
            ax.set_title(f'{metric.upper()} Performance', fontweight='bold')
            
            # Add value annotations
            for i_row in range(len(pivot_data.index)):
                for j_col in range(len(pivot_data.columns)):
                    value = pivot_data.iloc[i_row, j_col]
                    if not pd.isna(value):
                        text_color = 'white' if value < pivot_data.values.mean() else 'black'
                        ax.text(j_col, i_row, f'{value:.3f}', ha='center', va='center', 
                               color=text_color, fontsize=8)
            
            # Add colorbar
            plt.colorbar(im, ax=ax, shrink=0.8)
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)[:30]}...', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{metric.upper()}')
    
    # Remove extra subplots
    for i in range(len(metrics), 6):
        row = i // 3
        col = i % 3
        fig.delaxes(axes[row, col])
    
    plt.tight_layout()
    plt.savefig('heatmap_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_repetition_analysis(df):
    """Create specific analysis for repetition metrics"""
    repetition_metrics = [col for col in df.columns if col.upper() in ['REP_W', 'REP_N_2', 'REP_N_4', 'REP_R']]
    
    if not repetition_metrics:
        print("No repetition metrics found")
        return
    
    # Get all decode modes and create color mapping
    all_decode_modes = sorted(df['Decode_Mode'].unique())
    colors = plt.cm.Set3(np.linspace(0, 1, len(all_decode_modes)))
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Repetition Metrics Analysis Across Models and Datasets', fontsize=16, fontweight='bold')
    
    for i, metric in enumerate(repetition_metrics[:4]):  # Max 4 metrics
        row = i // 2
        col = i % 2
        ax = axes[row, col]
        
        # Create grouped bar plot
        datasets = sorted(df['Dataset'].unique())
        models = sorted(df['Model'].unique())
        
        x = np.arange(len(datasets))
        width = 0.8 / (len(models) * len(all_decode_modes))
        
        bar_index = 0
        for j, model in enumerate(models):
            for k, decode_mode in enumerate(all_decode_modes):
                values = []
                for dataset in datasets:
                    subset = df[(df['Dataset'] == dataset) & 
                               (df['Model'] == model) & 
                               (df['Decode_Mode'] == decode_mode)]
                    if not subset.empty:
                        values.append(subset[metric].iloc[0])
                    else:
                        values.append(0)
                
                offset = bar_index * width - width * (len(models) * len(all_decode_modes) - 1) / 2
                label = f'{model}-{decode_mode}' if i == 0 else ""
                ax.bar(x + offset, values, width, 
                      label=label, color=colors[k], alpha=0.7 if j == 0 else 0.9)
                bar_index += 1
        
        ax.set_xlabel('Dataset', fontsize=12)
        ax.set_ylabel(f'{metric.upper()} Score', fontsize=12)
        ax.set_title(f'{metric.upper()} Comparison', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(datasets)
        if i == 0:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('repetition_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    result_dir = "./test_results_new/"
    
    print("Reading experimental results...")
    data = read_all_summaries(result_dir)
    
    print(f"Successfully read {len(data)} experimental results")
    
    # Print found decode modes for debugging
    decode_modes = sorted(list(set([d['decode_mode'] for d in data])))
    print(f"Found decode modes: {decode_modes}")
    
    # Create comparison plots
    print("Creating comparison plots...")
    create_comparison_plots(data)
    
    # Create detailed analysis
    print("Creating detailed analysis...")
    df = create_detailed_analysis(data)
    
    # Create heatmap analysis
    print("Creating heatmap analysis...")
    create_heatmap_analysis(df)
    
    # Create repetition-specific analysis
    print("Creating repetition analysis...")
    create_repetition_analysis(df)
    
    print("\nAnalysis completed! Generated files:")
    print("- model_comparison.png: Main comparison charts")
    print("- heatmap_analysis.png: Heatmap analysis")
    print("- repetition_analysis.png: Repetition metrics analysis")
    print("- detailed_results.csv: Detailed results table")

if __name__ == "__main__":
    main()
