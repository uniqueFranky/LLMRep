import pandas as pd
import sys
from typing import Optional

def custom_method_sort_key(method_value):
    """
    为method列定义自定义排序键
    baseline排在最前面，带+的排在最后面，其余的在中间按字典序排序
    """
    method_str = str(method_value).lower()
    
    if method_str == 'baseline':
        return (0, method_str)  # 最高优先级
    elif '+' in method_str:
        return (2, method_str)  # 最低优先级
    else:
        return (1, method_str)  # 中等优先级

def csv_to_latex_table(csv_file: str, 
                      output_file: Optional[str] = None,
                      caption: str = "实验结果对比",
                      label: str = "tab:results",
                      resize_to_textwidth: bool = True,
                      exclude_columns: list = None) -> str:
    """
    将CSV文件转换为LaTeX表格格式
    
    Args:
        csv_file: CSV文件路径
        output_file: 输出文件路径（可选，如果不指定则只返回字符串）
        caption: 表格标题
        label: 表格标签
        resize_to_textwidth: 是否调整表格宽度适应页面
        exclude_columns: 要排除的列名列表
    
    Returns:
        LaTeX表格代码字符串
    """
    
    try:
        # 读取CSV文件
        df = pd.read_csv(csv_file)
        
        # 修改列名：将decode_mode相关的列名改为method
        column_mapping = {}
        for col in df.columns:
            if 'decode_mode' in col.lower() or 'decodemode' in col.lower():
                column_mapping[col] = 'Method'
        
        if column_mapping:
            df = df.rename(columns=column_mapping)
        
        # 处理Method列的值：将topp替换为baseline，下划线替换为加号
        if 'Method' in df.columns:
            df['Method'] = df['Method'].astype(str).str.replace('topp', 'baseline').str.replace('_', '+')
        
        # 排序逻辑
        sort_columns = []
        sort_keys = []
        
        # 检查是否存在这些列并添加到排序列表
        if 'Dataset' in df.columns:
            sort_columns.append('Dataset')
            sort_keys.append(True)  # 升序
        elif 'dataset' in df.columns:
            df = df.rename(columns={'dataset': 'Dataset'})
            sort_columns.append('Dataset')
            sort_keys.append(True)
            
        if 'Model' in df.columns:
            sort_columns.append('Model')
            sort_keys.append(True)  # 升序
        elif 'model' in df.columns:
            df = df.rename(columns={'model': 'Model'})
            sort_columns.append('Model')
            sort_keys.append(True)
            
        # 如果存在Method列，需要特殊处理排序
        if 'Method' in df.columns:
            # 添加临时排序键列
            df['_method_sort_key'] = df['Method'].apply(custom_method_sort_key)
            sort_columns.append('_method_sort_key')
            sort_keys.append(True)
        
        # 执行排序
        if sort_columns:
            df = df.sort_values(by=sort_columns, ascending=sort_keys)
            
        # 删除临时排序键列
        if '_method_sort_key' in df.columns:
            df = df.drop(columns=['_method_sort_key'])
        
        # 排除指定的列
        if exclude_columns:
            df = df.drop(columns=[col for col in exclude_columns if col in df.columns])
        
        # 重置索引
        df = df.reset_index(drop=True)
        
        # 获取列数
        num_cols = len(df.columns)
        
        # 生成列对齐字符串（第一列左对齐，其余居中对齐）
        col_alignment = 'l' + 'c' * (num_cols - 1)
        
        # 开始构建LaTeX代码
        latex_code = []
        latex_code.append("\\begin{table*}[!htbp]")
        latex_code.append("\\centering")
        latex_code.append(f"\\caption{{{caption}}}")
        latex_code.append(f"\\label{{{label}}}")
        latex_code.append("% \\renewcommand{\\arraystretch}{1.2}")
        latex_code.append("\\setcellgapes{5pt}\\makegapedcells")
        
        if resize_to_textwidth:
            latex_code.append("\\resizebox{\\textwidth}{!}{")
        
        latex_code.append(f"\\begin{{tabular}}{{{col_alignment}}}")
        latex_code.append("\\toprule")
        
        # 添加表头
        headers = [f"\\textbf{{{col}}}" for col in df.columns]
        latex_code.append(" & ".join(headers) + " \\\\ ")
        latex_code.append("\\midrule")
        
        # 添加数据行
        for _, row in df.iterrows():
            # 处理数值格式化
            formatted_row = []
            for i, (col_name, value) in enumerate(zip(df.columns, row)):
                if pd.isna(value):
                    formatted_row.append("-")
                elif isinstance(value, (int, float)):
                    # 特殊处理：BLEU和PPL为0时显示为"-"
                    if (col_name.upper() in ['BLEU', 'PERPLEXITY', 'PPL', 'METEOR'] and 
                        abs(value) < 1e-10):  # 使用很小的阈值来判断是否为0
                        formatted_row.append("-")
                    else:
                        # 对于其他数值，保留适当的小数位数
                        if isinstance(value, float):
                            if abs(value) < 0.001 and value != 0:
                                formatted_row.append(f"{value:.2e}")  # 科学计数法
                            elif abs(value) < 1:
                                formatted_row.append(f"{value:.4f}")  # 小数保留4位
                            else:
                                formatted_row.append(f"{value:.3f}")  # 小数保留3位
                        else:
                            formatted_row.append(str(value))
                else:
                    # 处理字符串值
                    str_value = str(value)
                    
                    # 对于字符串，如果是第一列（通常是模型名），加粗显示
                    if i == 0:
                        formatted_row.append(f"\\textbf{{{str_value}}}")
                    else:
                        formatted_row.append(str_value)
            
            latex_code.append(" & ".join(formatted_row) + " \\\\")
        
        latex_code.append("\\bottomrule")
        latex_code.append("\\end{tabular}")
        
        if resize_to_textwidth:
            latex_code.append("}")
        
        latex_code.append("\\end{table*}")
        
        # 合并所有行
        result = "\n".join(latex_code)
        
        # 如果指定了输出文件，写入文件
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(result)
            print(f"LaTeX表格已保存到: {output_file}")
        
        return result
        
    except FileNotFoundError:
        print(f"错误: 找不到文件 {csv_file}")
        return ""
    except Exception as e:
        print(f"处理文件时出错: {e}")
        return ""

def format_metric_names(latex_code: str) -> str:
    """
    格式化指标名称，使其更符合学术论文标准
    """
    replacements = {
        'BLEU': 'BLEU',
        'METEOR': 'METEOR', 
        'PERPLEXITY': 'PPL',
        'REP_W': 'Rep-W',
        'REP_N_1': 'Rep-1',
        'REP_N_2': 'Rep-2', 
        'REP_N_3': 'Rep-3',
        'REP_N_4': 'Rep-4',
        'REP_N_5': 'Rep-5',
        'REP_R': 'Rep-R',
        'Method': 'Method',
        'Dataset': 'Dataset',
        'Model': 'Model'
    }
    
    result = latex_code
    for old, new in replacements.items():
        result = result.replace(f"\\textbf{{{old}}}", f"\\textbf{{{new}}}")
    
    return result

def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("使用方法: python csv_to_latex.py <csv_file> [output_file]")
        print("示例: python csv_to_latex.py data.csv table.tex")
        return
    
    csv_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    # 生成LaTeX表格，排除Num_Samples列
    latex_result = csv_to_latex_table(
        csv_file=csv_file,
        output_file=output_file,
        caption="模型性能对比实验结果",
        label="tab:model_comparison",
        exclude_columns=['Num_Samples']  # 排除Num_Samples列
    )
    
    # 格式化指标名称
    latex_result = format_metric_names(latex_result)
    
    # 如果没有指定输出文件，直接打印结果
    if not output_file:
        print("\n生成的LaTeX表格代码:")
        print("=" * 50)
        print(latex_result)
    
    # 重新写入格式化后的结果
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(latex_result)
        print(f"格式化后的LaTeX表格已保存到: {output_file}")

if __name__ == "__main__":
    main()
