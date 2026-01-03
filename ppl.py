#!/usr/bin/env python3
import json
import sys
import math

def calculate_average_ppl(file_path):
    """
    读取 JSONL 文件，计算平均 perplexity（跳过 inf 值）
    
    Args:
        file_path: JSONL 文件路径
    
    Returns:
        平均 perplexity 值
    """
    total_ppl = 0.0
    valid_count = 0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
                ppl = data.get('perplexity')
                
                # 跳过 None、inf 和 nan 值
                if ppl is not None and math.isfinite(ppl):
                    total_ppl += ppl
                    valid_count += 1
                    
            except json.JSONDecodeError as e:
                print(f"Warning: 第 {line_num} 行 JSON 解析错误: {e}", file=sys.stderr)
                continue
            except Exception as e:
                print(f"Warning: 第 {line_num} 行处理错误: {e}", file=sys.stderr)
                continue
    
    if valid_count == 0:
        raise ValueError("没有有效的 perplexity 值")
    
    return total_ppl / valid_count

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("用法: python script.py <jsonl_file_path>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    try:
        avg_ppl = calculate_average_ppl(file_path)
        print(avg_ppl)
    except FileNotFoundError:
        print(f"错误: 文件 '{file_path}' 不存在", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)
