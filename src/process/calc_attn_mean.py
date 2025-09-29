#!/usr/bin/env python3
"""
计算注意力概率矩阵的平均值
支持两种计算方式：
1. 时空平均：对前两维取平均：(seq_len, num_heads, seq_len, seq_len) -> (seq_len, seq_len)
2. 头平均：对头维度取平均：(seq_len, num_heads, seq_len, seq_len) -> (seq_len, seq_len, seq_len)
"""
import numpy as np
from pathlib import Path
import argparse
import os
import sys

# 添加项目根目录到 sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def parse_arguments():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='计算注意力概率矩阵的平均值')
    parser.add_argument('--hidden_states_dir', default='data/hidden_states',
                       help='hidden states文件夹路径')
    parser.add_argument('--output_dir', default='data/processed/A_mean',
                       help='输出文件夹路径')
    parser.add_argument('--dataset_ids', nargs='*', default='all',
                       help='指定要处理的数据集ID列表，支持格式：all, "1,2,3", 1 2 3')
    parser.add_argument('--sample_ids', nargs='*', default='all',
                       help='指定要处理的样本ID列表，支持格式：all, "1,2,3", 1 2 3')
    
    return parser.parse_args()

def load_attention_matrices(hidden_states_dir, dataset_id, sample_id, layer_idx, attn_type):
    """
    加载注意力概率矩阵
    
    Args:
        hidden_states_dir: hidden states文件夹路径
        dataset_id: 数据集ID
        sample_id: 样本ID
        layer_idx: 层索引
        attn_type: 注意力类型 ('temporal' 或 'spatial')
        
    Returns:
        np.array: 注意力矩阵，形状为 (seq_len, num_heads, seq_len, seq_len)
    """
    sample_path = Path(hidden_states_dir) / f'dataset_{dataset_id}' / f'sample_{sample_id}.npz'
    
    if not sample_path.exists():
        print(f"Warning: {sample_path} not found")
        return None
    
    data = np.load(sample_path)
    attn_key = f'attn_{layer_idx}_{attn_type}/attn'
    
    if attn_key not in data:
        print(f"Warning: {attn_key} not found in {sample_path}")
        return None
    
    return data[attn_key]

def compute_mean_attention(attn_matrices):
    """
    计算平均注意力矩阵，对前两维取平均
    
    Args:
        attn_matrices: 注意力矩阵，形状为 (seq_len, num_heads, seq_len, seq_len)
        
    Returns:
        np.array: 平均注意力矩阵，形状为 (seq_len, seq_len)
    """
    # 对第二维（num_heads）取平均
    # attn_matrices: (seq_len, num_heads, seq_len, seq_len) -> (seq_len, seq_len, seq_len)
    head_avg = np.mean(attn_matrices, axis=1)
    
    # 对第一维（seq_len/T/patch维度）取平均
    # 从 (seq_len, seq_len, seq_len) 变为 (seq_len, seq_len)
    final_avg = np.mean(head_avg, axis=0)
    
    return final_avg

def compute_head_mean_attention(attn_matrices):
    """
    计算头平均注意力矩阵，仅对头维度取平均
    
    Args:
        attn_matrices: 注意力矩阵，形状为 (seq_len, num_heads, seq_len, seq_len)
        
    Returns:
        np.array: 头平均注意力矩阵，形状为 (seq_len, seq_len, seq_len)
    """
    # 对第二维（num_heads）取平均
    # attn_matrices: (seq_len, num_heads, seq_len, seq_len) -> (seq_len, seq_len, seq_len)
    head_avg = np.mean(attn_matrices, axis=1)
    
    return head_avg

def parse_ids(ids_arg):
    """
    解析ID参数，支持多种格式
    
    Args:
        ids_arg: ID参数，可以是列表或字符串
        
    Returns:
        list or None: 解析后的ID列表，如果是'all'则返回None
    """
    if ids_arg is None:
        return None
    
    # 如果是列表，直接返回
    if isinstance(ids_arg, list):
        # 检查是否包含逗号分隔的字符串
        result = []
        for item in ids_arg:
            if ',' in str(item):
                # 分割逗号分隔的字符串
                result.extend([id.strip() for id in str(item).split(',')])
            else:
                result.append(str(item))
        return result if result != ['all'] else None
    
    # 如果是字符串
    if isinstance(ids_arg, str):
        if ids_arg.lower() == 'all':
            return None
        if ',' in ids_arg:
            return [id.strip() for id in ids_arg.split(',')]
        return [ids_arg]
    
    return None

def main():
    args = parse_arguments()
    
    # 解析数据集和样本ID
    dataset_ids = parse_ids(args.dataset_ids)
    sample_ids = parse_ids(args.sample_ids)
    
    # 创建输出文件夹
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 处理每个数据集的样本
    hidden_states_path = Path(args.hidden_states_dir)
    
    # 获取要处理的数据集列表
    if dataset_ids is not None:
        dataset_dirs = [hidden_states_path / f'dataset_{dataset_id}' 
                       for dataset_id in dataset_ids
                       if (hidden_states_path / f'dataset_{dataset_id}').exists()]
    else:
        dataset_dirs = [d for d in hidden_states_path.iterdir() 
                       if d.is_dir() and d.name.startswith('dataset_')]
    
    for dataset_dir in dataset_dirs:
        dataset_id = dataset_dir.name.split('_')[1]
        print(f"\n处理数据集 {dataset_id}...")
        
        # 创建输出子文件夹
        dataset_output_dir = output_dir / f'dataset_{dataset_id}'
        dataset_output_dir.mkdir(exist_ok=True)
        
        # 获取要处理的样本列表
        if sample_ids is not None:
            sample_files = [dataset_dir / f'sample_{sample_id}.npz' 
                           for sample_id in sample_ids
                           if (dataset_dir / f'sample_{sample_id}.npz').exists()]
        else:
            sample_files = list(dataset_dir.glob('sample_*.npz'))
        
        # 注意力层数固定为6层
        num_layers = 6
        
        # 处理每个样本
        for sample_file in sample_files:
            sample_id = sample_file.stem.split('_')[1]
            print(f"  处理样本 {sample_id}...")
            
            sample_results = {}
            
            # 处理每一层的每种注意力类型
            for layer_idx in range(num_layers):
                for attn_type in ['temporal', 'spatial']:
                    # 加载注意力矩阵
                    attn_matrices = load_attention_matrices(
                        args.hidden_states_dir, dataset_id, sample_id, 
                        layer_idx, attn_type
                    )
                    
                    if attn_matrices is None:
                        continue
                    
                    # 计算时空平均值
                    mean_avg = compute_mean_attention(attn_matrices)
                    
                    # 计算头平均值
                    head_mean_avg = compute_head_mean_attention(attn_matrices)
                    
                    # 保存结果
                    mean_key = f'attn_{layer_idx}_{attn_type}_mean'
                    head_mean_key = f'attn_{layer_idx}_{attn_type}_head_mean'
                    
                    sample_results[mean_key] = mean_avg
                    sample_results[head_mean_key] = head_mean_avg
            
            # 保存样本的所有平均结果
            if sample_results:
                output_file = dataset_output_dir / f'sample_{sample_id}_mean.npz'
                np.savez_compressed(output_file, **sample_results)
                print(f"    保存到 {output_file}")
    
    print(f"\n完成！结果保存在 {output_dir}")

if __name__ == '__main__':
    main()
