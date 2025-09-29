#!/usr/bin/env python3
"""
空间注意力矩阵重塑计算模块
提供矩阵重塑、数据加载和保存功能
"""

import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse

def build_arg_parser():
    parser = argparse.ArgumentParser(description='计算空间注意力矩阵重塑数据')
    parser.add_argument('--data_dir', default='data/processed/A_mean',
                        help='处理过的数据文件夹路径')
    parser.add_argument('--output_dir', default='data/processed/As_reshaped',
                        help='重塑数据输出文件夹路径')
    parser.add_argument('--dataset_ids', nargs='*', default=['all'],
                        help='指定要处理的数据集ID列表，支持格式：all, "1,2,3", 1 2 3')
    parser.add_argument('--sample_ids', nargs='*', default=['all'],
                        help='指定要处理的样本ID列表，支持格式：all, "1,2,3", 1 2 3')
    parser.add_argument('--data_types', nargs='+', default=['both'],
                        choices=['mean', 'head_mean', 'both'],
                        help='指定要处理的数据类型：mean(时空平均), head_mean(头平均), both(两种都画)')
    parser.add_argument('--process_all_layers', action='store_true', default=True,
                        help='是否处理所有层')
    parser.add_argument('--specific_layers', nargs='+', type=int, default=[0, 2, 4, 5],
                        help='如果不处理所有层，指定要处理的层')
    return parser

def reshape_attention_matrix_by_rows(attn_matrix):
    """
    按行重塑注意力矩阵
    将(100,100)矩阵的每一行重塑为(10,10)，然后按(10,10)网格排列
    
    Args:
        attn_matrix: 形状为(100,100)的注意力矩阵
        
    Returns:
        numpy.ndarray: 重新排列后的(100,100)矩阵
    """
    if attn_matrix.shape != (100, 100):
        raise ValueError(f"输入矩阵形状应为(100,100)，实际为{attn_matrix.shape}")
    
    # 初始化输出矩阵
    reshaped_matrix = np.zeros((100, 100))
    
    # 对每一行进行处理
    for row_idx in range(100):
        # 取出第row_idx行，形状为(100,)
        row_data = attn_matrix[row_idx, :]
        
        # 重塑为(10,10)小块
        row_block = row_data.reshape(10, 10)
        
        # 计算在10x10网格中的位置
        grid_row = row_idx // 10
        grid_col = row_idx % 10
        
        # 放置到输出矩阵的对应位置
        start_row = grid_row * 10
        end_row = start_row + 10
        start_col = grid_col * 10
        end_col = start_col + 10
        
        reshaped_matrix[start_row:end_row, start_col:end_col] = row_block
    
    return reshaped_matrix

def reshape_attention_matrix_by_cols(attn_matrix):
    """
    按列重塑注意力矩阵
    将(100,100)矩阵的每一列重塑为(10,10)，然后按(10,10)网格排列
    
    Args:
        attn_matrix: 形状为(100,100)的注意力矩阵
        
    Returns:
        numpy.ndarray: 重新排列后的(100,100)矩阵
    """
    if attn_matrix.shape != (100, 100):
        raise ValueError(f"输入矩阵形状应为(100,100)，实际为{attn_matrix.shape}")
    
    # 初始化输出矩阵
    reshaped_matrix = np.zeros((100, 100))
    
    # 对每一列进行处理
    for col_idx in range(100):
        # 取出第col_idx列，形状为(100,)
        col_data = attn_matrix[:, col_idx]
        
        # 重塑为(10,10)小块
        col_block = col_data.reshape(10, 10)
        
        # 计算在10x10网格中的位置
        grid_row = col_idx // 10
        grid_col = col_idx % 10
        
        # 放置到输出矩阵的对应位置
        start_row = grid_row * 10
        end_row = start_row + 10
        start_col = grid_col * 10
        end_col = start_col + 10
        
        reshaped_matrix[start_row:end_row, start_col:end_col] = col_block
    
    return reshaped_matrix

def save_sample_reshaped_data(sample_data_dict, save_path):
    """
    保存一个样本的所有层重塑后的数据
    
    Args:
        sample_data_dict: 样本数据字典 {data_type: {layer_idx: {timestep: (row_reshaped, col_reshaped)} 或 (row_reshaped, col_reshaped)}}
        save_path: 保存路径
    """
    save_dict = {}
    
    for data_type, layer_data in sample_data_dict.items():
        for layer_idx, layer_content in layer_data.items():
            if isinstance(layer_content, dict):
                # head_mean数据：有多个时间步
                for timestep, (row_reshaped, col_reshaped) in layer_content.items():
                    save_dict[f'{data_type}_layer_{layer_idx}_timestep_{timestep:02d}_row_reshaped'] = row_reshaped
                    save_dict[f'{data_type}_layer_{layer_idx}_timestep_{timestep:02d}_col_reshaped'] = col_reshaped
            else:
                # mean数据：只有一个矩阵对
                row_reshaped, col_reshaped = layer_content
                save_dict[f'{data_type}_layer_{layer_idx}_row_reshaped'] = row_reshaped
                save_dict[f'{data_type}_layer_{layer_idx}_col_reshaped'] = col_reshaped
    
    np.savez(save_path, **save_dict)

def load_spatial_attention_data(data_dir, data_types=['mean']):
    """
    加载空间注意力数据
    
    Args:
        data_dir: 数据文件夹路径
        data_types: 要加载的数据类型列表 ['mean', 'head_mean', 'both']
        
    Returns:
        dict: 数据字典 {data_type: {dataset_id: {sample_id: {layer_idx: spatial_attn_matrix}}}}
    """
    data_dir = Path(data_dir)
    all_data = {}
    
    # 处理数据类型参数
    if 'both' in data_types:
        types_to_load = ['mean', 'head_mean']
    else:
        types_to_load = [t for t in data_types if t in ['mean', 'head_mean']]
    
    # 初始化数据结构
    for data_type in types_to_load:
        all_data[data_type] = {}
    
    for dataset_dir in data_dir.iterdir():
        if not dataset_dir.is_dir() or not dataset_dir.name.startswith('dataset_'):
            continue
        
        dataset_id = dataset_dir.name.split('_')[1]
        for data_type in types_to_load:
            all_data[data_type][dataset_id] = {}
        
        for sample_file in dataset_dir.glob('*_mean.npz'):
            sample_id = sample_file.stem.replace('_mean', '').split('_')[1]
            data = dict(np.load(sample_file))
            
            # 为每种数据类型提取空间注意力矩阵
            for data_type in types_to_load:
                spatial_data = {}
                for key, matrix in data.items():
                    if 'spatial' in key:
                        if data_type == 'head_mean' and 'head_mean' in key:
                            # 解析层索引: attn_{layer_idx}_spatial_head_mean
                            parts = key.split('_')
                            if len(parts) >= 2:
                                layer_idx = int(parts[1])
                                # 头平均数据形状为 (seq_len, seq_len, seq_len)，保留所有时间步
                                spatial_data[layer_idx] = matrix
                        elif data_type == 'mean' and 'mean' in key and 'head_mean' not in key:
                            # 解析层索引: attn_{layer_idx}_spatial_mean
                            parts = key.split('_')
                            if len(parts) >= 2:
                                layer_idx = int(parts[1])
                                spatial_data[layer_idx] = matrix
                
                if spatial_data:
                    all_data[data_type][dataset_id][sample_id] = spatial_data
    
    return all_data

def compute_and_save_reshaped_data(data_dir, output_dir, dataset_ids=None, sample_ids=None, 
                                 data_types=['mean'], layers_to_process=None):
    """
    计算并保存重塑后的数据
    
    Args:
        data_dir: 输入数据目录
        output_dir: 输出数据目录
        dataset_ids: 要处理的数据集ID列表，None表示处理所有
        sample_ids: 要处理的样本ID列表，None表示处理所有
        data_types: 要处理的数据类型列表
        layers_to_process: 要处理的层列表，None表示处理所有层
        
    Returns:
        dict: 计算结果统计信息
    """
    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载数据
    print("加载空间注意力数据...")
    all_data = load_spatial_attention_data(data_dir, data_types)
    
    if not all_data or not any(all_data.values()):
        print("没有找到任何空间注意力数据文件！")
        return {}
    
    # 获取可用的数据集
    first_data_type = data_types[0] if isinstance(data_types, list) else data_types
    available_dataset_ids = sorted(all_data[first_data_type].keys()) if first_data_type in all_data else []
    if dataset_ids is not None:
        filtered_dataset_ids = [d for d in dataset_ids if d in available_dataset_ids]
    else:
        filtered_dataset_ids = available_dataset_ids
    
    print(f"可用数据集: {available_dataset_ids}")
    print(f"将处理数据集: {filtered_dataset_ids}")
    
    stats = {
        'processed_samples': 0,
        'processed_layers': 0,
        'processed_timesteps': 0,
        'errors': []
    }
    
    # 处理每个数据集
    for dataset_id in tqdm(filtered_dataset_ids, desc="处理数据集"):
        # 检查所有数据类型中是否都有此数据集
        if not all(dataset_id in all_data[dt] and all_data[dt][dataset_id] for dt in data_types):
            continue
        
        # 获取可用的样本
        available_sample_ids = sorted(all_data[first_data_type][dataset_id].keys())
        if sample_ids is not None:
            filtered_sample_ids = [s for s in sample_ids if s in available_sample_ids]
        else:
            filtered_sample_ids = available_sample_ids
        
        print(f"数据集 {dataset_id}: 可用样本 {available_sample_ids}, 将处理 {filtered_sample_ids}")
        
        # 创建数据集输出目录
        dataset_output_dir = output_dir / f'dataset_{dataset_id}'
        dataset_output_dir.mkdir(parents=True, exist_ok=True)
        
        for sample_id in tqdm(filtered_sample_ids, desc=f"处理数据集 {dataset_id} 的样本", leave=False):
            # 检查所有数据类型中是否都有此样本
            if not all(sample_id in all_data[dt][dataset_id] for dt in data_types):
                continue
            
            # 初始化样本重塑数据字典
            sample_reshaped_data = {}
            for data_type in data_types:
                sample_reshaped_data[data_type] = {}
            
            # 获取可用的层
            available_layers = sorted(all_data[first_data_type][dataset_id][sample_id].keys())
            if layers_to_process is not None:
                layers_to_process_filtered = [layer for layer in layers_to_process if layer in available_layers]
            else:
                layers_to_process_filtered = available_layers
            
            # 处理每一层的每种数据类型
            for layer_idx in layers_to_process_filtered:
                # 检查所有数据类型中是否都有此层
                if not all(layer_idx in all_data[dt][dataset_id][sample_id] for dt in data_types):
                    continue
                
                # 处理每种数据类型
                for data_type in data_types:
                    spatial_matrix = all_data[data_type][dataset_id][sample_id][layer_idx]
                    
                    # 根据数据类型处理不同的矩阵形状
                    if data_type == 'head_mean' and spatial_matrix.ndim == 3:
                        # 头平均数据：(seq_len, seq_len, seq_len) - 为每个时间步计算重塑
                        num_timesteps = spatial_matrix.shape[0]
                        
                        # 初始化层的时间步数据字典
                        sample_reshaped_data[data_type][layer_idx] = {}
                        
                        for timestep in range(num_timesteps):
                            # 提取当前时间步的2D矩阵
                            current_matrix = spatial_matrix[timestep]
                            
                            # 检查矩阵形状
                            if current_matrix.shape != (100, 100):
                                error_msg = f"数据集 {dataset_id}, 样本 {sample_id}, 层 {layer_idx}, 类型 {data_type}, 时间步 {timestep} 的空间注意力矩阵形状为 {current_matrix.shape}，跳过"
                                print(f"警告：{error_msg}")
                                stats['errors'].append(error_msg)
                                continue
                            
                            # 进行重塑
                            try:
                                row_reshaped = reshape_attention_matrix_by_rows(current_matrix)
                                col_reshaped = reshape_attention_matrix_by_cols(current_matrix)
                                
                                # 保存到样本数据字典
                                sample_reshaped_data[data_type][layer_idx][timestep] = (row_reshaped, col_reshaped)
                                stats['processed_timesteps'] += 1
                                
                            except Exception as e:
                                error_msg = f"处理数据集 {dataset_id}, 样本 {sample_id}, 层 {layer_idx}, 类型 {data_type}, 时间步 {timestep} 时出错: {e}"
                                print(f"错误：{error_msg}")
                                stats['errors'].append(error_msg)
                                continue
                    
                    else:
                        # 时空平均数据或其他2D数据：(seq_len, seq_len)
                        # 检查矩阵形状
                        if spatial_matrix.shape != (100, 100):
                            error_msg = f"数据集 {dataset_id}, 样本 {sample_id}, 层 {layer_idx}, 类型 {data_type} 的空间注意力矩阵形状为 {spatial_matrix.shape}，跳过"
                            print(f"警告：{error_msg}")
                            stats['errors'].append(error_msg)
                            continue
                        
                        # 进行重塑
                        try:
                            row_reshaped = reshape_attention_matrix_by_rows(spatial_matrix)
                            col_reshaped = reshape_attention_matrix_by_cols(spatial_matrix)
                            
                            # 保存到样本数据字典
                            sample_reshaped_data[data_type][layer_idx] = (row_reshaped, col_reshaped)
                            
                        except Exception as e:
                            error_msg = f"处理数据集 {dataset_id}, 样本 {sample_id}, 层 {layer_idx}, 类型 {data_type} 时出错: {e}"
                            print(f"错误：{error_msg}")
                            stats['errors'].append(error_msg)
                            continue
                
                stats['processed_layers'] += 1
            
            # 保存样本的所有重塑数据到一个文件
            if any(sample_reshaped_data.values()):
                sample_data_save_path = dataset_output_dir / f'sample_{sample_id}.npz'
                save_sample_reshaped_data(sample_reshaped_data, sample_data_save_path)
                stats['processed_samples'] += 1
    
    return stats

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
    parser = build_arg_parser()
    args = parser.parse_args()

    # 解析参数
    dataset_ids = parse_ids(args.dataset_ids)
    sample_ids = parse_ids(args.sample_ids)

    # 处理数据类型参数
    if 'both' in args.data_types:
        data_types_to_process = ['mean', 'head_mean']
    else:
        data_types_to_process = [t for t in args.data_types if t in ['mean', 'head_mean']]

    # 处理层参数
    layers_to_process = None if args.process_all_layers else args.specific_layers

    print("空间注意力矩阵重塑计算")
    print("=" * 50)
    print(f"数据目录: {args.data_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"数据类型: {data_types_to_process}")
    print(f"数据集ID: {dataset_ids if dataset_ids else 'all'}")
    print(f"样本ID: {sample_ids if sample_ids else 'all'}")
    print(f"处理所有层: {args.process_all_layers}")
    if not args.process_all_layers:
        print(f"指定层: {args.specific_layers}")
    print()

    # 执行计算
    stats = compute_and_save_reshaped_data(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        dataset_ids=dataset_ids,
        sample_ids=sample_ids,
        data_types=data_types_to_process,
        layers_to_process=layers_to_process
    )

    print(f"\n计算完成！")
    print(f"处理样本数: {stats['processed_samples']}")
    print(f"处理层数: {stats['processed_layers']}")
    print(f"处理时间步数: {stats['processed_timesteps']}")
    print(f"错误数: {len(stats['errors'])}")
    if stats['errors']:
        print("错误详情:")
        for error in stats['errors'][:5]:  # 只显示前5个错误
            print(f"  - {error}")
        if len(stats['errors']) > 5:
            print(f"  ... 还有 {len(stats['errors']) - 5} 个错误")

    print(f"重塑数据已保存到 {args.output_dir}")
    print("文件结构: dataset_X/sample_Y.npz (包含该样本的所有层数据)")
if __name__ == '__main__':
    main()
