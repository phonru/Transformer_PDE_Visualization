#!/usr/bin/env python3
"""
将(100,100)的空间注意力矩阵按行和列分别重塑为(10,10)小块，
然后按(10,10)的网格排列重新组成两个(100,100)的大矩阵
支持处理时空平均和头平均数据
支持单图和双图对比模式，通过配置文件控制绘图参数
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import argparse
import yaml

# 设置字体
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False


def build_arg_parser():
    parser = argparse.ArgumentParser(description='绘制重新排列的空间注意力矩阵热力图')
    # 路径配置
    parser.add_argument('--reshaped_data_dir', default='data/processed/As_reshaped',
                       help='重塑数据文件夹路径')
    parser.add_argument('--output_dir', default='output/plots_As_reshaped',
                       help='图片输出文件夹路径')
    parser.add_argument('--config_path', default='/home/pkq2/project/transformer_PDE_2/config/plot_spatial_attention_reshaped_config.yaml',
                       help='配置文件路径')

    # 绘图配置
    parser.add_argument('--plot_mode', default='comparison',
                       choices=['comparison', 'single_row', 'single_col'],
                       help='绘图模式：comparison(双图对比), single_row(行重塑单图), single_col(列重塑单图)')
    parser.add_argument('--dataset_ids', nargs='*', default=['2,3,4,5,6,8'],
                       help='指定要处理的数据集ID列表，支持格式：all, "1,2,3", 1 2 3')
    parser.add_argument('--sample_ids', nargs='*', default=['all'],
                       help='指定要处理的样本ID列表，支持格式：all, "1,2,3", 1 2 3')
    parser.add_argument('--data_types', nargs='+', default=['both'],
                       choices=['mean', 'head_mean', 'both'],
                       help='指定要处理的数据类型：mean(时空平均), head_mean(头平均), both(两种都画)')
    parser.add_argument('--timesteps', nargs='*', default=['all'],
                       help='指定要处理的时间步列表（仅对head_mean数据有效），支持格式：all, "1,2,3", 1 2 3')
    parser.add_argument('--process_all_layers', action='store_true', default=True,
                       help='是否处理所有层')
    parser.add_argument('--specific_layers', nargs='+', type=int, default=[0, 2, 5, 8, 11, 14],
                       help='如果不处理所有层，指定要处理的层')
    return parser

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

def load_config(config_path):
    """
    加载配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        dict: 配置字典
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def plot_single_reshaped_attention(matrix, title_prefix, save_path, config):
    """
    绘制单个重塑矩阵的热力图
    
    Args:
        matrix: 重塑后的矩阵
        title_prefix: 标题前缀
        save_path: 保存路径
        config: 绘图配置
    """
    fig, ax = plt.subplots(1, 1, figsize=config['figure']['figsize'])
    
    # 绘制热力图，不显示colorbar，消除像素边框
    sns.heatmap(matrix, ax=ax, cmap=config['colormap'],
                cbar=False, square=True, xticklabels=False, yticklabels=False)
    
    # 添加x轴和y轴标签（从配置文件读取）
    if 'axis_labels' in config:
        axis_config = config['axis_labels']
        # 将x轴标签放在上边界
        ax.xaxis.set_label_position('top')
        ax.set_xlabel(axis_config['x_label'], 
                     fontsize=axis_config['fontsize'], 
                     rotation=0,  # 不旋转标签
                     labelpad=axis_config.get('labelpad', 5))
        ax.set_ylabel(axis_config['y_label'], 
                     fontsize=axis_config['fontsize'], 
                     rotation=0,  # 不旋转标签
                     labelpad=axis_config.get('labelpad', 5))
    
    # 添加刻度，并将x轴刻度移动到上边界
    if config['grid_ticks']['enabled']:
        tick_positions = config['grid_ticks']['positions']
        ax.set_xticks(tick_positions)
        ax.set_yticks(tick_positions)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        # 将x轴刻度移动到上边界
        ax.xaxis.set_ticks_position('top')
    
    # 恢复黑色外边框
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(0.5)

    cbar_config = config['colorbar']
    cax = fig.add_axes([cbar_config['position']['left'], 
                       cbar_config['position']['bottom'], 
                       cbar_config['position']['width'], 
                       cbar_config['position']['height']])
    
    # 创建colorbar
    vmin, vmax = matrix.min(), matrix.max()
    sm = plt.cm.ScalarMappable(cmap=config['colormap'], 
                               norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cax)
    cbar.ax.tick_params(labelsize=cbar_config['fontsize'])
    
    # 设置colorbar刻度
    num_ticks = cbar_config['tick_count']
    ticks = np.linspace(vmin, vmax, num_ticks)
    cbar.set_ticks(ticks)
    
    # 计算公共指数并设置科学计数法格式
    max_abs_val = max(abs(vmin), abs(vmax))
    if max_abs_val == 0:
        common_exp = 0
    else:
        common_exp = int(np.floor(np.log10(max_abs_val)))
    
    if common_exp != 0:
        scaled_ticks = ticks / (10 ** common_exp)
        cbar.set_ticklabels([f'{tick:.1f}' for tick in scaled_ticks])
        cbar.ax.text(1.0, 1.02, f'×10$^{{{common_exp}}}$', 
                     transform=cbar.ax.transAxes, 
                     ha='center', va='bottom',
                     fontsize=cbar_config['fontsize'] * 0.8)
    else:
        cbar.set_ticklabels([f'{tick:.2f}' for tick in ticks])
    
    # 调整布局（不设置总标题）
    layout_config = config['layout']
    plt.subplots_adjust(left=layout_config['subplot_left'], 
                       right=layout_config['subplot_right'],
                       top=layout_config['subplot_top'], 
                       bottom=layout_config['subplot_bottom'])
    
    # SVG是矢量格式，不需要DPI参数，但保留以确保文本和线条大小一致性
    plt.savefig(save_path, bbox_inches='tight', format='svg')
    plt.close()

def plot_reshaped_attention_comparison(row_reshaped, col_reshaped,
                                     title_prefix, save_path, config):
    """
    绘制重塑后矩阵的对比图
    
    Args:
        row_reshaped: 按行重塑的矩阵
        col_reshaped: 按列重塑的矩阵
        title_prefix: 标题前缀
        save_path: 保存路径
        config: 绘图配置
    """
    fig, axes = plt.subplots(1, 2, figsize=config['figure']['figsize'])
    
    # 计算统一的颜色范围
    vmin = min(row_reshaped.min(), col_reshaped.min())
    vmax = max(row_reshaped.max(), col_reshaped.max())
    
    # 绘制热力图，但不显示各自的colorbar，消除像素边框
    sns.heatmap(row_reshaped, ax=axes[0], cmap=config['colormap'],
                cbar=False, square=True, xticklabels=False, yticklabels=False,
                vmin=vmin, vmax=vmax)
    axes[0].set_title(config['titles']['subplot_titles'][0], 
                      fontsize=config['titles']['subplot_title_fontsize'])
    
    sns.heatmap(col_reshaped, ax=axes[1], cmap=config['colormap'],
                cbar=False, square=True, xticklabels=False, yticklabels=False,
                vmin=vmin, vmax=vmax)
    axes[1].set_title(config['titles']['subplot_titles'][1], 
                      fontsize=config['titles']['subplot_title_fontsize'])
    
    # 恢复黑色外边框
    for ax in axes:
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(1.0)
    
    # 添加刻度，并将x轴刻度移动到上边界
    if config['grid_ticks']['enabled']:
        tick_positions = config['grid_ticks']['positions'][:-1]  # 去掉最后一个100
        for ax in axes:
            ax.set_xticks(tick_positions)
            ax.set_yticks(tick_positions)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            # 将x轴刻度移动到上边界
            ax.xaxis.set_ticks_position('top')
            ax.xaxis.set_label_position('top')
    
    # 调整子图间距
    layout_config = config['layout']
    plt.subplots_adjust(wspace=layout_config.get('wspace', 0.05))
    
    # 添加独立的colorbar
    cbar_config = config['colorbar']
    cax = fig.add_axes([cbar_config['position']['left'], 
                       cbar_config['position']['bottom'], 
                       cbar_config['position']['width'], 
                       cbar_config['position']['height']])
    
    # 创建统一的colorbar
    sm = plt.cm.ScalarMappable(cmap=config['colormap'], 
                               norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cax)
    cbar.ax.tick_params(labelsize=cbar_config['fontsize'])
    
    # 设置colorbar刻度
    num_ticks = cbar_config['tick_count']
    ticks = np.linspace(vmin, vmax, num_ticks)
    cbar.set_ticks(ticks)
    
    # 计算公共指数
    max_abs_val = max(abs(vmin), abs(vmax))
    if max_abs_val == 0:
        common_exp = 0
    else:
        common_exp = int(np.floor(np.log10(max_abs_val)))
    
    # 设置科学计数法格式，保留两位小数
    if common_exp != 0:
        # 将数值除以10^common_exp
        scaled_ticks = ticks / (10 ** common_exp)
        cbar.set_ticklabels([f'{tick:.2f}' for tick in scaled_ticks])
        
        # 在colorbar上方添加公共指数标签
        cbar.ax.text(0.5, 1.05, f'×10$^{{{common_exp}}}$', 
                     transform=cbar.ax.transAxes, 
                     ha='center', va='bottom',
                     fontsize=cbar_config['fontsize'] * 0.9)
    else:
        # 如果指数为0，直接显示原始数值
        cbar.set_ticklabels([f'{tick:.2f}' for tick in ticks])
    
    # 设置总标题
    plt.suptitle(f'{title_prefix}', 
                 fontsize=config['titles']['suptitle_fontsize'], 
                 y=config['titles']['suptitle_y'])
    
    # 手动调整布局而不使用tight_layout，避免与独立colorbar冲突
    plt.subplots_adjust(left=layout_config['subplot_left'], 
                       right=layout_config['subplot_right'], 
                       top=layout_config['subplot_top'], 
                       bottom=layout_config['subplot_bottom'],
                       wspace=layout_config.get('wspace', 0.05))
    
    # SVG是矢量格式，不需要DPI参数，但保留以确保文本和线条大小一致性
    plt.savefig(save_path, bbox_inches='tight', format='svg')
    plt.close()

def get_filename_from_config(config, plot_mode, data_type, timestep=None):
    """
    根据配置生成文件名
    
    Args:
        config: 配置字典
        plot_mode: 绘图模式
        data_type: 数据类型
        timestep: 时间步（可选）
        
    Returns:
        str: 文件名
    """
    filename_pattern = config['file_naming'][plot_mode]
    filename = filename_pattern.format(data_type=data_type)
    
    if timestep is not None:
        timestep_suffix = config['file_naming']['timestep_suffix']
        filename += timestep_suffix.format(timestep=timestep)
    
    return filename

def load_reshaped_data(data_dir, dataset_id, sample_id):
    """
    加载重塑后的数据
    
    Args:
        data_dir: 数据目录
        dataset_id: 数据集ID
        sample_id: 样本ID
        
    Returns:
        dict: 重塑数据字典
    """
    data_path = Path(data_dir) / f'dataset_{dataset_id}' / f'sample_{sample_id}.npz'
    
    if not data_path.exists():
        return None
    
    data = dict(np.load(data_path))
    
    # 重新组织数据结构
    organized_data = {}
    
    for key, matrix in data.items():
        # 解析键名: {data_type}_layer_{layer_idx}_timestep_{XX}_{reshape_type}_reshaped 或
        #          {data_type}_layer_{layer_idx}_{reshape_type}_reshaped
        parts = key.split('_')
        if len(parts) >= 4 and key.endswith('_reshaped'):
            reshape_type = parts[-2]  # 'row' 或 'col'
            
            if 'timestep' in key:
                # head_mean数据格式: head_mean_layer_{layer_idx}_timestep_{XX}_{reshape_type}_reshaped
                # parts = ['head', 'mean', 'layer', 'layer_idx', 'timestep', 'timestep_value', 'reshape_type', 'reshaped']
                data_type = '_'.join(parts[:2])  # 'head_mean'
                layer_idx = int(parts[3])  # layer_idx 在第4个位置
                timestep = int(parts[5])   # timestep_value 在第6个位置
                
                if data_type not in organized_data:
                    organized_data[data_type] = {}
                if layer_idx not in organized_data[data_type]:
                    organized_data[data_type][layer_idx] = {}
                if timestep not in organized_data[data_type][layer_idx]:
                    organized_data[data_type][layer_idx][timestep] = {}
                
                organized_data[data_type][layer_idx][timestep][f'{reshape_type}_reshaped'] = matrix
            else:
                # mean数据格式: mean_layer_{layer_idx}_{reshape_type}_reshaped
                # parts = ['mean', 'layer', 'layer_idx', 'reshape_type', 'reshaped']
                data_type = parts[0]  # 'mean'
                layer_idx = int(parts[2])  # layer_idx 在第3个位置
                
                if data_type not in organized_data:
                    organized_data[data_type] = {}
                if layer_idx not in organized_data[data_type]:
                    organized_data[data_type][layer_idx] = {}
                
                organized_data[data_type][layer_idx][f'{reshape_type}_reshaped'] = matrix
    
    return organized_data

def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    
    # 解析数据集和样本ID
    dataset_ids = parse_ids(args.dataset_ids)
    sample_ids = parse_ids(args.sample_ids)
    
    # 解析时间步ID
    timesteps = parse_ids(args.timesteps)
    
    # 处理数据类型参数
    if 'both' in args.data_types:
        data_types_to_process = ['mean', 'head_mean']
    else:
        data_types_to_process = [t for t in args.data_types if t in ['mean', 'head_mean']]
    
    # 加载配置文件
    try:
        config = load_config(args.config_path)
        plot_config = config['plot_modes'][args.plot_mode]
    except Exception as e:
        print(f"加载配置文件失败: {e}")
        return
    
    print("空间注意力矩阵重塑可视化")
    print("=" * 50)
    print(f"配置文件: {args.config_path}")
    print(f"绘图模式: {args.plot_mode} - {plot_config['description']}")
    print(f"重塑数据目录: {args.reshaped_data_dir}")
    print(f"图片输出目录: {args.output_dir}")
    print(f"数据类型: {data_types_to_process}")
    print(f"数据集ID: {dataset_ids if dataset_ids else 'all'}")
    print(f"样本ID: {sample_ids if sample_ids else 'all'}")
    if 'head_mean' in data_types_to_process:
        print(f"时间步: {timesteps if timesteps else 'all'}")
    print(f"图片尺寸: {plot_config['figure']['figsize']}")
    print(f"输出格式: SVG (矢量格式，无需分辨率设置)")
    print(f"颜色映射: {plot_config['colormap']}")
    print(f"处理所有层: {args.process_all_layers}")
    if not args.process_all_layers:
        print(f"指定层: {args.specific_layers}")
    print()
    
    # 创建输出文件夹
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取可用的数据集
    reshaped_data_dir = Path(args.reshaped_data_dir)
    if not reshaped_data_dir.exists():
        print(f"重塑数据目录不存在: {reshaped_data_dir}")
        return
    
    available_dataset_dirs = [d for d in reshaped_data_dir.iterdir() 
                             if d.is_dir() and d.name.startswith('dataset_')]
    available_dataset_ids = sorted([d.name.split('_')[1] for d in available_dataset_dirs])
    
    if dataset_ids is not None:
        filtered_dataset_ids = [d for d in dataset_ids if d in available_dataset_ids]
    else:
        filtered_dataset_ids = available_dataset_ids
    
    print(f"可用数据集: {available_dataset_ids}")
    print(f"将处理数据集: {filtered_dataset_ids}")
    
    total_plots = 0
    
    # 处理每个数据集
    for dataset_id in tqdm(filtered_dataset_ids, desc="处理数据集"):
        dataset_dir = reshaped_data_dir / f'dataset_{dataset_id}'
        if not dataset_dir.exists():
            continue
        
        # 获取可用的样本
        available_sample_files = [f for f in dataset_dir.glob('sample_*.npz')]
        available_sample_ids = sorted([f.stem.split('_')[1] for f in available_sample_files])
        
        if sample_ids is not None:
            filtered_sample_ids = [s for s in sample_ids if s in available_sample_ids]
        else:
            filtered_sample_ids = available_sample_ids
        
        print(f"数据集 {dataset_id}: 可用样本 {available_sample_ids}, 将处理 {filtered_sample_ids}")
        
        # 创建数据集文件夹
        dataset_output_dir = output_dir / f'dataset_{dataset_id}'
        dataset_output_dir.mkdir(parents=True, exist_ok=True)
        
        for sample_id in tqdm(filtered_sample_ids, desc=f"处理数据集 {dataset_id} 的样本", leave=False):
            # 加载重塑数据
            reshaped_data = load_reshaped_data(args.reshaped_data_dir, dataset_id, sample_id)
            if reshaped_data is None:
                continue
            
            # 创建样本文件夹
            sample_output_dir = dataset_output_dir / f'sample_{sample_id}'
            sample_output_dir.mkdir(parents=True, exist_ok=True)
            
            # 处理每种数据类型
            for data_type in data_types_to_process:
                if data_type not in reshaped_data:
                    continue
                
                # 获取可用的层
                available_layers = sorted(reshaped_data[data_type].keys())
                if args.process_all_layers:
                    layers_to_process = available_layers
                else:
                    layers_to_process = [layer for layer in args.specific_layers if layer in available_layers]
                
                # 处理每一层
                for layer_idx in layers_to_process:
                    if layer_idx not in reshaped_data[data_type]:
                        continue
                    
                    layer_data = reshaped_data[data_type][layer_idx]
                    
                    # 创建层文件夹
                    layer_output_dir = sample_output_dir / f'layer_{layer_idx}'
                    layer_output_dir.mkdir(parents=True, exist_ok=True)
                    
                    # 检查是否是时间步数据
                    if isinstance(list(layer_data.values())[0], dict):
                        # head_mean数据：有多个时间步
                        available_timesteps = sorted(layer_data.keys())
                        
                        # 根据用户指定过滤时间步
                        if timesteps is not None:
                            # 将timesteps转换为整数列表
                            try:
                                timesteps_int = [int(t) for t in timesteps]
                                filtered_timesteps = [t for t in timesteps_int if t in available_timesteps]
                            except ValueError:
                                print(f"警告: 无效的时间步参数 {timesteps}，将处理所有时间步")
                                filtered_timesteps = available_timesteps
                        else:
                            filtered_timesteps = available_timesteps
                        
                        for timestep in filtered_timesteps:
                            timestep_data = layer_data[timestep]
                            if 'row_reshaped' not in timestep_data or 'col_reshaped' not in timestep_data:
                                continue
                            
                            row_reshaped = timestep_data['row_reshaped']
                            col_reshaped = timestep_data['col_reshaped']
                            
                            # 生成图片
                            title_prefix = f'Dataset {dataset_id} Sample {sample_id} Layer {layer_idx} Timestep {timestep}'
                            filename = get_filename_from_config(config, args.plot_mode, data_type, timestep)
                            save_path = layer_output_dir / f'{filename}.svg'
                            
                            if args.plot_mode == 'comparison':
                                plot_reshaped_attention_comparison(row_reshaped, col_reshaped,
                                                                 title_prefix, save_path, plot_config)
                            elif args.plot_mode == 'single_row':
                                plot_single_reshaped_attention(row_reshaped, title_prefix, 
                                                              save_path, plot_config)
                            elif args.plot_mode == 'single_col':
                                plot_single_reshaped_attention(col_reshaped, title_prefix, 
                                                              save_path, plot_config)
                            
                            total_plots += 1
                    
                    else:
                        # mean数据：单个矩阵对
                        if 'row_reshaped' not in layer_data or 'col_reshaped' not in layer_data:
                            continue
                        
                        row_reshaped = layer_data['row_reshaped']
                        col_reshaped = layer_data['col_reshaped']
                        
                        # 生成图片
                        title_prefix = f'Dataset {dataset_id} Sample {sample_id} Layer {layer_idx}'
                        filename = get_filename_from_config(config, args.plot_mode, data_type)
                        save_path = layer_output_dir / f'{filename}.svg'
                        
                        if args.plot_mode == 'comparison':
                            plot_reshaped_attention_comparison(row_reshaped, col_reshaped,
                                                             title_prefix, save_path, plot_config)
                        elif args.plot_mode == 'single_row':
                            plot_single_reshaped_attention(row_reshaped, title_prefix, 
                                                          save_path, plot_config)
                        elif args.plot_mode == 'single_col':
                            plot_single_reshaped_attention(col_reshaped, title_prefix, 
                                                          save_path, plot_config)
                        
                        total_plots += 1
    
    print(f"\n完成！共生成 {total_plots} 张图片")
    print(f"图片已保存到 {output_dir}")

if __name__ == '__main__':
    main()
