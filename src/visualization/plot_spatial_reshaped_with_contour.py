#!/usr/bin/env python3
"""
绘制重塑的空间注意力矩阵热力图，并添加原场的0值轮廓线（所有时间步）
从保存的重塑数据中读取按行或按列重塑的矩阵，并从原始数据中提取0值轮廓
支持行重塑和列重塑两种模式
使用平均Ap数据（所有时间步平均），对应流场所有16个时间步(0-15)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import seaborn as sns
import h5py
import yaml
import argparse
from pathlib import Path
from tqdm import tqdm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os

# 设置字体
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

def load_config(config_path):
    """加载YAML配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='绘制重塑的空间注意力矩阵热力图，并添加原场的0值轮廓线')
    
    # 路径参数
    parser.add_argument('--reshaped-data-dir', type=str,
        default='data/processed/As_reshaped',
        help='重塑数据文件夹路径')
    parser.add_argument('--output-dir', type=str,
        default='output/plots_As_reshaped_with_contour',
        help='图片输出文件夹路径')
    parser.add_argument('--config', type=str,
        default='/home/pkq2/project/transformer_PDE_2/config/plot_spatial_reshaped_with_contour_config.yaml',
        help='配置文件路径')

    # 重塑类型参数
    parser.add_argument('--reshape-type', type=str, default='both',
                        help='重塑类型：row（行重塑）、col（列重塑）或both（两种都生成）（默认: row）')

    # 处理参数 - 支持多个值和"all"
    parser.add_argument(
        '--dataset-id', type=str, nargs='+', default='2,3,4,5,6,8',
        help='指定数据集ID，支持多个值或"all"（默认: 0）。例如: --dataset-id 0 1 或 --dataset-id all')
    parser.add_argument(
        '--sample-id', type=str, nargs='+', default=['all'],
        help='指定样本ID，支持多个值或"all"（默认: 100）。例如: --sample-id 100 101 或 --sample-id all')
    parser.add_argument(
        '--layer-id', type=str, nargs='+', default=['all'],
        help='指定层ID，支持多个值或"all"（默认: all）。例如: --layer-id 0 1 2 或 --layer-id all')
    parser.add_argument(
        '--time-id', type=str, nargs='+', default=['all'],
        help='指定时间步ID，支持多个值或"all"（默认: 5）。例如: --time-id 5 10 15 或 --time-id all')
    parser.add_argument(
        '--channel-id', type=str, nargs='+', default=['all'],
        help='指定通道ID，支持多个值或"all"（默认: 1）。例如: --channel-id 0 1 或 --channel-id all')
    
    # 轮廓线控制参数
    parser.add_argument('--draw-contour', action='store_true', default=True,
                        help='是否绘制速度场0值轮廓线（默认: True）')
    parser.add_argument('--no-contour', action='store_false', dest='draw_contour',
                        help='不绘制速度场0值轮廓线')
    
    return parser.parse_args()

def resolve_parameter_values(param_list, available_values, param_name):
    """解析参数值，支持"all"和多个具体值"""
    if 'all' in param_list:
        return sorted(available_values)
    
    # 验证每个值是否在可用值中
    resolved_values = []
    for value in param_list:
        if value in available_values:
            resolved_values.append(value)
        else:
            print(f"警告: {param_name} '{value}' 不存在，可用值: {sorted(available_values)}")
    
    return sorted(list(set(resolved_values)))  # 去重并排序

def resolve_numeric_parameter_values(param_list, max_value, param_name):
    """解析数值参数值，支持"all"和多个具体值"""
    if 'all' in param_list:
        return list(range(max_value))
    
    # 验证并转换每个值为整数
    resolved_values = []
    for value in param_list:
        try:
            int_value = int(value)
            if 0 <= int_value < max_value:
                resolved_values.append(int_value)
            else:
                print(f"警告: {param_name} '{value}' 超出范围 (0-{max_value-1})")
        except ValueError:
            print(f"警告: {param_name} '{value}' 不是有效的整数")
    
    return sorted(list(set(resolved_values)))  # 去重并排序

def load_h5_velocity_data(file_path):
    """从H5文件中加载速度场数据 (samples, channels, time, height, width)"""
    print(f"正在加载H5文件: {file_path}")
    try:
        with h5py.File(file_path, 'r') as file:
            # 检查可用的键
            available_keys = list(file.keys())
            print(f"可用的键: {available_keys}")
            
            # 加载 u_in 数据
            u_in = None
            if 'u_in' in available_keys:
                u_in = file['u_in'][:].astype('float32')
                print(f"u_in 数据形状: {u_in.shape}")
            elif 'u' in available_keys:
                u_in = file['u'][:].astype('float32')
                print(f"使用 u 作为 u_in 数据，形状: {u_in.shape}")
                
            return u_in
    except Exception as e:
        print(f"加载文件 {file_path} 时出错: {e}")
        return None

def load_reshaped_data(data_dir):
    """加载重塑后的数据（平均数据）- 适配新的数据格式"""
    data_dir = Path(data_dir)
    all_data = {}
    
    for dataset_dir in data_dir.iterdir():
        if not dataset_dir.is_dir() or not dataset_dir.name.startswith('dataset_'):
            continue
        
        dataset_id = dataset_dir.name.split('_')[1]
        all_data[dataset_id] = {}
        
        # 新格式：直接加载sample_{sample_id}.npz文件
        for sample_file in dataset_dir.glob('sample_*.npz'):
            # 提取样本ID: sample_{sample_id}.npz
            sample_id = sample_file.stem.split('_')[1]
            all_data[dataset_id][sample_id] = {}
            
            # 加载整个样本文件
            data = dict(np.load(sample_file))
            
            # 解析数据结构并重新组织
            for key, matrix in data.items():
                # 解析键名: {data_type}_layer_{layer_idx}_row_reshaped 或 {data_type}_layer_{layer_idx}_col_reshaped
                # 或 {data_type}_layer_{layer_idx}_timestep_{XX}_row_reshaped 等
                parts = key.split('_')
                
                if len(parts) >= 4 and 'layer' in parts and ('row_reshaped' in key or 'col_reshaped' in key):
                    # 找到data_type和layer_idx
                    layer_idx_pos = parts.index('layer') + 1
                    if layer_idx_pos < len(parts):
                        try:
                            layer_idx = int(parts[layer_idx_pos])
                        except ValueError:
                            continue
                        
                        # 初始化层数据结构
                        if layer_idx not in all_data[dataset_id][sample_id]:
                            all_data[dataset_id][sample_id][layer_idx] = {}
                        
                        # 提取数据类型和重塑类型
                        if 'timestep' in key:
                            # 头平均数据，暂时跳过，因为当前脚本主要处理mean数据
                            continue
                        else:
                            # mean数据
                            if 'row_reshaped' in key:
                                all_data[dataset_id][sample_id][layer_idx]['row_reshaped'] = matrix
                            elif 'col_reshaped' in key:
                                all_data[dataset_id][sample_id][layer_idx]['col_reshaped'] = matrix
    
    return all_data

def extract_velocity_zero_contours(velocity_field, levels=[0]):
    """
    从速度场数据中提取0值轮廓线
    
    Args:
        velocity_field: 速度场数据，形状为 (height, width)
        levels: 等高线水平，默认为[0]
        
    Returns:
        list: 轮廓线路径列表
    """
    # 创建临时图形来提取轮廓
    fig, ax = plt.subplots(figsize=(1, 1))
    
    # 生成坐标网格
    y, x = np.mgrid[0:velocity_field.shape[0], 0:velocity_field.shape[1]]
    
    # 提取轮廓
    cs = ax.contour(x, y, velocity_field, levels=levels, colors='white', linewidths=1)
    
    # 提取轮廓路径 - 修复API变更问题
    contour_paths = []
    try:
        # 新版本matplotlib使用allsegs属性
        if hasattr(cs, 'allsegs'):
            for level_segs in cs.allsegs:
                for seg in level_segs:
                    if len(seg) > 0:
                        contour_paths.append(seg)
        # 旧版本matplotlib使用collections属性
        elif hasattr(cs, 'collections'):
            for collection in cs.collections:
                for path in collection.get_paths():
                    vertices = path.vertices
                    if len(vertices) > 0:
                        contour_paths.append(vertices)
        else:
            print("警告：无法从contour对象中提取路径")
    except Exception as e:
        print(f"警告：提取轮廓路径时出错: {e}")
    
    plt.close(fig)
    return contour_paths

def map_velocity_contour_to_attention(contour_paths, velocity_shape, attention_shape):
    """
    将速度场轮廓坐标映射到注意力矩阵坐标系
    
    Args:
        contour_paths: 轮廓路径列表
        velocity_shape: 速度场形状 (height, width)
        attention_shape: 注意力矩阵形状 (100, 100)
        
    Returns:
        list: 映射后的轮廓路径列表
    """
    mapped_paths = []
    
    # 计算缩放比例
    scale_y = (attention_shape[0] - 1) / (velocity_shape[0] - 1)
    scale_x = (attention_shape[1] - 1) / (velocity_shape[1] - 1)
    
    for path in contour_paths:
        if len(path) > 0:
            # 映射坐标
            mapped_path = path.copy()
            mapped_path[:, 0] = path[:, 0] * scale_x  # x坐标
            mapped_path[:, 1] = path[:, 1] * scale_y  # y坐标
            mapped_paths.append(mapped_path)
    
    return mapped_paths

def plot_reshaped_with_contour_single_channel(reshaped_data, velocity_slice, title_prefix, save_path, channel_idx,
                                                   reshape_type, figsize=(8, 8), dpi=300, cmap='Reds',
                                                   title_fontsize=24, colorbar_tick_fontsize=18,
                                                   contour_color='white', contour_linewidth=1.5,
                                                   colorbar_ticks=5, colorbar_shrink=0.8,
                                                   axis_label_fontsize=22, origin_label_fontsize=20,
                                                   colorbar_width=0.03, title_pad=20, label_pad=10,
                                                   colorbar_exponent_fontsize=20, image_format='svg',
                                                   draw_contour=True):
    """
    绘制重塑的矩阵并添加单个通道的0值轮廓线
    
    Args:
        reshaped_data: 重塑的矩阵（行重塑或列重塑）
        velocity_slice: 单个通道的速度场数据 (height, width)
        title_prefix: 标题前缀
        save_path: 保存路径
        channel_idx: 通道索引
        reshape_type: 重塑类型 ('row' 或 'col')
        figsize: 图片大小
        dpi: 分辨率
        cmap: 颜色映射
        title_fontsize: 标题字体大小
        colorbar_tick_fontsize: colorbar刻度字体大小
        contour_color: 轮廓线颜色
        contour_linewidth: 轮廓线宽度
        colorbar_ticks: colorbar刻度数量
        colorbar_shrink: colorbar收缩比例
        axis_label_fontsize: 坐标轴标签字体大小
        origin_label_fontsize: 原点标签字体大小
        colorbar_width: colorbar宽度
        title_pad: 标题填充距离
        label_pad: 标签填充距离
        colorbar_exponent_fontsize: colorbar指数字体大小
        image_format: 图像格式
        draw_contour: 是否绘制轮廓线
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # 计算颜色范围
    vmin = reshaped_data.min()
    vmax = reshaped_data.max()
    
    # 使用seaborn绘制热力图
    im = sns.heatmap(reshaped_data, cmap=cmap, vmin=vmin, vmax=vmax,
                     ax=ax, cbar=False, square=True, 
                     xticklabels=False, yticklabels=False)
    
    # 添加10x10网格线
    # 网格线位置：每10个像素一条线，共11条线（0, 10, 20, ..., 100）
    grid_positions = np.arange(0, 110, 10)  # 直接使用整数位置
    for pos in grid_positions:
        ax.axhline(y=pos, color='white', linewidth=1, alpha=0.8)  # 水平网格线
        ax.axvline(x=pos, color='white', linewidth=1, alpha=0.8)  # 垂直网格线
    
    # 提取并绘制该通道的0值轮廓线（如果启用）
    if draw_contour:
        try:
            # 提取轮廓
            contour_paths = extract_velocity_zero_contours(velocity_slice, levels=[0])
            
            # 将轮廓坐标映射到注意力矩阵坐标系
            mapped_paths = map_velocity_contour_to_attention(contour_paths, velocity_slice.shape, reshaped_data.shape)
            
            # 绘制轮廓线
            for path in mapped_paths:
                if len(path) > 0:
                    ax.plot(path[:, 0], path[:, 1], color=contour_color, 
                           linewidth=contour_linewidth, alpha=0.8)
                           
        except Exception as e:
            print(f"警告：提取通道 {channel_idx} 轮廓线时出错: {e}")
    else:
        print(f"跳过通道 {channel_idx} 轮廓线绘制")
    
    # # 设置标题
    # ax.set_title(f'{reshape_type}-reshaped $A_p$ with contour', fontsize=title_fontsize, pad=title_pad)
    
    # 添加坐标轴标签 - 不随重塑类型改变，始终保持一致
    ax.xaxis.set_label_position('top')
    ax.set_xlabel('$x$', fontsize=axis_label_fontsize, ha='center', style='italic', labelpad=label_pad)
    ax.set_ylabel('$y$', fontsize=axis_label_fontsize, va='center', style='italic', rotation=0, labelpad=label_pad+10)
    
    # # 添加原点标记O
    # ax.text(-0.05, 1.02, 'O', transform=ax.transAxes, 
    #         fontsize=origin_label_fontsize, style='italic',
    #         horizontalalignment='center', verticalalignment='center')
    
    # 添加刻度：每10个像素一个刻度，标在像素前，无标签
    tick_positions = np.arange(0, 110, 10)
    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    # 将刻度移到下边界和右边界
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('left')
    
    # 添加黑色边框
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(0.5)
    
    # 添加colorbar，使用指定宽度
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=f"{colorbar_width*100:.1f}%", pad=0.2)
    
    # 创建colorbar的mappable对象
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(mappable, cax=cax)
    
    # 设置colorbar刻度 - 强制使用科学计数法
    ticks = np.linspace(vmin, vmax, colorbar_ticks)
    cbar.set_ticks(ticks)
    
    # 计算指数并格式化刻度标签
    if np.max(np.abs(ticks)) != 0:
        # 找到数据的数量级（使用最大绝对值）
        max_abs_value = np.max(np.abs(ticks[ticks != 0]))
        magnitude = int(np.floor(np.log10(max_abs_value)))
        
        # 强制使用科学计数法
        # 缩放刻度值
        scaled_ticks = ticks / (10 ** magnitude)
        # 格式化刻度标签，保留1位小数
        tick_labels = [f'{tick:.1f}' for tick in scaled_ticks]
        cbar.set_ticklabels(tick_labels)
        
        # 在colorbar上方添加指数标识
        cbar.ax.text(2, 1.07, f'×10$^{{{magnitude}}}$', 
                    transform=cbar.ax.transAxes, 
                    ha='center', va='bottom',
                    fontsize=colorbar_exponent_fontsize)
    else:
        # 所有值都是0的情况
        cbar.set_ticklabels([f'{tick:.1f}' for tick in ticks])
        cbar.ax.text(0.8, 1.01, '×10$^{0}$', 
                    transform=cbar.ax.transAxes, 
                    ha='center', va='bottom',
                    fontsize=colorbar_exponent_fontsize)
    
    cbar.ax.tick_params(labelsize=colorbar_tick_fontsize)
    
    plt.tight_layout()
    plt.savefig(save_path, format=image_format, bbox_inches='tight')
    plt.close()

def plot_small_patches(reshaped_data, save_path_base, channel_idx, 
                      patch_positions, cmap='Reds', image_format='svg'):
    """
    将数据切成10x10的小图，并生成指定位置的小图（不包含轮廓线）
    
    Args:
        reshaped_data: 重塑的矩阵 (100, 100)
        save_path_base: 保存路径基础（不包含位置索引）
        channel_idx: 通道索引
        patch_positions: 要生成的小图位置列表 [(row, col), ...]
        cmap: 颜色映射
        image_format: 图像格式
    """
    # 计算全局颜色范围
    vmin = reshaped_data.min()
    vmax = reshaped_data.max()
    
    # 将100x100的数据切成10x10的小块
    patch_size = 10
    grid_size = 10
    
    for row_idx, col_idx in patch_positions:
        if row_idx >= grid_size or col_idx >= grid_size or row_idx < 0 or col_idx < 0:
            print(f"警告：位置索引 ({row_idx}, {col_idx}) 超出范围 (0-{grid_size-1})")
            continue
            
        # 提取对应的小块数据
        start_row = row_idx * patch_size
        end_row = start_row + patch_size
        start_col = col_idx * patch_size
        end_col = start_col + patch_size
        
        # 提取注意力数据小块
        patch_data = reshaped_data[start_row:end_row, start_col:end_col]
        
        # 创建图像，不包含标题、颜色条等
        fig, ax = plt.subplots(1, 1, figsize=(2, 2))
        
        # 使用seaborn绘制热力图，使用全局颜色范围
        sns.heatmap(patch_data, cmap=cmap, vmin=vmin, vmax=vmax,
                   ax=ax, cbar=False, square=True,
                   xticklabels=False, yticklabels=False)
        
        # 小图中不绘制轮廓线
        
        # 移除所有刻度和标签
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        
        # 移除边框
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        # 生成保存路径
        save_path = Path(str(save_path_base).replace(f'.{image_format}', f'_patch_{row_idx}_{col_idx}.{image_format}'))
        
        plt.tight_layout()
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig(save_path, format=image_format, bbox_inches='tight', pad_inches=0)
        plt.close()
        
        print(f"  生成小图: 位置 ({row_idx}, {col_idx}) -> {save_path.name}")

def plot_reshaped_with_contour(reshaped_data, velocity_field, title_prefix, save_path_template, reshape_type,
                                figsize=(8, 8), dpi=300, cmap='Reds',
                                title_fontsize=24, colorbar_tick_fontsize=18,
                                contour_colors=['white', 'cyan'], contour_linewidth=1.5,
                                colorbar_ticks=5):
    """
    为每个通道分别绘制重塑的矩阵和0值轮廓线
    
    Args:
        reshaped_data: 重塑的矩阵
        velocity_field: 速度场数据 (channels, height, width)
        title_prefix: 标题前缀
        save_path_template: 保存路径模板（不包含通道信息）
        reshape_type: 重塑类型 ('row' 或 'col')
        figsize: 图片大小
        dpi: 分辨率
        cmap: 颜色映射
        title_fontsize: 标题字体大小
        colorbar_tick_fontsize: colorbar刻度字体大小
        contour_colors: 轮廓线颜色列表，对应每个通道
        contour_linewidth: 轮廓线宽度
        colorbar_ticks: colorbar刻度数量
    
    Returns:
        int: 生成的图片数量
    """
    plots_created = 0
    
    try:
        if velocity_field.ndim == 3 and velocity_field.shape[0] >= 2:
            # 处理两个通道，分别生成图片
            for channel_idx in range(min(2, velocity_field.shape[0])):
                velocity_slice = velocity_field[channel_idx, :, :]
                contour_color = contour_colors[channel_idx % len(contour_colors)]
                
                # 生成该通道的保存路径
                save_path = save_path_template.parent / f'{save_path_template.stem}_channel_{channel_idx}.png'
                
                # 绘制单个通道的图像
                plot_reshaped_with_contour_single_channel(
                    reshaped_data, velocity_slice, title_prefix, save_path, channel_idx, reshape_type,
                    figsize, dpi, cmap, title_fontsize, colorbar_tick_fontsize,
                    contour_color, contour_linewidth, colorbar_ticks
                )
                
                plots_created += 1
        else:
            print(f"警告：速度场维度不正确或通道数不足: {velocity_field.shape}")
                       
    except Exception as e:
        print(f"警告：生成多通道图像时出错: {e}")
    
    return plots_created

def main():
    # 解析命令行参数
    args = parse_args()
    
    # 加载配置文件
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"错误：无法加载配置文件 {args.config}: {e}")
        return
    
    # 从配置文件和命令行参数合并配置
    # 路径配置
    RESHAPED_DATA_DIR = args.reshaped_data_dir
    OUTPUT_DIR = args.output_dir
    # os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    
    # H5数据集文件路径配置
    H5_FILE_PATHS = config['datasets']['h5_file_paths']
    DATASET_NAMES = config['datasets']['dataset_names']
    
    # 图片配置（直接使用配置文件）
    FIGSIZE = tuple(config['figure']['figsize'])
    DPI = config['figure']['dpi']
    IMAGE_FORMAT = config['figure']['format']
    CMAP = config['colors']['cmap']
    
    # 字体配置
    TITLE_FONTSIZE = config['fonts']['title_fontsize']
    AXIS_LABEL_FONTSIZE = config['fonts']['axis_label_fontsize']
    ORIGIN_LABEL_FONTSIZE = config['fonts']['origin_label_fontsize']
    
    # Colorbar配置
    COLORBAR_TICK_FONTSIZE = config['colorbar']['tick_fontsize']
    COLORBAR_EXPONENT_FONTSIZE = config['colorbar']['exponent_fontsize']
    COLORBAR_TICKS = config['colorbar']['ticks']
    COLORBAR_SHRINK = config['colorbar']['shrink']
    COLORBAR_WIDTH = config['colorbar']['width']
    
    # 布局配置
    TITLE_PAD = config['layout']['title_pad']
    LABEL_PAD = config['layout']['label_pad']
    
    # 轮廓线配置
    CONTOUR_COLORS = config['colors']['contour_colors']
    CONTOUR_LINEWIDTH = config['colors']['contour_linewidth']
    DRAW_CONTOUR = args.draw_contour  # 从命令行参数获取
    
    # 处理配置 - 从命令行参数获取（现在是列表）
    DATASET_IDS = args.dataset_id
    SAMPLE_IDS = args.sample_id  
    LAYER_IDS = args.layer_id
    TIME_IDS = args.time_id
    CHANNEL_IDS = args.channel_id
    RESHAPE_TYPE = args.reshape_type  # 新增：重塑类型
    
    # 小图配置
    PATCH_POSITIONS = [tuple(pos) for pos in config['patches']['positions']]
    
    print("重塑空间注意力矩阵可视化 (带速度场0值轮廓)")
    print("=" * 60)
    print(f"重塑数据目录: {RESHAPED_DATA_DIR}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"配置文件: {args.config}")
    print(f"重塑类型: {RESHAPE_TYPE}")
    print(f"H5数据集文件:")
    for i, (path, name) in enumerate(zip(H5_FILE_PATHS, DATASET_NAMES)):
        print(f"  数据集{i}: {name} -> {path}")
    print(f"图片尺寸: {FIGSIZE}")
    print(f"分辨率: {DPI}")
    print(f"图片格式: {IMAGE_FORMAT}")
    print(f"颜色映射: {CMAP}")
    print(f"标题字体大小: {TITLE_FONTSIZE}")
    print(f"Colorbar刻度字体大小: {COLORBAR_TICK_FONTSIZE}")
    print(f"Colorbar指数字体大小: {COLORBAR_EXPONENT_FONTSIZE}")
    print(f"坐标轴标签字体大小: {AXIS_LABEL_FONTSIZE}")
    print(f"原点标签字体大小: {ORIGIN_LABEL_FONTSIZE}")
    print(f"Colorbar收缩比例: {COLORBAR_SHRINK}")
    print(f"Colorbar宽度: {COLORBAR_WIDTH}")
    print(f"标题填充距离: {TITLE_PAD}")
    print(f"标签填充距离: {LABEL_PAD}")
    print(f"轮廓线颜色: {CONTOUR_COLORS}")
    print(f"轮廓线宽度: {CONTOUR_LINEWIDTH}")
    print(f"绘制轮廓线: {DRAW_CONTOUR}")
    print(f"指定数据集: {DATASET_IDS}")
    print(f"指定样本: {SAMPLE_IDS}")
    print(f"指定层: {LAYER_IDS}")
    print(f"指定时间步: {TIME_IDS}")
    print(f"指定通道: {CHANNEL_IDS}")
    print(f"小图位置: {PATCH_POSITIONS}")
    print()
    
    # 创建输出文件夹
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载H5数据集
    print("加载H5数据集...")
    h5_datasets = {}
    for i, (file_path, dataset_name) in enumerate(zip(H5_FILE_PATHS, DATASET_NAMES)):
        velocity_data = load_h5_velocity_data(file_path)
        if velocity_data is not None:
            h5_datasets[str(i)] = {
                'data': velocity_data,
                'name': dataset_name
            }
    
    # 加载重塑数据
    print("加载重塑数据...")
    reshaped_data = load_reshaped_data(RESHAPED_DATA_DIR)
    
    if not h5_datasets:
        print("没有找到任何H5数据集文件！")
        return
        
    if not reshaped_data:
        print("没有找到任何重塑数据文件！")
        return
    
    # 找到H5数据集和重塑数据的交集
    common_dataset_ids = set(h5_datasets.keys()) & set(reshaped_data.keys())
    common_dataset_ids = sorted(common_dataset_ids)
    
    if not common_dataset_ids:
        print("H5数据集和重塑数据没有共同的数据集！")
        print(f"H5数据集: {list(h5_datasets.keys())}")
        print(f"重塑数据集: {list(reshaped_data.keys())}")
        return
    
    # 解析数据集参数
    resolved_dataset_ids = resolve_parameter_values(DATASET_IDS, common_dataset_ids, '数据集ID')
    if not resolved_dataset_ids:
        print("没有有效的数据集ID！")
        return
    
    print(f"找到共同数据集: {common_dataset_ids}")
    print(f"将处理以下数据集: {resolved_dataset_ids}")
    
    total_plots = 0
    
    # 处理多个数据集
    for dataset_id in resolved_dataset_ids:
        print(f"\n处理数据集 {dataset_id}...")
        
        # 获取H5数据
        h5_info = h5_datasets[dataset_id]
        velocity_data = h5_info['data']
        dataset_name = h5_info['name']
        
        # 解析样本参数
        available_samples = list(reshaped_data[dataset_id].keys())
        resolved_sample_ids = resolve_parameter_values(SAMPLE_IDS, available_samples, '样本ID')
        
        if not resolved_sample_ids:
            print(f"  数据集 {dataset_id} 没有有效的样本ID")
            continue
        
        print(f"  数据集 {dataset_id} ({dataset_name}): 处理样本 {resolved_sample_ids}")
        
        # 处理多个样本
        for sample_id_str in resolved_sample_ids:
            try:
                sample_idx = int(sample_id_str)
            except ValueError:
                print(f"    错误：样本ID {sample_id_str} 无法转换为整数")
                continue
            
            # 检查样本索引是否在H5数据范围内
            if sample_idx >= velocity_data.shape[0]:
                print(f"    错误：样本 {sample_idx} 超出H5数据范围 ({velocity_data.shape[0]})")
                continue
            
            # 解析层参数
            available_layers = [str(layer_id) for layer_id in reshaped_data[dataset_id][sample_id_str].keys()]
            resolved_layer_ids = resolve_parameter_values(LAYER_IDS, available_layers, '层ID')
            
            if not resolved_layer_ids:
                print(f"    样本 {sample_idx} 没有有效的层ID")
                continue
            
            # 处理多个层
            for layer_id_str in resolved_layer_ids:
                try:
                    layer_idx = int(layer_id_str)
                except ValueError:
                    print(f"      错误：层ID {layer_id_str} 无法转换为整数")
                    continue
                
                try:
                    # 获取该层的平均数据
                    layer_data = reshaped_data[dataset_id][sample_id_str][layer_idx]
                    
                    # 根据重塑类型处理数据
                    reshape_types_to_process = []
                    if RESHAPE_TYPE == 'both':
                        reshape_types_to_process = ['row', 'col']
                    else:
                        reshape_types_to_process = [RESHAPE_TYPE]
                    
                    for current_reshape_type in reshape_types_to_process:
                        reshape_key = f'{current_reshape_type}_reshaped'
                        
                        # 检查是否包含指定类型的重塑数据
                        if reshape_key not in layer_data:
                            print(f"      错误：数据集 {dataset_id}, 样本 {sample_idx}, 层 {layer_idx} 缺少{current_reshape_type}重塑数据")
                            continue
                        
                        reshaped_matrix = layer_data[reshape_key]
                        
                        # 检查矩阵形状
                        if reshaped_matrix.shape != (100, 100):
                            print(f"      错误：数据集 {dataset_id}, 样本 {sample_idx}, 层 {layer_idx} {current_reshape_type}重塑矩阵形状不正确: {reshaped_matrix.shape}")
                            continue
                        
                        # 解析时间步参数
                        max_time = velocity_data.shape[2]
                        resolved_time_ids = resolve_numeric_parameter_values(TIME_IDS, max_time, '时间步ID')
                        
                        if not resolved_time_ids:
                            print(f"      层 {layer_idx} 没有有效的时间步ID")
                            continue
                        
                        # 处理多个时间步
                        for velocity_time_idx in resolved_time_ids:
                            # 获取对应时间步的速度场数据
                            velocity_field = velocity_data[sample_idx, :, velocity_time_idx, :, :]  # (channels, height, width)
                            
                            # 解析通道参数
                            max_channel = velocity_field.shape[0]
                            resolved_channel_ids = resolve_numeric_parameter_values(CHANNEL_IDS, max_channel, '通道ID')
                            
                            if not resolved_channel_ids:
                                print(f"        时间步 {velocity_time_idx} 没有有效的通道ID")
                                continue
                            
                            # 处理多个通道
                            for channel_idx in resolved_channel_ids:
                                # 获取指定通道的速度场数据
                                velocity_slice = velocity_field[channel_idx, :, :]
                                contour_color = CONTOUR_COLORS[channel_idx % len(CONTOUR_COLORS)]
                                
                                # 绘制指定通道的图像 - 保存到输出目录
                                title_prefix = f'Dataset {dataset_id} ({dataset_name}) Sample {sample_idx} Layer {layer_idx} Time {velocity_time_idx}'
                                
                                # 根据是否绘制轮廓线添加文件名标识
                                contour_suffix = "_with_contour" if DRAW_CONTOUR else "_no_contour"
                                # 在输出路径下增加 dataset 与 sample，再到 layer 和 channel 四级子目录
                                out_dir_nested = (
                                    output_dir
                                    / f'dataset_{dataset_id}'
                                    / f'sample_{sample_idx}'
                                    / f'layer_{layer_idx}'
                                    / f'channel_{channel_idx}'
                                )
                                out_dir_nested.mkdir(parents=True, exist_ok=True)
                                save_path = out_dir_nested / (
                                    f'dataset_{dataset_id}_sample_{sample_idx}_layer_{layer_idx}_time_{velocity_time_idx:03d}_channel_{channel_idx}_{current_reshape_type}{contour_suffix}.{IMAGE_FORMAT}'
                                )
                                
                                plot_reshaped_with_contour_single_channel(
                                    reshaped_matrix, velocity_slice, title_prefix, save_path, channel_idx, current_reshape_type,
                                    FIGSIZE, DPI, CMAP, TITLE_FONTSIZE, COLORBAR_TICK_FONTSIZE,
                                    contour_color, CONTOUR_LINEWIDTH, COLORBAR_TICKS, COLORBAR_SHRINK,
                                    AXIS_LABEL_FONTSIZE, ORIGIN_LABEL_FONTSIZE, COLORBAR_WIDTH,
                                    TITLE_PAD, LABEL_PAD, COLORBAR_EXPONENT_FONTSIZE, IMAGE_FORMAT,
                                    DRAW_CONTOUR
                                )
                                
                                # 生成指定位置的小图
                                if PATCH_POSITIONS:  # 只在有小图位置配置时才生成
                                    plot_small_patches(
                                        reshaped_matrix, save_path, channel_idx,
                                        PATCH_POSITIONS, CMAP, IMAGE_FORMAT
                                    )
                                    total_plots += 1 + len(PATCH_POSITIONS)  # 大图 + 小图数量
                                else:
                                    total_plots += 1  # 只有大图
                                
                                print(f"        完成: dataset_{dataset_id}_sample_{sample_idx}_layer_{layer_idx}_time_{velocity_time_idx:03d}_channel_{channel_idx}_{current_reshape_type}")
                        
                except Exception as e:
                    print(f"      错误：处理数据集 {dataset_id}, 样本 {sample_idx}, 层 {layer_idx} 时出错: {e}")
                    continue
    
    print(f"\n完成！共生成 {total_plots} 张图像")
    print(f"处理配置: 数据集 {resolved_dataset_ids}, 样本 {SAMPLE_IDS}, 层 {LAYER_IDS}, 时间步 {TIME_IDS}, 通道 {CHANNEL_IDS}")
    print(f"重塑类型: {RESHAPE_TYPE}")
    print(f"图片已保存到 {output_dir}")
    print("数据使用策略：Ap使用平均数据（取所有时间步平均），流场使用指定时间步的指定通道")
    if total_plots > 0:
        if PATCH_POSITIONS:
            print(f"每组生成: 1张大图 + {len(PATCH_POSITIONS)}张小图，小图位置: {PATCH_POSITIONS}")
        else:
            print("每组生成: 1张大图")
        print("文件命名格式: dataset_[ID]_sample_[ID]_layer_[ID]_time_[ID]_channel_[ID]_[reshape_type]_[with/no]_contour.[格式]")
        print("小图命名格式: [大图名]_patch_[行]_[列].[格式]")
        print("其中 Ap为层的平均数据，流场为指定时间步的指定通道")
        print("重塑类型标识: row表示行重塑，col表示列重塑")
        print("轮廓线标识: _with_contour 表示包含轮廓线，_no_contour 表示不包含轮廓线")

if __name__ == '__main__':
    main()
