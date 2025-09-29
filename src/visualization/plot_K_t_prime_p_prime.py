#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import matplotlib
import matplotlib.ticker as ticker
import argparse
import yaml
import sys
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 设置matplotlib后端以提高性能
matplotlib.use('Agg')  # 使用非交互式后端

# 设置字体
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

# 性能优化参数
plt.rcParams['figure.max_open_warning'] = 0  # 禁用最大打开图形警告
plt.rcParams['axes.formatter.useoffset'] = False  # 关闭科学计数法偏移

def parse_arguments():
    parser = argparse.ArgumentParser(description='绘制K张量热力图')
    
    # 路径参数
    parser.add_argument('--data-dir', type=str,
                       default='data/processed/K_simplified',
                       help='K张量数据目录路径')
    parser.add_argument('--output-dir', type=str,
                       default='output/plots_K_simplified',
                       help='图片输出目录路径')  
    parser.add_argument('--config', type=str,
                       default='/home/pkq2/project/transformer_PDE_2/config/plot_K_t_prime_p_prime_config.yaml',
                       help='YAML配置文件路径')
    
    # 数据选择参数
    parser.add_argument('--datasets', type=str, default='all',
                       help='要处理的数据集，支持格式: "all", "0", "0,1,2", "0-2"')
    
    parser.add_argument('--samples', type=str, default='all',
                       help='要处理的样本，支持格式: "all", "0", "0,1,2", "0-5"')
    parser.add_argument('--layers', type=str, default='all',
                       help='要处理的层，支持格式: "all", "0", "0,1,2", "0-5" ')
    parser.add_argument('--time-steps', type=str, default='all',
                       help='要处理的时间步，支持格式: "all", "0", "0,1,2", "0-5", 设为None表示不处理')
    parser.add_argument('--space-steps', type=str, default='all',
                       help='要处理的空间步，支持格式: "all", "0", "0,1,2", "0-5", 设为None表示不处理')
    
    # 绘制选项参数
    parser.add_argument('--plot-main', action='store_true', default=True,
                       help='是否绘制主要K张量 (默认: True)')
    parser.add_argument('--no-plot-main', dest='plot_main', action='store_false',
                       help='不绘制主要K张量')
    parser.add_argument('--plot-terms', action='store_true', default=True,
                       help='是否绘制K张量四个项 (默认: False)')
    parser.add_argument('--no-plot-terms', dest='plot_terms', action='store_false',
                       help='不绘制K张量四个项')
    
    # 文字标注参数
    parser.add_argument('--text-annotations', type=str, default=None,
                       help='在指定位置添加文字标注，格式: "x1,y1,text1;x2,y2,text2" (默认: None),5+10*t')
    
    return parser.parse_args()

def load_config(config_path):
    """
    加载YAML配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        dict: 配置字典
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"警告：配置文件 {config_path} 不存在，将使用默认配置")
        return None
    except yaml.YAMLError as e:
        print(f"警告：配置文件 {config_path} 格式错误: {e}，将使用默认配置")
        return None

def reshape_t_prime_p_prime_for_visualization(matrix):
    """
    将K(t',p')张量重塑用于可视化：
    1. 将s'维度（p'维度，100维）重塑为10x10
    2. 按t'从左到右拼接
    
    Args:
        matrix: 形状为(t', p')的矩阵，通常是(100, 100)
        
    Returns:
        numpy.ndarray: 重新排列后的矩阵，形状为(10, t'*10)
    """
    t_prime_max, p_prime_max = matrix.shape
    
    if p_prime_max != 100:
        raise ValueError(f"p'维度应为100，实际为{p_prime_max}")
    
    # 每个t'对应一个10x10的子图
    # 最终图像高度为10，宽度为t'_max * 10
    final_height = 10
    final_width = t_prime_max * 10
    
    # 初始化最终矩阵
    final_matrix = np.zeros((final_height, final_width))
    
    # 对每个t'进行处理
    for t_idx in range(t_prime_max):
        # 取出当前t'对应的p'向量，形状为(100,)
        p_prime_vector = matrix[t_idx, :]  # (100,)
        
        # 将p'向量重塑为10x10
        p_prime_reshaped = p_prime_vector.reshape(10, 10)
        
        # 放置到最终矩阵的对应位置
        start_col = t_idx * 10
        end_col = start_col + 10
        
        final_matrix[:, start_col:end_col] = p_prime_reshaped
    
    return final_matrix

def  plot_heatmap_fast(matrix, title, save_path,
                     figsize=(10, 8), dpi=150, cmap='viridis',
                     title_fontsize=16, colorbar_tick_fontsize=12,
                     label_fontsize=14, colorbar_exponent_fontsize=12,
                     title_pad=20, label_pad=10,
                     add_grid=True, grid_interval=10, colorbar_tick_count=7,
                     colorbar_shrink=0.9, colorbar_width=0.03, colorbar_pad=0.1,
                     colorbar_position='right', colorbar_bbox=[0.92, 0.1, 0.02, 0.8],
                     use_custom_colorbar=False,
                     margins=None,
                     is_tpp=True, is_t_prime_p_prime_reshaped=False,
                     vmax_ratio=None, text_annotations=None):
    """
    快速绘制单个热力图（优化版本）
    
    Args:
        matrix: 要绘制的矩阵
        title: 图片标题
        save_path: 保存路径
        figsize: 图片大小
        dpi: 分辨率（默认降低到150以提高速度）
        cmap: 颜色映射
        title_fontsize: 标题字体大小
        colorbar_tick_fontsize: colorbar刻度字体大小
        label_fontsize: 坐标轴标签字体大小
        colorbar_exponent_fontsize: colorbar指数部分字体大小
        title_pad: 标题与图的间距
        label_pad: 坐标轴标签与轴的间距
        add_grid: 是否添加网格
        grid_interval: 网格间隔
        colorbar_tick_count: colorbar刻度数量
        colorbar_shrink: colorbar缩放比例
        colorbar_width: colorbar宽度
        colorbar_pad: colorbar与主图的间距
        is_tpp: 是否为K(t,p,p')图（True）还是K(t,p,t')图（False）
        is_t_prime_p_prime_reshaped: 是否为K(t',p')重塑后的图像
        vmax_ratio: 颜色上限比例，如果设置则使用 max_value * vmax_ratio 作为vmax
        text_annotations: 文字标注列表，格式为[(x, y, text, kwargs), ...]
    """
    # 使用更快的绘图方法
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # 计算数据的实际范围
    vmin = matrix.min()
    vmax_original = matrix.max()
    
    # 如果设置了vmax_ratio，则使用自定义的颜色上限
    if vmax_ratio is not None:
        vmax = vmax_original * vmax_ratio
    else:
        vmax = vmax_original
    
    # 使用seaborn绘制热力图以生成更好的矢量图
    im = sns.heatmap(matrix, cmap=cmap, ax=ax, 
                    vmin=vmin, vmax=vmax,
                    cbar=False,  # 我们稍后自定义colorbar
                    square=True,  # 保持方形比例
                    xticklabels=False, yticklabels=False)  # 不显示刻度标签
    
    # 添加黑色边框
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(0.5)
    
    # 添加colorbar - 支持自定义位置和大小
    # 从seaborn热力图获取映射器用于创建colorbar
    mappable = im.get_children()[0]  # 获取seaborn热力图的mappable对象
    
    if use_custom_colorbar:
        # 方法：使用自定义轴位置和大小
        # colorbar_bbox格式: [left, bottom, width, height] (相对于figure的比例)
        cax = fig.add_axes(colorbar_bbox)  # [x, y, width, height]
        cbar = plt.colorbar(mappable, cax=cax)
    else:
        # 传统方法：使用fraction参数
        cbar = plt.colorbar(mappable, ax=ax, shrink=colorbar_shrink, fraction=colorbar_width, pad=colorbar_pad)
    
    # # 去掉colorbar的边框
    # cbar.outline.set_visible(False)
    
    cbar.ax.tick_params(labelsize=colorbar_tick_fontsize)
    
    # 设置指定数量的刻度
    tick_values = np.linspace(vmin, vmax, colorbar_tick_count)
    cbar.set_ticks(tick_values)
    
    # 计算共同的指数（基于最大绝对值）
    max_abs_value = max(abs(vmin), abs(vmax))
    if max_abs_value == 0:
        common_exponent = 0
    else:
        common_exponent = int(np.floor(np.log10(max_abs_value)))
    
    # 设置科学计数法格式，保留一位小数
    def format_func(x, pos):
        if x == 0:
            return '0.0'
        # 将数值转换为基于共同指数的尾数
        mantissa = x / (10 ** common_exponent)
        return f'{mantissa:.1f}'
    
    formatter = ticker.FuncFormatter(format_func)
    cbar.ax.yaxis.set_major_formatter(formatter)
    
    # 设置指数部分在colorbar上方
    if max_abs_value != 0:
        cbar.ax.text(1.5, 1.04, f'×10$^{{{common_exponent}}}$', 
                    transform=cbar.ax.transAxes, 
                    horizontalalignment='center',
                    fontsize=colorbar_exponent_fontsize)
    
    # 设置标题
    ax.set_title("t'", fontsize=title_fontsize, pad=title_pad, style='italic')
    
    # 设置坐标轴标签
    if is_t_prime_p_prime_reshaped:
        # K(t',p')重塑后的图像：y轴是s'的重塑维度，x轴是t'拼接的维度
        # 将x轴标签放在第一个10格的中间位置
        height, width = matrix.shape
        ax.text(5, -1, "$x$", fontsize=label_fontsize,
                ha='center', va='bottom', style='italic')
        ax.set_ylabel("$y$", fontsize=label_fontsize, labelpad=label_pad, 
                      style='italic', rotation=0, ha='right', va='center')
        # 不使用默认的xlabel，因为我们用text手动放置了
    
    # # 在(-1,-1)位置添加斜体O
    # ax.text(-0.05, 1.02, '$O$', fontsize=label_fontsize, 
    #         style='italic', ha='center', va='center', transform=ax.transAxes)
    
    # 设置刻度
    if is_t_prime_p_prime_reshaped:
        # K(t',p')重塑图像的特殊刻度设置
        height, width = matrix.shape
        # 在每个10的边界处添加刻度，取消0.5偏移
        x_ticks = np.arange(0, width, 10)  # t'维度的边界，取消偏移
        ax.set_xticks(x_ticks)
        ax.set_yticks([])  # 移除y轴刻度
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(axis='x', top=True, bottom=False)
        ax.tick_params(axis='y', left=False, right=False)  # 完全移除y轴刻度
    elif is_tpp:
        # 为tpp图添加刻度但不显示数字
        # 设置主要刻度位置（每10个单位一个刻度），取消0.5偏移
        major_ticks = np.arange(0, matrix.shape[0]+10, 10)
        ax.set_xticks(major_ticks)
        ax.set_yticks([])  # 移除y轴刻度
        # 移除刻度标签（不显示数字）
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        # 设置刻度显示在右边界和下边界
        ax.tick_params(axis='x', top=True, bottom=False)  # x轴刻度在下边界
        ax.tick_params(axis='y', left=False, right=False)  # 完全移除y轴刻度
    else:
        # 移除tpt图的刻度标签
        ax.set_xticks([])
        ax.set_yticks([])  # 移除y轴刻度
    
    # 为K(t',p')重塑图像添加竖向白线分隔（每10格一根），取消0.5偏移
    if is_t_prime_p_prime_reshaped:
        height, width = matrix.shape
        # 每10列添加一根竖向白线来分隔不同的t'区域，取消偏移
        for x_pos in range(10, width, 10):
            ax.axvline(x=x_pos, color='white', linewidth=1, alpha=0.8)
    
    # 设置图像边距
    if margins is not None:
        plt.subplots_adjust(
            left=margins.get('left', 0.05),
            bottom=margins.get('bottom', 0.1),
            right=margins.get('right', 0.95),
            top=margins.get('top', 0.9),
            wspace=margins.get('wspace', 0.0),
            hspace=margins.get('hspace', 0.0)
        )
    else:
        # 使用tight_layout作为默认
        plt.tight_layout()
    
    # 添加文字标注
    if text_annotations is not None:
        for annotation in text_annotations:
            if len(annotation) >= 3:
                x, y, text = annotation[0], annotation[1], annotation[2]
                # 如果有第四个元素，作为额外的kwargs
                kwargs = annotation[3] if len(annotation) > 3 else {}
                
                # 设置默认的文字样式
                default_kwargs = {
                    'fontsize': label_fontsize,
                    'ha': 'center',
                    'va': 'bottom',
                    'style': 'italic',
                    'color': 'black'
                }
                default_kwargs.update(kwargs)
                
                ax.text(x, y, text, **default_kwargs)
    
    # 保存为SVG格式
    plt.savefig(save_path, format='svg', bbox_inches='tight', 
                facecolor='white', edgecolor='none',
                pad_inches=0.1)
    plt.close(fig)  # 立即关闭图形释放内存

def load_K_tensor_data(data_dir):
    """
    加载K张量主要数据和四个项数据
    
    Args:
        data_dir: 数据文件夹路径
        
    Returns:
        dict: 数据字典 {dataset_id: {sample_id: {layer_idx: {key: matrix}}}}
    """
    data_dir = Path(data_dir)
    all_data = {}
    
    for dataset_dir in data_dir.iterdir():
        if not dataset_dir.is_dir() or not dataset_dir.name.startswith('dataset_'):
            continue
        
        dataset_id = dataset_dir.name.split('_')[1]
        all_data[dataset_id] = {}
        
        for file_path in dataset_dir.glob('*_K_simplified.npz'):
            # 从文件名提取样本ID：sample_XXX_K_simplified.npz
            sample_id = file_path.stem.split('_')[1]
            
            try:
                data = np.load(file_path)
                sample_data = {}
                
                # 处理每一层的数据
                for layer_idx in range(6):
                    layer_data = {}
                    
                    # 加载主要K张量数据 - 优先加载新格式K_t_prime_p_prime
                    k_tp_key = f'layer_{layer_idx}_K_t_prime_p_prime'
                    if k_tp_key in data:
                        layer_data['K_t_prime_p_prime'] = data[k_tp_key]
                    else:
                        # 兼容旧格式
                        for tensor_name in ['K_tp_p_prime', 'K_tp_t_prime']:
                            key = f'layer_{layer_idx}_{tensor_name}'
                            if key in data:
                                layer_data[tensor_name] = data[key]
                    
                    # 加载K(t',p')的四个项
                    for term_name in ['term1_t_p', 'term2_t_p', 'term3_t_p', 'term4_t_p']:
                        key = f'layer_{layer_idx}_{term_name}'
                        if key in data:
                            layer_data[term_name] = data[key]
                    
                    # 加载term1和term2的和
                    term12_sum_key = f'layer_{layer_idx}_term12_sum_t_p'
                    if term12_sum_key in data:
                        layer_data['term12_sum_t_p'] = data[term12_sum_key]
                    
                    # 兼容旧格式的四个项
                    for term_name in ['term1_tp_p', 'term2_tp_p', 'term3_tp_p', 'term4_tp_p']:
                        key = f'layer_{layer_idx}_{term_name}'
                        if key in data:
                            layer_data[term_name] = data[key]
                    
                    # 兼容旧格式的K(t,p,t')四个项
                    for term_name in ['term1_tp_t', 'term2_tp_t', 'term3_tp_t', 'term4_tp_t']:
                        key = f'layer_{layer_idx}_{term_name}'
                        if key in data:
                            layer_data[term_name] = data[key]
                    
                    if layer_data:
                        sample_data[layer_idx] = layer_data
                
                if sample_data:
                    all_data[dataset_id][sample_id] = sample_data
                    
            except Exception as e:
                print(f"加载文件 {file_path} 时出错: {e}")
    
    return all_data

def parse_range_string(range_str, max_val):
    """
    解析范围字符串，支持格式：
    - None       -> [] (空列表，表示不处理)
    - "None"     -> [] (空列表，表示不处理)
    - "0"        -> [0]
    - "0,1,2"    -> [0,1,2]
    - "0-5"      -> [0,1,2,3,4,5]
    - "0-5,8"    -> [0,1,2,3,4,5,8]
    - "all"      -> list(range(max_val))
    
    Args:
        range_str: 范围字符串或None
        max_val: 最大值
        
    Returns:
        list: 解析后的索引列表，如果range_str为None或"None"则返回空列表
    """
    # 处理None的情况
    if range_str is None or range_str.lower() == 'none':
        return []
    
    if range_str.lower() == 'all':
        return list(range(max_val))
    
    result = []
    parts = range_str.split(',')
    
    for part in parts:
        part = part.strip()
        if '-' in part:
            start, end = map(int, part.split('-'))
            result.extend(range(start, end + 1))
        else:
            result.append(int(part))
    
    # 过滤掉超出范围的值
    result = [x for x in result if 0 <= x < max_val]
    return sorted(list(set(result)))

def parse_text_annotations(annotations_str):
    """
    解析文字标注字符串
    
    Args:
        annotations_str: 标注字符串，格式: "x1,y1,text1;x2,y2,text2" 或 None
        
    Returns:
        list: 解析后的标注列表，格式: [(x, y, text), ...] 或 None
    """
    if annotations_str is None or annotations_str.lower() == 'none':
        return None
    
    annotations = []
    try:
        # 分割多个标注
        parts = annotations_str.split(';')
        for part in parts:
            part = part.strip()
            if not part:
                continue
            
            # 分割x,y,text
            components = part.split(',')
            if len(components) >= 3:
                x = float(components[0].strip())
                y = float(components[1].strip())
                text = ','.join(components[2:]).strip()  # 处理文字中可能包含逗号的情况
                annotations.append((x, y, text))
    except Exception as e:
        print(f"警告：解析文字标注时出错: {e}")
        return None
    
    return annotations if annotations else None

def extract_suffix_from_path(path):
    """
    从路径中提取后缀
    
    Args:
        path: 路径字符串
        
    Returns:
        str: 提取的后缀，如果没有后缀则返回空字符串
    """
    path_obj = Path(path)
    # 获取路径的最后一部分
    last_part = path_obj.name
    
    # 检查是否有_后缀格式
    if '_' in last_part:
        # 提取最后一个下划线后的部分作为后缀
        suffix = last_part.split('_')[-1]
        return suffix
    
    return "None"

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_arguments()
    
    # 从输入路径中提取后缀
    input_suffix = extract_suffix_from_path(args.data_dir)
    
    # 自动设置输出路径（如果使用默认值且检测到输入路径有后缀）
    if args.output_dir == 'process5_4/K_simplified/random/plots' and input_suffix:
        args.output_dir = f'process5_4/K_simplified/random/plots_{input_suffix}'
    
    # 加载YAML配置
    config_path = Path(args.config)
    if not config_path.is_absolute():
        # 如果是相对路径，则相对于脚本所在目录
        script_dir = Path(__file__).parent
        config_path = script_dir / config_path
    
    config = load_config(config_path)
    
    # 设置默认配置
    default_config = {
        'figure': {
            'figsize': [5, 4],
            'dpi': 150,
            'cmap': 'Reds'
        },
        'font': {
            'title_fontsize': 28,
            'label_fontsize': 31,
            'colorbar_tick_fontsize': 31,
            'colorbar_exponent_fontsize': 31
        },
        'spacing': {
            'title_pad': 10,
            'label_pad': 5
        },
        'margins': {
            'left': 0.05,
            'bottom': 0.1,
            'right': 0.95,
            'top': 0.9,
            'wspace': 0.0,
            'hspace': 0.0
        },
        'colorbar': {
            'tick_count': 5,
            'shrink': 1.0,
            'width': 0.05,
            'pad': 0.05,
            'method': 'fraction',
            'position': 'right',
            'bbox': [0.92, 0.1, 0.02, 0.8]
        },
        'processing': {
            'add_grid': False,
            'grid_interval': 10
        },
        'plotting': {
            'plot_main_k': True,
            'plot_k_terms': False
        }
    }
    
    # 合并配置
    if config is None:
        config = default_config
    else:
        # 递归合并配置，确保所有必需的键都存在
        def merge_configs(default, user):
            result = default.copy()
            for key, value in user.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = merge_configs(result[key], value)
                else:
                    result[key] = value
            return result
        
        config = merge_configs(default_config, config)
    
    # 从命令行参数覆盖绘制选项
    config['plotting']['plot_main_k'] = args.plot_main
    config['plotting']['plot_k_terms'] = args.plot_terms
    
    # 解析文字标注
    text_annotations = parse_text_annotations(args.text_annotations)
    
    print("K张量热力图绘制脚本")
    print("=" * 60)
    print("📋 当前配置:")
    print(f"  数据目录: {args.data_dir}")
    print(f"  图片输出目录: {args.output_dir}")
    print(f"  配置文件: {config_path}")
    if input_suffix:
        print(f"  检测到路径后缀: {input_suffix}")
    print(f"  数据集: {args.datasets}")
    print(f"  样本: {args.samples}")
    print(f"  层: {args.layers}")
    print(f"  时间步: {args.time_steps}")
    print(f"  空间步: {args.space_steps}")
    print(f"  绘制主要K张量: {config['plotting']['plot_main_k']}")
    print(f"  绘制K张量四个项: {config['plotting']['plot_k_terms']}")
    print(f"  图片尺寸: {tuple(config['figure']['figsize'])}")
    print(f"  分辨率: {config['figure']['dpi']}")
    print(f"  颜色映射: {config['figure']['cmap']}")
    print(f"  添加网格: {config['processing']['add_grid']}")
    if text_annotations:
        print(f"  文字标注: {len(text_annotations)} 个")
        for i, annotation in enumerate(text_annotations):
            print(f"    {i+1}. 位置({annotation[0]}, {annotation[1]}): '{annotation[2]}'")
    else:
        print(f"  文字标注: 无")
    print()
    print("💡 要修改图片样式配置，请编辑 YAML 配置文件")
    print("💡 要修改数据选择配置，请使用命令行参数")
    print("=" * 60)
    print()
    
    # =============================================================================
    # 参数配置（从config字典和args获取）
    # =============================================================================
    
    # 路径配置
    DATA_DIR = args.data_dir
    OUTPUT_DIR = args.output_dir
    
    # 图片配置
    FIGSIZE = tuple(config['figure']['figsize'])
    DPI = config['figure']['dpi']
    CMAP = config['figure']['cmap']
    
    # 字体配置
    TITLE_FONTSIZE = config['font']['title_fontsize']
    COLORBAR_TICK_FONTSIZE = config['font']['colorbar_tick_fontsize']
    LABEL_FONTSIZE = config['font']['label_fontsize']
    COLORBAR_EXPONENT_FONTSIZE = config['font']['colorbar_exponent_fontsize']
    COLORBAR_TICK_COUNT = config['colorbar']['tick_count']
    
    # 间距配置
    TITLE_PAD = config['spacing']['title_pad']
    LABEL_PAD = config['spacing']['label_pad']
    
    # 边距配置
    MARGINS = config.get('margins', None)
    
    # Colorbar配置
    COLORBAR_SHRINK = config['colorbar']['shrink']
    COLORBAR_WIDTH = config['colorbar']['width']
    COLORBAR_PAD = config['colorbar']['pad']
    COLORBAR_METHOD = config['colorbar'].get('method', 'fraction')
    COLORBAR_POSITION = config['colorbar'].get('position', 'right')
    COLORBAR_BBOX = config['colorbar'].get('bbox', [0.92, 0.1, 0.02, 0.8])
    USE_CUSTOM_COLORBAR = (COLORBAR_METHOD == 'custom')
    
    # 网格配置
    ADD_GRID = config['processing']['add_grid']
    GRID_INTERVAL = config['processing']['grid_interval']
    
    # 处理配置
    PLOT_MAIN_K = config['plotting']['plot_main_k']
    PLOT_K_TERMS = config['plotting']['plot_k_terms']
    
    # 创建输出文件夹
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载数据
    print("加载K张量数据...")
    all_data = load_K_tensor_data(DATA_DIR)
    
    if not all_data:
        print("没有找到任何数据文件！")
        return
    
    print(f"找到数据集: {list(all_data.keys())}")
    print()
    
    # 解析选择范围
    available_datasets = sorted([int(k) for k in all_data.keys()])
    selected_datasets = parse_range_string(args.datasets, max(available_datasets) + 1)
    selected_datasets = [str(d) for d in selected_datasets if str(d) in all_data]
    
    print(f"将处理数据集: {selected_datasets}")
    
    total_plots = 0
    
    # K张量主要数据的名称
    main_k_info = {
        'K_t_prime_p_prime': 'K(t\',p\')'
    }
    
    # K(t',p')四个项的名称和描述
    tp_term_info = {
        'term1_t_p': 'Term1 (A_TT * W_sigma_T * A_PP * W_dd1 * W_sigma_S)',
        'term2_t_p': 'Term2 (I_TT * W_sigma_T * A_PP * W_dd2 * W_sigma_S)',
        'term3_t_p': 'Term3 (I_PP * A_TT * W_dd3 * W_sigma_S * W_sigma_T)',
        'term4_t_p': 'Term4 (I_TT * I_PP * W_dd4 * W_sigma_S * W_sigma_T)',
        'term12_sum_t_p': 'Term1+Term2 Sum (Term1 + Term2)'
    }
    
    # 处理每个数据集
    for dataset_id in selected_datasets:
        print(f"处理数据集 {dataset_id}...")
        
        # 创建数据集输出文件夹
        dataset_output_dir = output_dir / f'dataset_{dataset_id}'
        dataset_output_dir.mkdir(parents=True, exist_ok=True)
        
        dataset_data = all_data[dataset_id]
        
        # 解析样本范围
        available_samples = sorted([int(k) for k in dataset_data.keys()])
        if available_samples:
            selected_samples = parse_range_string(args.samples, max(available_samples) + 1)
            selected_samples = [str(s) for s in selected_samples if str(s) in dataset_data]
        else:
            selected_samples = []
        
        print(f"  将处理样本: {selected_samples}")
        
        for sample_id in tqdm(selected_samples, desc=f"处理数据集 {dataset_id} 的样本"):
            # 创建样本输出文件夹
            sample_output_dir = dataset_output_dir / f'sample_{sample_id}'
            sample_output_dir.mkdir(parents=True, exist_ok=True)
            
            sample_data = dataset_data[sample_id]
            
            # 解析层范围
            available_layers = sorted(sample_data.keys())
            if available_layers:
                selected_layers = parse_range_string(args.layers, max(available_layers) + 1)
                selected_layers = [l for l in selected_layers if l in available_layers]
            else:
                selected_layers = []
            
                            # 处理每一层
            for layer_idx in selected_layers:
                layer_data = sample_data[layer_idx]
                
                # 创建层输出文件夹
                layer_output_dir = sample_output_dir / f'layer_{layer_idx}'
                layer_output_dir.mkdir(parents=True, exist_ok=True)
                
                # ================= 处理主要K张量 =================
                if PLOT_MAIN_K:
                    main_output_dir = None  # 延迟创建文件夹
                    
                    # 处理 K_t_prime_p_prime: (t', p') -> 直接画一张图，带s'重塑和t'拼接
                    if 'K_t_prime_p_prime' in layer_data:
                        K_t_prime_p_prime = layer_data['K_t_prime_p_prime']
                        
                        if K_t_prime_p_prime.shape[0] > 0 and K_t_prime_p_prime.shape[1] > 0:
                            # 创建文件夹（只在需要时创建）
                            if main_output_dir is None:
                                main_output_dir = layer_output_dir / 'main_K'
                                main_output_dir.mkdir(parents=True, exist_ok=True)
                            
                            try:
                                # 使用新的重塑函数：将s'重塑为10x10，按t'拼接
                                reshaped_matrix = reshape_t_prime_p_prime_for_visualization(K_t_prime_p_prime)
                                
                                # 生成标题和保存路径
                                title = f'K($t\'$,$s\'$) reshaped'
                                save_path = main_output_dir / f'K_t_prime_p_prime_reshaped.svg'
                                
                                # 绘制热力图 - 使用新的标志
                                plot_heatmap_fast(reshaped_matrix, title, save_path,
                                                FIGSIZE, DPI, CMAP,
                                                TITLE_FONTSIZE, COLORBAR_TICK_FONTSIZE,
                                                LABEL_FONTSIZE, COLORBAR_EXPONENT_FONTSIZE,
                                                TITLE_PAD, LABEL_PAD,
                                                ADD_GRID, GRID_INTERVAL, COLORBAR_TICK_COUNT,
                                                COLORBAR_SHRINK, COLORBAR_WIDTH, COLORBAR_PAD,
                                                COLORBAR_POSITION, COLORBAR_BBOX, USE_CUSTOM_COLORBAR,
                                                MARGINS,
                                                is_tpp=False, is_t_prime_p_prime_reshaped=True,
                                                text_annotations=text_annotations)
                                
                                total_plots += 1
                                
                            except Exception as e:
                                print(f"  警告：处理K_t_prime_p_prime时出错: {e}")
                
                # ================= 处理 K(t',p') 四个项 =================
                if PLOT_K_TERMS:
                    tp_output_dir = None  # 延迟创建文件夹
                    
                    for term_name, term_description in tp_term_info.items():
                        if term_name not in layer_data:
                            continue
                        
                        term_matrix = layer_data[term_name]  # 形状: (t', p')
                        
                        if term_matrix.shape[0] == 0 or term_matrix.shape[1] == 0:
                            continue
                        
                        # 创建文件夹（只在需要时创建）
                        if tp_output_dir is None:
                            tp_output_dir = layer_output_dir / 'K_tp_terms'
                            tp_output_dir.mkdir(parents=True, exist_ok=True)
                        
                        try:
                            # 使用新的重塑函数：将s'重塑为10x10，按t'拼接
                            reshaped_matrix = reshape_t_prime_p_prime_for_visualization(term_matrix)
                            
                            # 生成标题和保存路径
                            term_num = term_name.split('_')[0]  # 提取term1, term2等
                            title = f'K($t\'$,$s\'$) {term_num} reshaped'
                            save_path = tp_output_dir / f'{term_name}_reshaped.svg'
                            
                            # 特殊处理term12_sum: 使用最大值的60%作为颜色上限
                            vmax_ratio = 1 if term_name == 'term12_sum_t_p' else None
                            
                            # 绘制热力图 - 使用新的标志
                            plot_heatmap_fast(reshaped_matrix, title, save_path,
                                            FIGSIZE, DPI, CMAP,
                                            TITLE_FONTSIZE, COLORBAR_TICK_FONTSIZE,
                                            LABEL_FONTSIZE, COLORBAR_EXPONENT_FONTSIZE,
                                            TITLE_PAD, LABEL_PAD,
                                            ADD_GRID, GRID_INTERVAL, COLORBAR_TICK_COUNT,
                                            COLORBAR_SHRINK, COLORBAR_WIDTH, COLORBAR_PAD,
                                            COLORBAR_POSITION, COLORBAR_BBOX, USE_CUSTOM_COLORBAR,
                                            MARGINS,
                                            is_tpp=False, is_t_prime_p_prime_reshaped=True,
                                            vmax_ratio=vmax_ratio, text_annotations=text_annotations)
                            
                            total_plots += 1
                            
                        except Exception as e:
                            print(f"  警告：处理{term_name}时出错: {e}")
    
    print(f"\n完成！共生成 {total_plots} 张热力图")
    print(f"图片已保存到 {output_dir}")

if __name__ == '__main__':
    main()
