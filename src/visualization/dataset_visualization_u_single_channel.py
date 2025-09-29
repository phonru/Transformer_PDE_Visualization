import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import argparse
import yaml
import matplotlib.colors as mcolors

# 设置字体
plt.rcParams['font.family'] = 'Arial'

def load_config(config_path):
    """
    从YAML文件加载配置
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"加载配置文件失败: {e}")
        return None

def parse_arguments():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='数据集可视化工具')
    
    parser.add_argument('--config', '-c', 
                        default='/home/pkq2/project/transformer_PDE_2/config/dataset_visualization_u_single_channel_config.yaml',
                        help='配置文件路径')
        
    parser.add_argument('--output-dir', '-o', type=str,
                        default='output/plots_datasets/u_selective_single_channel',
                        help='输出目录路径')
    
    parser.add_argument('--datasets', '-d', nargs='+', type=int, default=[9],
                        help='要可视化的数据集索引，例如: -d 0 1 2')
    
    parser.add_argument('--samples', '-s', nargs='+', type=int, default=[0,1,2,3,4,5],
                        help='要可视化的样本索引，例如: -s 100 200')
    
    parser.add_argument('--times', '-t', nargs='+', type=int, default=[0],
                        help='要可视化的时间步，支持"all"或数字列表，例如: -t 0 5 10')

    parser.add_argument('--channel', type=int, default=0,
                        help='指定要显示的通道索引 (0-based)，默认为0')
    
    parser.add_argument('--contour', action='store_true', default=False,
                        help='是否绘制等高线')
    
    return parser.parse_args()

def convert_config_format(yaml_config):
    """
    将YAML配置转换为原有的config格式
    """
    vis_config = yaml_config['visualization']
    
    config = {
        # 图像尺寸参数
        'FIGURE_WIDTH': vis_config['figure']['width'],
        'FIGURE_HEIGHT': vis_config['figure']['height'],
        
        # 字体大小参数
        'MAIN_TITLE_FONTSIZE': vis_config['fonts']['main_title'],
        'TIME_TITLE_FONTSIZE': vis_config['fonts']['time_title'],
        'COLORBAR_LABEL_FONTSIZE': vis_config['fonts']['colorbar_label'],
        
        # 坐标轴标签参数
        'X_LABELPAD': vis_config['axis_labels']['x_labelpad'],
        'Y_LABELPAD': vis_config['axis_labels']['y_labelpad'],
        'X_FONTSIZE': vis_config['axis_labels']['x_fontsize'],
        'Y_FONTSIZE': vis_config['axis_labels']['y_fontsize'],
        
        # 布局参数
        'SUBPLOT_LEFT': vis_config['layout']['subplot_left'],
        'SUBPLOT_RIGHT': vis_config['layout']['subplot_right'],
        'SUBPLOT_BOTTOM': vis_config['layout']['subplot_bottom'],
        'SUBPLOT_TOP': vis_config['layout']['subplot_top'],
        
        # 颜色条参数
        'COLORBAR_LEFT': vis_config['colorbar']['left'],
        'COLORBAR_BOTTOM': vis_config['colorbar']['bottom'],
        'COLORBAR_WIDTH': vis_config['colorbar']['width'],
        'COLORBAR_HEIGHT': vis_config['colorbar']['height'],
        'COLORBAR_TICKS_NUM': vis_config['colorbar']['ticks_num'],
        'COLORBAR_FORMAT': vis_config['colorbar']['format'],
        
        # 绘图参数
        'CONTOUR_LEVELS': vis_config['plot']['contour_levels'],
        'COLORMAP': vis_config['plot']['colormap'],
        'DPI': vis_config['figure']['dpi'],
        
        # 等高线参数（样式参数，是否绘制由命令行控制）
        'CONTOUR_LEVEL_VALUES': vis_config.get('contour', {}).get('levels', [0]),
        'CONTOUR_COLOR': vis_config.get('contour', {}).get('color', 'white'),
        'CONTOUR_LINEWIDTH': vis_config.get('contour', {}).get('linewidth', 1.5),
    }
    
    return config

def load_h5_data(file_path):
    """
    从H5文件中加载u_in数据
    """
    print(f"正在加载文件: {file_path}")
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

def visualize_velocity_field_selective(u_data, time_steps, sample_indices, dataset_name, dataset_index, output_dir, config, channel_index=0, draw_contour=False):
    """
    可视化速度场数据的指定时间步和样本（单通道模式）
    
    Args:
        u_data: 速度场数据，形状为 (samples, channels, time, height, width)
        time_steps: 要可视化的时间步列表
        sample_indices: 要可视化的样本索引列表
        dataset_name: 数据集名称
        dataset_index: 数据集索引（用于文件夹命名）
        output_dir: 输出目录
        config: 配置参数字典
        channel_index: 要显示的通道索引 (0-based)
        draw_contour: 是否绘制等高线
    """
    if u_data is None:
        print(f"跳过 {dataset_name}，数据为空")
        return
    
    print(f"\n--- 开始可视化 {dataset_name} 数据集 ---")
    print(f"速度场数据形状: {u_data.shape}")
    print(f"选择的时间步: {time_steps}")
    print(f"选择的样本: {sample_indices}")
    
    # 为当前数据集创建专门的文件夹，使用数字索引
    dataset_output_dir = os.path.join(output_dir, f'dataset_{dataset_index}')
    os.makedirs(dataset_output_dir, exist_ok=True)
    print(f"数据集输出目录: {dataset_output_dir}")
    
    # 获取数据维度信息
    n_samples, n_channels, n_time, height, width = u_data.shape
    
    # 获取要显示的通道索引
    target_channel = channel_index
    if target_channel >= n_channels:
        print(f"警告: 指定的通道索引 {target_channel} 超出范围 (0-{n_channels-1})，使用通道 0")
        target_channel = 0
    print(f"显示通道: {target_channel}")
    
    # 验证时间步范围
    valid_time_steps = [t for t in time_steps if 0 <= t < n_time]
    if len(valid_time_steps) != len(time_steps):
        print(f"警告: 一些时间步超出范围 (0-{n_time-1})，有效时间步: {valid_time_steps}")
    time_steps = valid_time_steps
    
    # 验证样本索引范围
    valid_sample_indices = [idx for idx in sample_indices if 0 <= idx < n_samples]
    if len(valid_sample_indices) != len(sample_indices):
        print(f"警告: 一些样本索引超出范围 (0-{n_samples-1})，有效样本索引: {valid_sample_indices}")
    sample_indices = valid_sample_indices
    
    if len(time_steps) == 0 or len(sample_indices) == 0:
        print(f"没有有效的时间步或样本索引可以可视化")
        return
    
    for sample_idx in sample_indices:
        print(f"  处理样本 {sample_idx}")
        
        # 为当前样本创建专门的文件夹
        sample_output_dir = os.path.join(dataset_output_dir, f'sample_{sample_idx}')
        os.makedirs(sample_output_dir, exist_ok=True)
        
        for time_step in time_steps:
            print(f"    处理时间步 {time_step}")
            
            # 获取当前样本当前时间步指定通道的数据
            data_slice = u_data[sample_idx, target_channel, time_step, :, :]
            sample_vmin = data_slice.min()
            sample_vmax = data_slice.max()
            
            # 创建单通道图像
            fig, ax = plt.subplots(1, 1, 
                                  figsize=(config['FIGURE_WIDTH'], 
                                          config['FIGURE_HEIGHT']))
            
            # fig.suptitle(f'{dataset_name} - Sample {sample_idx} - Time {time_step} - Channel {target_channel}', 
            #             fontsize=config['MAIN_TITLE_FONTSIZE'])
            
            # 绘制等高线图
            contour = ax.contourf(data_slice, levels=config['CONTOUR_LEVELS'], cmap=config['COLORMAP'], 
                                 vmin=sample_vmin, vmax=sample_vmax, extend='both')
            
            # 翻转y轴
            ax.invert_yaxis()
            
            # 添加轮廓线（如果启用）
            if draw_contour:
                try:
                    cs = ax.contour(data_slice, levels=config['CONTOUR_LEVEL_VALUES'], colors=[config['CONTOUR_COLOR']],
                                   linewidths=config['CONTOUR_LINEWIDTH'], alpha=0.8)
                    print(f"      绘制等高线: 水平 {config['CONTOUR_LEVEL_VALUES']}, 颜色 {config['CONTOUR_COLOR']}")
                except Exception as e:
                    print(f"      等高线绘制失败: {e}")
                    print(f"      数据范围: {data_slice.min():.4f} 到 {data_slice.max():.4f}")
            
            # 设置坐标轴标签
            ax.set_xlabel('x', fontsize=config['X_FONTSIZE'], ha='center', style='italic', labelpad=config['X_LABELPAD'])
            ax.set_ylabel('y', fontsize=config['Y_FONTSIZE'], va='center', style='italic', rotation=0, labelpad=config['Y_LABELPAD'])
            
            # 设置x轴标签位置在上边界
            ax.xaxis.set_label_position('top')
            ax.xaxis.tick_top()
            
            # 设置y轴标签位置在左边界
            ax.yaxis.set_label_position('left')
            
            # 隐藏刻度但保留标签
            ax.set_xticks([])
            ax.set_yticks([])
            
            ax.set_aspect('equal')
            
            # 调整布局
            fig.subplots_adjust(left=config['SUBPLOT_LEFT'], right=config['SUBPLOT_RIGHT'], 
                               bottom=config['SUBPLOT_BOTTOM'], top=config['SUBPLOT_TOP'])
            
            # 添加右侧的颜色条，使用连续的颜色映射
            cbar_ax = fig.add_axes([config['COLORBAR_LEFT'], config['COLORBAR_BOTTOM'], 
                                   config['COLORBAR_WIDTH'], config['COLORBAR_HEIGHT']])
            
            # 创建一个连续的颜色条而不是离散的
            norm = mcolors.Normalize(vmin=sample_vmin, vmax=sample_vmax)
            sm = plt.cm.ScalarMappable(cmap=config['COLORMAP'], norm=norm)
            sm.set_array([])
            
            cbar = fig.colorbar(sm, cax=cbar_ax, format=config['COLORBAR_FORMAT'])
            cbar.ax.tick_params(labelsize=config['COLORBAR_LABEL_FONTSIZE'])
            
            # 设置精确的刻度数量和位置
            tick_locations = np.linspace(sample_vmin, sample_vmax, config['COLORBAR_TICKS_NUM'])
            cbar.set_ticks(tick_locations)
            
            # 保存图像到样本专门的文件夹
            save_path = os.path.join(sample_output_dir, 
                                   f'{dataset_name}_sample_{sample_idx}_time_{time_step}_ch{target_channel}.svg')
            plt.savefig(save_path, format='svg', bbox_inches='tight', dpi=config['DPI'])
            print(f"      图像已保存: {save_path}")
            plt.close(fig)



def main():
    """主函数"""
    # 解析命令行参数
    args = parse_arguments()
    
    # 获取配置文件的绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, args.config)
    
    # 加载配置
    yaml_config = load_config(config_path)
    if yaml_config is None:
        print("配置文件加载失败，退出程序")
        return
    
    # 转换配置格式
    config = convert_config_format(yaml_config)
    
    # 获取通道索引和轮廓线参数
    channel_index = args.channel
    draw_contour = args.contour
    
    print(f"使用通道索引: {channel_index}")
    print(f"绘制等高线: {draw_contour}")
    
    # 获取数据集信息
    dataset_info = yaml_config['datasets']
    
    # 使用命令行参数
    selected_datasets = args.datasets
    selected_samples = args.samples
    selected_times = args.times
    
    # 设置输出目录
    output_dir = args.output_dir
    
    print("=== 数据集可视化配置 ===")
    print(f"配置文件: {config_path}")
    print(f"选择的数据集: {[dataset_info[idx]['name'] for idx in selected_datasets]}")
    print(f"选择的样本: {selected_samples}")
    print(f"选择的时间步: {selected_times}")
    print(f"目标通道: {channel_index}")
    print(f"绘制等高线: {draw_contour}")
    if draw_contour:
        print(f"等高线参数: 水平={config['CONTOUR_LEVEL_VALUES']}, 颜色={config['CONTOUR_COLOR']}, 宽度={config['CONTOUR_LINEWIDTH']}")
    print(f"输出目录: {output_dir}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n=== 开始生成可视化 ===")
    
    # 处理选择的数据集
    for dataset_idx in selected_datasets:
        dataset_name = dataset_info[dataset_idx]['name']
        file_path = dataset_info[dataset_idx]['path']
        
        print(f"\n处理数据集: {dataset_name}")
        
        # 加载数据
        u_data = load_h5_data(file_path)
        
        if u_data is not None:
            # 确定时间步
            n_time = u_data.shape[2]
            if selected_times == "all":
                time_steps = list(range(n_time))
            else:
                time_steps = selected_times
            
            print(f"数据时间步总数: {n_time}")
            print(f"将可视化时间步: {time_steps}")
            
            # 可视化
            visualize_velocity_field_selective(
                u_data, time_steps, selected_samples, 
                dataset_name, dataset_idx, output_dir, config, channel_index, draw_contour
            )
        else:
            print(f"跳过数据集 {dataset_name}，数据加载失败")
    
    print(f"\n所有可视化任务已完成！输出目录: {output_dir}")

if __name__ == '__main__':
    main()
