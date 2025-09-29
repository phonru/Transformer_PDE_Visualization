import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

# 设置字体
plt.rcParams['font.family'] = 'Arial'

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



def visualize_velocity_field_single_time(u_data, time_step, indices, dataset_name, dataset_index, output_dir, config):
    """
    可视化速度场数据的单个时间步
    
    Args:
        u_data: 速度场数据，形状为 (samples, channels, time, height, width)
        time_step: 要可视化的时间步
        indices: 要可视化的样本索引列表
        dataset_name: 数据集名称
        dataset_index: 数据集索引（用于文件夹命名）
        output_dir: 输出目录
        config: 配置参数字典
    """
    if u_data is None:
        print(f"跳过 {dataset_name}，数据为空")
        return
    
    print(f"\n--- 开始可视化 {dataset_name} 数据集时间步 {time_step} ---")
    print(f"速度场数据形状: {u_data.shape}")
    
    # 为当前数据集创建专门的文件夹，使用数字索引
    dataset_output_dir = os.path.join(output_dir, f'dataset_{dataset_index}')
    os.makedirs(dataset_output_dir, exist_ok=True)
    print(f"数据集输出目录: {dataset_output_dir}")
    
    # 如果indices为None，则可视化所有样本
    if indices is None:
        indices = list(range(u_data.shape[0]))
    
    # 确保索引不超出范围
    valid_indices = [idx for idx in indices if idx < u_data.shape[0]]
    if len(valid_indices) != len(indices):
        print(f"警告: 一些索引超出范围，有效索引: {valid_indices}")
    indices = valid_indices
    
    if len(indices) == 0:
        print(f"没有有效的索引可以可视化")
        return
    
    # 获取数据维度信息
    n_samples, n_channels, n_time, height, width = u_data.shape
    
    # 检查时间步是否有效
    if time_step >= n_time:
        print(f"警告: 时间步 {time_step} 超出范围 (0-{n_time-1})")
        return
    
    for sample_idx in indices:
        print(f"  处理样本 {sample_idx}")
        
        # 为当前样本创建专门的文件夹
        sample_output_dir = os.path.join(dataset_output_dir, f'sample_{sample_idx}')
        os.makedirs(sample_output_dir, exist_ok=True)
        
        # 计算当前样本的颜色范围（每个样本独立的颜色映射）
        sample_data = u_data[sample_idx, :, time_step, :, :]
        sample_vmin = sample_data.min()
        sample_vmax = sample_data.max()
        
        # 创建子图：1行 x n_channels列
        fig, axes = plt.subplots(1, n_channels, 
                                figsize=(config['FIGURE_WIDTH_PER_TIME']*n_channels, 
                                        config['FIGURE_HEIGHT_PER_CHANNEL']))
        
        # 确保axes是一维数组
        if n_channels == 1:
            axes = [axes]
        
        fig.suptitle(f'{dataset_name} - Sample {sample_idx} - Time {time_step}', 
                    fontsize=config['MAIN_TITLE_FONTSIZE'])
        
        # 创建用于共享颜色条的隐藏图像
        hidden_ax = fig.add_axes([0, 0, 1, 1], visible=False)
        mappable = hidden_ax.imshow(np.zeros((height, width)), cmap=config['COLORMAP'], 
                                   vmin=sample_vmin, vmax=sample_vmax)
        
        for channel in range(n_channels):
            ax = axes[channel]
            
            # 获取指定时间步的数据
            data_slice = u_data[sample_idx, channel, time_step, :, :]
            
            # 绘制等高线图
            ax.contourf(data_slice, levels=config['CONTOUR_LEVELS'], cmap=config['COLORMAP'], 
                       vmin=sample_vmin, vmax=sample_vmax)
            
            # 翻转y轴
            ax.invert_yaxis()
            
            # 设置通道标题
            ax.set_title(f'Channel {channel}', fontsize=config['TIME_TITLE_FONTSIZE'])
            
            ax.set_aspect('equal')
            ax.axis('off')
        
        # 调整布局
        fig.subplots_adjust(left=config['SUBPLOT_LEFT'], right=config['SUBPLOT_RIGHT'], 
                           bottom=config['SUBPLOT_BOTTOM'], top=config['SUBPLOT_TOP'], 
                           wspace=config['SUBPLOT_WSPACE'])
        
        # 添加右侧的共享颜色条
        cbar_ax = fig.add_axes([config['COLORBAR_LEFT'], config['COLORBAR_BOTTOM'], 
                               config['COLORBAR_WIDTH'], config['COLORBAR_HEIGHT']])
        cbar = fig.colorbar(mappable, cax=cbar_ax, 
                          ticks=np.linspace(sample_vmin, sample_vmax, config['COLORBAR_TICKS_NUM']),
                          format=config['COLORBAR_FORMAT'])
        cbar.ax.tick_params(labelsize=config['COLORBAR_LABEL_FONTSIZE'])
        
        # 清理隐藏轴
        hidden_ax.remove()
        
        # 保存图像到样本专门的文件夹
        save_path = os.path.join(sample_output_dir, f'{dataset_name}_sample_{sample_idx}_time_{time_step}.svg')
        plt.savefig(save_path, format='svg', bbox_inches='tight', dpi=config['DPI'])
        print(f"    图像已保存: {save_path}")
        plt.close(fig)

def main():
    """主函数"""
    # ======================== 可视化超参数设置 ========================
    config = {
        # 图像尺寸参数
        'FIGURE_WIDTH_PER_TIME': 8,         # 每个通道的宽度
        'FIGURE_HEIGHT_PER_CHANNEL': 8,     # 每个通道的高度
        
        # 字体大小参数
        'MAIN_TITLE_FONTSIZE': 56,          # 主标题字体大小
        'TIME_TITLE_FONTSIZE': 40,          # 通道标题字体大小
        'COLORBAR_LABEL_FONTSIZE': 36,      # 颜色条标签字体大小
        
        # 布局参数
        'SUBPLOT_LEFT': 0.08,               # 子图左边距
        'SUBPLOT_RIGHT': 0.90,              # 子图右边距  
        'SUBPLOT_BOTTOM': 0.05,             # 子图底部边距
        'SUBPLOT_TOP': 0.80,                # 子图顶部边距
        'SUBPLOT_WSPACE': 0.01,             # 子图水平间距
        
        # 颜色条参数
        'COLORBAR_LEFT': 0.91,              # 颜色条左边距
        'COLORBAR_BOTTOM': 0.05,            # 颜色条底部边距
        'COLORBAR_WIDTH': 0.02,             # 颜色条宽度
        'COLORBAR_HEIGHT': 0.75,            # 颜色条高度
        'COLORBAR_TICKS_NUM': 5,            # 颜色条刻度数量
        'COLORBAR_FORMAT': '%.2f',          # 颜色条数值格式
        
        # 绘图参数
        'CONTOUR_LEVELS': 50,               # 等高线层数
        'COLORMAP': 'RdBu_r',               # 颜色映射
        'DPI': 300,                         # 图像DPI
    }
    # ================================================================
    
    # 定义文件路径和对应的索引
    file_paths = [
        '/data/1/pkq2/data_new/spec/test_pope_cut_new.h5',
        '/data/1/pkq2/data_new/cylinder/test_cut.h5',
        '/data/1/pkq2/data_new/channel/test_wins.h5',
        '/data/1/pkq2/data_new/s809/test_cut.h5',
        '/data/1/pkq2/data_new/vel/test_cut_rich.h5',

        '/data/1/pkq2/data_pkq/add_data/channel_flow_1000_test.h5',
        '/data/1/pkq2/data_pkq/add_data/fish_foil_oval2_all.h5',
        '/data/1/pkq2/data_pkq/add_data/test_160_cut.h5',
        '/data/1/pkq2/data_pkq/add_data/tree_foil_all.h5',

        '/data/1/pkq2/data4/cylinder_data1_train_time.h5', # 连续时间步圆柱绕流
    ]
    
    dataset_names = ['pope', 'cylinder', 'channel', 's809', 'vel', 'channel_2', 'fish', 'cylinder_2', 'tree', 'cylinder_time']
    
    # 配置每个数据集要可视化的样本索引，None表示可视化所有样本
    infer_indices = [
        # [0, 50, 100, 150, 200],   # pope
        # [0, 5, 10, 15],       # cylinder
        # [0, 15, 30, 45],      # channel
        # None,                 # s809 全部
        # [0, 5, 10, 15]        # vel
        [],[],[],[],[],[],[],[],[],None
        # None,
        # [2, 9, 16, 23, 30, 37],
        # [5, 15, 25, 35, 45, 55],
        # [5, 15, 25, 34]
    ]
    
    # 创建输出目录
    output_dir = 'output/plots_datasets/u_all_time'
    os.makedirs(output_dir, exist_ok=True)
    print(f"输出目录: {output_dir}")
    
    # 加载所有数据
    all_u_in_data = []
    for file_path in file_paths:
        u_in = load_h5_data(file_path)
        all_u_in_data.append(u_in)
    
    print("\n=== 生成每个时间步单独图像的可视化 ===")
    # 为每个数据集的每个时间步创建单独的可视化
    for i, (u_in_data, name, indices) in enumerate(zip(all_u_in_data, dataset_names, infer_indices)):
        if u_in_data is not None:
            n_time = u_in_data.shape[2]
            for t in range(n_time):
                visualize_velocity_field_single_time(u_in_data, t, indices, name, i, output_dir, config)
    
    print(f"\n所有可视化任务已完成！输出目录: {output_dir}")

if __name__ == '__main__':
    main()
