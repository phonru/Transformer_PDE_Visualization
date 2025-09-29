import os
import glob
import argparse
import numpy as np
import sys
# 添加项目根目录到 sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from src.utils.plotting_utils import load_config, plot_from_config

def parse_ids(ids_str):
    """解析数据集或样本ID字符串"""
    if ids_str is None:
        return None
    if ids_str.lower() == 'all':
        return 'all'
    # 支持逗号分隔的格式，如 '1,2,3'
    if ',' in ids_str:
        return [id_str.strip() for id_str in ids_str.split(',')]
    # 支持空格分隔的格式
    return ids_str.split()

def filter_datasets(dataset_dirs, dataset_ids):
    """根据指定的数据集ID过滤数据集目录"""
    if dataset_ids is None or dataset_ids == 'all':
        return dataset_dirs
    
    filtered_dirs = []
    for dataset_dir in dataset_dirs:
        dataset_name = os.path.basename(dataset_dir)
        dataset_id = dataset_name.split('_')[1]
        if dataset_id in dataset_ids:
            filtered_dirs.append(dataset_dir)
    
    return filtered_dirs

def filter_samples(npz_files, sample_ids):
    """根据指定的样本ID过滤样本文件"""
    if sample_ids is None or sample_ids == 'all':
        return npz_files
    
    filtered_files = []
    for npz_file in npz_files:
        sample_name = os.path.basename(npz_file).replace('_mean.npz', '')
        sample_id = sample_name.split('_')[1]
        if sample_id in sample_ids:
            filtered_files.append(npz_file)
    
    return filtered_files

def plot_at_matrices(data_dir, output_dir, config, dataset_ids=None, sample_ids=None, use_head_mean=False):
    """
    绘制时间注意力矩阵热力图
    
    Args:
        data_dir: 数据目录路径
        output_dir: 输出目录路径
        dataset_ids: 要处理的数据集ID列表，None表示处理所有
        sample_ids: 要处理的样本ID列表，None表示处理所有
        use_head_mean: 是否使用头平均数据（3D，第一维为空间patch）而不是时空平均数据（2D）
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有数据集目录
    dataset_dirs = glob.glob(os.path.join(data_dir, 'dataset_*'))
    dataset_dirs.sort()  # 确保目录顺序一致
    
    # 根据指定的数据集ID过滤
    dataset_dirs = filter_datasets(dataset_dirs, dataset_ids)
    
    print(f"找到 {len(dataset_dirs)} 个数据集")
    
    # 为每个数据集处理
    for dataset_dir in dataset_dirs:
        dataset_name = os.path.basename(dataset_dir)
        print(f"处理数据集: {dataset_name}")
        
        # 获取该数据集下的所有npz文件
        npz_files = glob.glob(os.path.join(dataset_dir, '*.npz'))
        npz_files.sort()  # 确保文件顺序一致
        
        # 根据指定的样本ID过滤
        npz_files = filter_samples(npz_files, sample_ids)
        
        # 首先检查有多少层（检查第一个文件）
        if not npz_files:
            print(f"  跳过：没有找到符合条件的样本文件")
            continue
            
        sample_data = np.load(npz_files[0])
        layers = []
        
        # 根据是否使用头平均选择不同的键模式
        key_suffix = '_temporal_head_mean' if use_head_mean else '_temporal_mean'
        
        for key in sample_data.keys():
            if key.endswith(key_suffix):
                layer_num = key.split('_')[1]
                layers.append(int(layer_num))
        layers = sorted(set(layers))
        
        print(f"  处理 {len(npz_files)} 个样本，{len(layers)} 层")
        
        # 为每一层的每个样本单独绘制时间注意力矩阵
        for npz_file in npz_files:
            # 从文件名提取样本编号
            sample_name = os.path.basename(npz_file).replace('_mean.npz', '')
            
            for layer in layers:
                attn_key = f'attn_{layer}_temporal_head_mean' if use_head_mean else f'attn_{layer}_temporal_mean'
                
                data = np.load(npz_file)
                if attn_key in data:
                    at_matrix = data[attn_key]
                    
                    # 处理不同维度的数据
                    if use_head_mean and at_matrix.ndim == 3:
                        # 头平均数据是3D的，第一维是空间patch，需要为每个patch绘制
                        patch_num = at_matrix.shape[0]
                        matrices_to_plot = [(at_matrix[p], f'_p{p}') for p in range(patch_num)]
                    else:
                        # 时空平均数据是2D的，直接绘制
                        matrices_to_plot = [(at_matrix, '')]
                    
                    # 为每个矩阵绘制图像
                    for matrix, patch_suffix in matrices_to_plot:
                        # 生成文件名及输出目录后再绘图
                        # 创建数据集、样本和层目录结构
                        dataset_output_dir = os.path.join(output_dir, dataset_name)
                        sample_dir = os.path.join(dataset_output_dir, sample_name)
                        layer_dir = os.path.join(sample_dir, f'layer_{layer}')
                        os.makedirs(layer_dir, exist_ok=True)
                        
                        # 生成文件名
                        data_type = 'head_mean' if use_head_mean else 'mean'
                        file_ext = config.get('output', {}).get('format', 'svg')
                        filename = f'attn_{layer}_temporal_{data_type}{patch_suffix}.{file_ext}'
                        output_path = os.path.join(layer_dir, filename)
                        # 标题可选: 若配置中已有标题需求，可在此传入
                        title_txt = None  # 如需: f"$A_t$ Layer {layer+1}"
                        plot_from_config(matrix, config, output_path, title_text=title_txt)
            
            print(f"    完成样本 {sample_name}")
def main():
    parser = argparse.ArgumentParser(description='绘制时间注意力 (A_t) 矩阵热力图 (基于配置驱动)')
    parser.add_argument('--data_dir', default='data/processed/A_mean',
                        help='输入数据目录路径，结构需包含 dataset_*/sample_*.npz')
    parser.add_argument('--output_dir', default='output/plots_At',
                        help='输出图像目录路径')
    parser.add_argument('--dataset_ids', type=str, default='all',
                        help='指定要处理的数据集ID，格式: all | 1,2,3 | 1 2 3')
    parser.add_argument('--sample_ids', type=str, default='all',
                        help='指定要处理的样本ID，格式: all | 0,1,2 | 0 1 2')
    parser.add_argument('--data_types', nargs='+', default=['both'],
                        choices=['mean', 'head_mean', 'both'],
                        help='数据类型: mean(时空平均) head_mean(头平均) both(两种')
    parser.add_argument('--config', type=str, default='config/plot_At_config.yaml',
                        help='绘图样式配置文件 (YAML) 路径')

    args = parser.parse_args()

    # 加载配置
    if not os.path.isfile(args.config):
        raise FileNotFoundError(f"未找到配置文件: {args.config}")
    config = load_config(args.config)

    # 解析数据集和样本ID
    dataset_ids = parse_ids(args.dataset_ids)
    sample_ids = parse_ids(args.sample_ids)

    # 处理数据类型参数
    if 'both' in args.data_types:
        data_types_to_process = ['mean', 'head_mean']
    else:
        data_types_to_process = [t for t in args.data_types if t in ['mean', 'head_mean']]

    print(f"数据目录: {args.data_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"配置文件: {args.config}")
    print(f"数据集ID: {dataset_ids}")
    print(f"样本ID: {sample_ids}")
    print(f"数据类型: {data_types_to_process}")
    print()

    # 为每种数据类型分别处理
    for data_type in data_types_to_process:
        use_head_mean = (data_type == 'head_mean')
        print(f"正在处理{'头平均' if use_head_mean else '时空平均'}数据...")
        plot_at_matrices(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            config=config,
            dataset_ids=dataset_ids,
            sample_ids=sample_ids,
            use_head_mean=use_head_mean
        )
        print()

    print("所有时间注意力热力图已生成完成！")

if __name__ == "__main__":
    main()
