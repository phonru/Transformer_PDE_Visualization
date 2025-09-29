import numpy as np
import os
import torch
import argparse
from tqdm import tqdm

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="处理隐状态数据，计算Wdd和W_sigma权重")
    
    # 路径配置
    parser.add_argument('--attention-weights', type=str, 
                       default="data/weight/attn",
                       help='注意力权重路径 (默认: %(default)s)')
    
    parser.add_argument('--layer-norm-weights', type=str, 
                       default="data/weight/LN",
                       help='Layer Norm权重路径 (默认: %(default)s)')
    
    parser.add_argument('--hidden-states-dirs', type=str, nargs='+',
                       default=["data/hidden_states",],
                       help='隐状态目录列表 (默认: %(default)s)')
    
    parser.add_argument('--output-dir', type=str, 
                       default="data/processed/HS_p",
                       help='输出目录路径 (默认: %(default)s)')
    
    # 数据选择 - 这些配置会同时应用于：
    # 1. σ权重计算 (load_sigma_weights)
    # 2. At, As矩阵处理 (process_sample)
    # 3. 最终输出数据的选择
    parser.add_argument('--datasets', type=str, default='2,3,4,5,6,8',
                       help='要处理的数据集，支持格式: "all", "0", "0,1,2", "0-2" (默认: %(default)s)')
    
    parser.add_argument('--samples', type=str, default='all',
                       help='要处理的样本，支持格式: "all", "0", "0,1,2", "0-5" (默认: %(default)s)')
    
    return parser.parse_args()

# ==================== 默认配置 ====================
DEFAULT_CONFIG = {
    # 模型参数配置
    "num_layers": 6,
    "d_model": 640,
    "num_heads": 8,
}
# ================================================

def parse_range_string(range_str, available_items):
    """
    解析范围字符串，支持格式：
    - "all"      -> 所有可用项
    - "0"        -> [0]
    - "0,1,2"    -> [0,1,2]
    - "0-5"      -> [0,1,2,3,4,5]
    - "0-5,8"    -> [0,1,2,3,4,5,8]
    
    Args:
        range_str: 范围字符串
        available_items: 可用项列表
        
    Returns:
        list: 解析后的项列表
    """
    if range_str.lower() == 'all':
        return available_items
    
    result = []
    parts = range_str.split(',')
    
    for part in parts:
        part = part.strip()
        if '-' in part:
            start, end = map(int, part.split('-'))
            result.extend(range(start, end + 1))
        else:
            result.append(int(part))
    
    # 过滤掉不在可用项中的值
    result = [x for x in result if x in available_items]
    return sorted(list(set(result)))

def fuse_attention_weights(attn_path, d_model, num_heads):
    """融合单个注意力层的Wv和Wo权重"""
    Wv = np.load(f"{attn_path}/Wv_weight.npy")  # (640, 640)
    Wo = np.load(f"{attn_path}/Wo_weight.npy")  # (640, 640)
    
    # 分成8个头并融合
    d_k = d_model // num_heads  # 640 / 8 = 80
    W_sum = np.zeros((d_model, d_model))
    
    for head in range(num_heads):
        start_idx = head * d_k
        end_idx = (head + 1) * d_k
        # Wvi: (640, 80), Woi: (80, 640)
        Wvi = Wv[:, start_idx:end_idx]
        Woi = Wo[start_idx:end_idx, :]
        W_sum += Wvi @ Woi
    
    return W_sum

def load_attention_weights(attention_weights_path, num_layers, d_model, num_heads):
    """加载注意力权重并融合Wv和Wo"""
    base_path = attention_weights_path
    
    Wt_layers = []  # 时间注意力权重
    Ws_layers = []  # 空间注意力权重
    
    # 处理6层注意力
    for layer in range(num_layers):
        print(f"Processing layer {layer+1} attention weights...")
        
        # 时间注意力
        temp_path = f"{base_path}/layer{layer+1}_temporal_attn"
        Wt = fuse_attention_weights(temp_path, d_model, num_heads)
        Wt_layers.append(Wt)
        
        # 空间注意力
        spat_path = f"{base_path}/layer{layer+1}_spatial_attn" 
        Ws = fuse_attention_weights(spat_path, d_model, num_heads)
        Ws_layers.append(Ws)
    
    return Wt_layers, Ws_layers

def load_layer_norm_weights(layer_norm_weights_path, num_layers, d_model):
    """加载Layer Norm参数"""
    base_path = layer_norm_weights_path
    
    gamma_t_layers = []  # 时间层的缩放参数 (W_γ^T)
    gamma_s_layers = []  # 空间层的缩放参数 (W_γ^S)
    mu_t_layers = []     # 时间层的偏移参数 (W_μ^T)
    mu_s_layers = []     # 空间层的偏移参数 (W_μ^S)
    
    for layer in range(num_layers):
        print(f"Processing layer {layer} LN weights...")
        
        # gamma_t (W_γ^T) - sublayer_0 (时间注意力的LN参数)
        gamma_t_path = f"{base_path}/st_encoder_layers_{layer}_sublayer_0_norm_a_2.npy"
        gamma_t = np.load(gamma_t_path)  # (640,)
        gamma_t_diag = np.diag(gamma_t)  # (640, 640)
        gamma_t_layers.append(gamma_t_diag)
        
        # gamma_s (W_γ^S) - sublayer_1 (空间注意力的LN参数)
        gamma_s_path = f"{base_path}/st_encoder_layers_{layer}_sublayer_1_norm_a_2.npy"
        gamma_s = np.load(gamma_s_path)  # (640,)
        gamma_s_diag = np.diag(gamma_s)  # (640, 640)
        gamma_s_layers.append(gamma_s_diag)
        
        # mu_t (W_μ^T) - 全1/d矩阵，其中d=640
        mu_t_matrix = np.ones((d_model, d_model)) / d_model  # 全1/d矩阵
        mu_t_layers.append(mu_t_matrix)
        
        # mu_s (W_μ^S) - 全1/d矩阵，其中d=640
        mu_s_matrix = np.ones((d_model, d_model)) / d_model  # 全1/d矩阵
        mu_s_layers.append(mu_s_matrix)
    
    return gamma_t_layers, gamma_s_layers, mu_t_layers, mu_s_layers

def load_sigma_weights(hidden_states_dirs, num_layers, datasets_filter='all', samples_filter='all'):
    """计算W^σ^T和W^σ^S权重（注意力残差后输出的方差，类似LN中的方差）"""
    print("Computing sigma weights from attention residual outputs...")
    
    # 收集所有样本每一层的方差矩阵
    temporal_variances_by_layer = [[] for _ in range(num_layers)]  # 每层一个列表
    spatial_variances_by_layer = [[] for _ in range(num_layers)]   # 每层一个列表
    
    sample_count = 0
    # 从所有目录的所有数据集中收集样本
    for hidden_states_dir in hidden_states_dirs:
        if not os.path.exists(hidden_states_dir):
            print(f"Warning: {hidden_states_dir} not found, skipping...")
            continue
            
        # 获取该目录下的所有数据集
        dataset_dirs = [d for d in os.listdir(hidden_states_dir) 
                       if os.path.isdir(os.path.join(hidden_states_dir, d)) and d.startswith('dataset_')]
        
        # 应用数据集过滤
        available_dataset_ids = [int(d.split('_')[1]) for d in dataset_dirs]
        selected_dataset_ids = parse_range_string(datasets_filter, available_dataset_ids)
        selected_dataset_dirs = [f'dataset_{id}' for id in selected_dataset_ids 
                                if f'dataset_{id}' in dataset_dirs]
        
        print(f"Sigma calculation - Available datasets: {sorted(available_dataset_ids)}")
        print(f"Sigma calculation - Selected datasets: {sorted(selected_dataset_ids)}")
        
        for dataset_name in sorted(selected_dataset_dirs):
            dataset_path = os.path.join(hidden_states_dir, dataset_name)
            
            # 使用每个数据集的样本来计算方差
            sample_files = [f for f in os.listdir(dataset_path) if f.endswith('.npz')]
            sample_files = sorted(sample_files)
            
            # 应用样本过滤
            available_sample_ids = []
            for sample_file in sample_files:
                sample_name = sample_file.replace('.npz', '')
                if sample_name.startswith('sample_'):
                    try:
                        sample_id = int(sample_name.split('_')[1])
                        available_sample_ids.append(sample_id)
                    except:
                        pass
            
            available_sample_ids = sorted(list(set(available_sample_ids)))
            selected_sample_ids = parse_range_string(samples_filter, available_sample_ids)
            
            # 过滤样本文件
            selected_sample_files = []
            for sample_file in sample_files:
                sample_name = sample_file.replace('.npz', '')
                if sample_name.startswith('sample_'):
                    try:
                        sample_id = int(sample_name.split('_')[1])
                        if sample_id in selected_sample_ids:
                            selected_sample_files.append(sample_file)
                    except:
                        pass
            
            print(f"Using {len(selected_sample_files)} samples from {dataset_path} to compute sigma weights...")
            print(f"  Selected sample IDs: {sorted(selected_sample_ids)}")
            
            for sample_file in selected_sample_files:
                sample_path = os.path.join(dataset_path, sample_file)
                data = np.load(sample_path)
                sample_count += 1
                
                for layer in range(num_layers):
                    # 时间注意力 - 计算完整的残差连接后的隐状态
                    temporal_input = data[f'attn_{layer}_temporal/input']  # (p, t, d_model)
                    temporal_output = data[f'attn_{layer}_temporal/output']  # (p, t, d_model)
                    temporal_hidden = temporal_input + temporal_output  # 残差连接 (p, t, d_model)
                    
                    # 计算每个token的方差（沿着d_model维度）
                    # temporal_hidden: (p, t, d_model) -> var: (p, t)
                    temporal_var = np.var(temporal_hidden, axis=2)  # (p, t)
                    # 转置为 (t, p) 格式
                    temporal_var_tp = temporal_var.T  # (t, p)
                    temporal_variances_by_layer[layer].append(temporal_var_tp)
                    
                    # 空间注意力 - 计算完整的残差连接后的隐状态
                    spatial_input = data[f'attn_{layer}_spatial/input']  # (t, p, d_model)
                    spatial_output = data[f'attn_{layer}_spatial/output']  # (t, p, d_model)
                    spatial_hidden = spatial_input + spatial_output  # 残差连接 (t, p, d_model)
                    
                    # 计算每个token的方差（沿着d_model维度）
                    # spatial_hidden: (t, p, d_model) -> var: (t, p)
                    spatial_var = np.var(spatial_hidden, axis=2)  # (t, p)
                    spatial_variances_by_layer[layer].append(spatial_var)
    
    # 检查是否有样本数据
    if sample_count == 0:
        print("Warning: No samples found for sigma calculation, using default matrices")
        # 创建默认的(t,p)矩阵 - t=15, p=100
        W_sigma_t_layers = [np.ones((15, 100)) for _ in range(num_layers)]
        W_sigma_s_layers = [np.ones((15, 100)) for _ in range(num_layers)]
        return W_sigma_t_layers, W_sigma_s_layers
    
    # 计算每一层的平均方差矩阵
    W_sigma_t_layers = []
    W_sigma_s_layers = []
    
    for layer in range(num_layers):
        if temporal_variances_by_layer[layer] and spatial_variances_by_layer[layer]:
            # 对每一层的所有方差矩阵求平均
            W_sigma_t_layer = np.mean(temporal_variances_by_layer[layer], axis=0)  # (t, p)
            W_sigma_s_layer = np.mean(spatial_variances_by_layer[layer], axis=0)   # (t, p)
            
            # 避免方差为0的情况
            W_sigma_t_layer = np.maximum(W_sigma_t_layer, 1e-8)
            W_sigma_s_layer = np.maximum(W_sigma_s_layer, 1e-8)
            
            W_sigma_t_layers.append(W_sigma_t_layer)
            W_sigma_s_layers.append(W_sigma_s_layer)
            
            print(f"Layer {layer}: W_sigma_t shape = {W_sigma_t_layer.shape}, range = [{W_sigma_t_layer.min():.6f}, {W_sigma_t_layer.max():.6f}]")
            print(f"Layer {layer}: W_sigma_s shape = {W_sigma_s_layer.shape}, range = [{W_sigma_s_layer.min():.6f}, {W_sigma_s_layer.max():.6f}]")
        else:
            print(f"Warning: No residual data found for layer {layer}, using default matrices")
            # 创建默认的(t,p)矩阵 - t=15, p=100
            W_sigma_t_layers.append(np.ones((15, 100)))
            W_sigma_s_layers.append(np.ones((15, 100)))
    
    return W_sigma_t_layers, W_sigma_s_layers

def load_attention_matrices(sample_file, num_layers):
    """加载注意力概率矩阵，只保留头平均数据"""
    data = np.load(sample_file)
    
    At_layers = []  # 时间注意力概率矩阵
    As_layers = []  # 空间注意力概率矩阵
    
    for layer in range(num_layers):
        # 时间注意力 (p, heads, t, t) -> (p, t, t) 仅对heads维度求平均
        temporal_attn = data[f'attn_{layer}_temporal/attn']
        At = np.mean(temporal_attn, axis=1)  # 对heads维度（axis=1）求平均，结果: (p, t, t)
        At_layers.append(At)
        
        # 空间注意力 (t, heads, p, p) -> (t, p, p) 仅对heads维度求平均
        spatial_attn = data[f'attn_{layer}_spatial/attn']
        As = np.mean(spatial_attn, axis=1)  # 对heads维度（axis=1）求平均，结果: (t, p, p)
        As_layers.append(As)
    
    return At_layers, As_layers

def load_hidden_states(sample_file, num_layers):
    """加载隐状态矩阵"""
    data = np.load(sample_file)
    
    X_layers = []
    
    for layer in range(num_layers):
        # 时间注意力层的输入 (p, t, d_model)
        X = data[f'attn_{layer}_temporal/input']
        X_layers.append(X)
    
    return X_layers

def compute_intermediate_terms(Wv_t, Wv_s, gamma_t, gamma_s, mu_t, mu_s, d_model):
    """计算中间项 W_dd^1 到 W_dd^3 (移除 W_dd^4)
    
    根据新的公式：
    W_dd^1 = W_V^T * W_γ^T * W_V^S * W_γ^S * (I - W_μ^T) * (I - W_μ^S)
    W_dd^2 = W_V^S * W_γ^S * (I - W_μ^S)
    W_dd^3 = W_V^T * W_γ^T * (I - W_μ^T)
    """
    # 创建单位矩阵
    I = np.eye(d_model)
    
    # W_dd^1 = W_V^T * W_γ^T * W_V^S * W_γ^S * (I - W_μ^T) * (I - W_μ^S)
    Wdd1 = gamma_t @ Wv_t @ gamma_s @ Wv_s @ (I - mu_t) @ (I - mu_s)
    
    # W_dd^2 = W_V^S * W_γ^S * (I - W_μ^S)
    Wdd2 = gamma_s @ Wv_s @ (I - mu_s)
    
    # W_dd^3 = W_V^T * W_γ^T * (I - W_μ^T)
    Wdd3 = gamma_t @ Wv_t @ (I - mu_t)
    
    return Wdd1, Wdd2, Wdd3

def process_sample(sample_file, Wv_t_layers, Wv_s_layers, gamma_t_layers, gamma_s_layers, 
                  mu_t_layers, mu_s_layers, W_sigma_t_layers, W_sigma_s_layers, output_dir, 
                  num_layers, d_model):
    """处理单个样本，计算并保存Wdd和Wσ权重"""
    sample_name = os.path.basename(sample_file).replace('.npz', '')
    print(f"Processing {sample_name}...")
    
    # 加载注意力矩阵和隐状态
    At_layers, As_layers = load_attention_matrices(sample_file, num_layers)
    X_layers = load_hidden_states(sample_file, num_layers)
    
    # 处理每一层
    Wdd1_layers = []
    Wdd2_layers = []
    Wdd3_layers = []
    
    for layer in range(num_layers):
        print(f"  Processing layer {layer+1}...")
        
        # 获取当前层的参数
        Wv_t = Wv_t_layers[layer]
        Wv_s = Wv_s_layers[layer] 
        gamma_t = gamma_t_layers[layer]
        gamma_s = gamma_s_layers[layer]
        mu_t = mu_t_layers[layer]
        mu_s = mu_s_layers[layer]
        
        # 计算中间项 W_dd^1 到 W_dd^3 (移除 W_dd^4)
        Wdd1, Wdd2, Wdd3 = compute_intermediate_terms(
            Wv_t, Wv_s, gamma_t, gamma_s, mu_t, mu_s, d_model)
        
        # 保存中间项
        Wdd1_layers.append(Wdd1)
        Wdd2_layers.append(Wdd2)
        Wdd3_layers.append(Wdd3)
    
    # 保存Wdd权重和Wσ权重
    output_file = os.path.join(output_dir, f'{sample_name}_weights.npz')
    save_dict = {}
    
    # 保存Wdd权重（每层，移除 Wdd4）
    for i in range(num_layers):
        save_dict[f'layer_{i}_Wdd1'] = Wdd1_layers[i]
        save_dict[f'layer_{i}_Wdd2'] = Wdd2_layers[i] 
        save_dict[f'layer_{i}_Wdd3'] = Wdd3_layers[i]
        save_dict[f'layer_{i}_W_sigma_t'] = W_sigma_t_layers[i]
        save_dict[f'layer_{i}_W_sigma_s'] = W_sigma_s_layers[i]
    
    # 保存注意力矩阵和隐状态
    for i in range(num_layers):
        save_dict[f'layer_{i}_At'] = At_layers[i]
        save_dict[f'layer_{i}_As'] = As_layers[i] 
    
    np.savez(output_file, **save_dict)
    print(f"Saved weights to {output_file}")

def get_all_datasets_and_samples(hidden_states_dirs):
    """扫描所有隐状态目录，获取所有数据集和样本信息"""
    all_samples = []
    
    for hidden_states_dir in hidden_states_dirs:
        if not os.path.exists(hidden_states_dir):
            print(f"Warning: {hidden_states_dir} not found, skipping...")
            continue
            
        # 获取该目录下的所有数据集
        dataset_dirs = [d for d in os.listdir(hidden_states_dir) 
                       if os.path.isdir(os.path.join(hidden_states_dir, d)) and d.startswith('dataset_')]
        
        for dataset_name in sorted(dataset_dirs):
            dataset_path = os.path.join(hidden_states_dir, dataset_name)
            
            # 获取该数据集下的所有样本文件
            sample_files = [f for f in os.listdir(dataset_path) if f.endswith('.npz')]
            
            # 添加到总列表，保存完整路径和数据集信息
            for sample_file in sample_files:
                sample_path = os.path.join(dataset_path, sample_file)
                all_samples.append({
                    'sample_path': sample_path,
                    'sample_name': sample_file,
                    'dataset_name': dataset_name,
                    'source_dir': hidden_states_dir
                })
            
            print(f"Found {len(sample_files)} samples in {dataset_path}")
    
    # 按数据集名称和样本文件名排序
    all_samples.sort(key=lambda x: (x['dataset_name'], x['sample_name']))
    
    return all_samples

def main():
    """主函数 - 计算并保存Wdd和Wσ权重"""
    # 解析命令行参数
    args = parse_arguments()
    
    print("Starting weights calculation (Wdd and W_sigma)...")
    print(f"Configuration:")
    print(f"  Attention weights path: {args.attention_weights}")
    print(f"  Layer norm weights path: {args.layer_norm_weights}")
    print(f"  Hidden states directories: {args.hidden_states_dirs}")
    print(f"  Output base directory: {args.output_dir}")
    print(f"  Model layers: {DEFAULT_CONFIG['num_layers']}")
    print(f"  Model dimension: {DEFAULT_CONFIG['d_model']}")
    print(f"  Attention heads: {DEFAULT_CONFIG['num_heads']}")
    print(f"  Datasets: {args.datasets}")
    print(f"  Samples: {args.samples}")
    
    # 创建输出目录
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 加载Wv权重（时间和空间注意力的Value权重）
    print("\nStep 1: Loading and fusing attention weights...")
    Wv_t_layers, Wv_s_layers = load_attention_weights(
        args.attention_weights, DEFAULT_CONFIG['num_layers'], DEFAULT_CONFIG['d_model'], DEFAULT_CONFIG['num_heads'])
    
    # 2. 加载LN参数（包括γ和μ）
    print("\nStep 2: Loading Layer Norm weights...")
    gamma_t_layers, gamma_s_layers, mu_t_layers, mu_s_layers = load_layer_norm_weights(
        args.layer_norm_weights, DEFAULT_CONFIG['num_layers'], DEFAULT_CONFIG['d_model'])
    
    # 3. 加载σ权重（每层的方差矩阵）
    print("\nStep 3: Loading sigma weights...")
    W_sigma_t_layers, W_sigma_s_layers = load_sigma_weights(
        args.hidden_states_dirs, DEFAULT_CONFIG['num_layers'], args.datasets, args.samples)
    
    # 4. 获取所有数据集和样本信息
    print("\nStep 4: Scanning all datasets and samples...")
    all_samples = get_all_datasets_and_samples(args.hidden_states_dirs)
    
    if not all_samples:
        print("Error: No samples found in any hidden states directory!")
        return
    
    print(f"\nTotal samples found: {len(all_samples)}")
    
    # 5. 按数据集分组并应用过滤
    print("\nStep 5: Filtering datasets and samples...")
    
    # 按数据集分组
    datasets = {}
    for sample_info in all_samples:
        dataset_name = sample_info['dataset_name']
        if dataset_name not in datasets:
            datasets[dataset_name] = []
        datasets[dataset_name].append(sample_info)
    
    # 解析数据集过滤
    available_dataset_ids = [int(d.split('_')[1]) for d in datasets.keys()]
    selected_dataset_ids = parse_range_string(args.datasets, available_dataset_ids)
    selected_datasets = {f'dataset_{id}': datasets[f'dataset_{id}'] 
                        for id in selected_dataset_ids if f'dataset_{id}' in datasets}
    
    print(f"Available datasets: {sorted(available_dataset_ids)}")
    print(f"Selected datasets: {sorted(selected_dataset_ids)}")
    
    # 为每个数据集创建输出目录并处理样本
    for dataset_name, samples in selected_datasets.items():
        print(f"\nProcessing dataset: {dataset_name}")
        
        # 解析样本过滤
        available_sample_ids = []
        for sample_info in samples:
            sample_name = sample_info['sample_name'].replace('.npz', '')
            if sample_name.startswith('sample_'):
                try:
                    sample_id = int(sample_name.split('_')[1])
                    available_sample_ids.append(sample_id)
                except:
                    pass
        
        available_sample_ids = sorted(list(set(available_sample_ids)))
        selected_sample_ids = parse_range_string(args.samples, available_sample_ids)
        
        # 过滤样本
        selected_samples = []
        for sample_info in samples:
            sample_name = sample_info['sample_name'].replace('.npz', '')
            if sample_name.startswith('sample_'):
                try:
                    sample_id = int(sample_name.split('_')[1])
                    if sample_id in selected_sample_ids:
                        selected_samples.append(sample_info)
                except:
                    pass
        
        print(f"  Available samples: {len(available_sample_ids)} (IDs: {available_sample_ids})")
        print(f"  Selected samples: {len(selected_samples)} (IDs: {selected_sample_ids})")
        
        if not selected_samples:
            print(f"  No samples selected for {dataset_name}, skipping...")
            continue
        
        # 创建数据集输出目录
        dataset_output_dir = os.path.join(output_dir, dataset_name)
        os.makedirs(dataset_output_dir, exist_ok=True)
        
        # 处理该数据集的选中样本
        for sample_info in tqdm(selected_samples, desc=f"Processing {dataset_name}"):
            process_sample(sample_info['sample_path'], Wv_t_layers, Wv_s_layers, 
                          gamma_t_layers, gamma_s_layers, mu_t_layers, mu_s_layers, 
                          W_sigma_t_layers, W_sigma_s_layers, dataset_output_dir,
                          DEFAULT_CONFIG['num_layers'], DEFAULT_CONFIG['d_model'])
    
    print(f"\nWeights calculation completed!")
    print(f"Results saved to: {output_dir}")
    
    # 打印处理统计信息
    print("\nProcessing summary:")
    for dataset_name, samples in selected_datasets.items():
        # 重新计算选中的样本数量用于显示
        available_sample_ids = []
        for sample_info in samples:
            sample_name = sample_info['sample_name'].replace('.npz', '')
            if sample_name.startswith('sample_'):
                try:
                    sample_id = int(sample_name.split('_')[1])
                    available_sample_ids.append(sample_id)
                except:
                    pass
        available_sample_ids = sorted(list(set(available_sample_ids)))
        selected_sample_ids = parse_range_string(args.samples, available_sample_ids)
        print(f"  {dataset_name}: {len(selected_sample_ids)} samples processed")

if __name__ == "__main__":
    main()
