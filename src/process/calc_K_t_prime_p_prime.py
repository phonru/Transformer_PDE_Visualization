import numpy as np
import os
import torch
import argparse
from tqdm import tqdm
import sys
import os

# 添加项目根目录到 sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
    
"""
简化K张量计算脚本

原始公式：
K(t,p,d;t',p',d') = A_TT(p',t,t') * W^σ^T(t',p) * A_PP(t,p,p') * W^σ^S(t,p') * W_dd^1(d',d)
                   + I_TT(t,t') * A_PP(t,p,p') * W^σ^S(t,p') * W_dd^2(d',d)
                   + I_PP(p',p) * A_TT(p,t,t') * W^σ^T(t',p) * W_dd^3(d',d)
简化为2维张量：
K_t_prime_p_prime: (t', p') - 对应固定(t,p,d,d')下的核函数值
"""

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="计算简化的K张量 - 固定(t,p,d,d')让(t',p')变化")
    
    # 路径配置
    parser.add_argument('--input-dir', type=str, 
                       default="data/processed/HS_p",
                       help='输入数据目录路径 (默认: %(default)s)')
    parser.add_argument('--output-dir', type=str, 
                       default="data/processed/K_simplified",
                       help='输出目录路径 (默认: %(default)s)')
    
    # 数据集样本选择
    parser.add_argument('--datasets', type=str, default='all',
                       help='要处理的数据集，支持格式: "all", "0", "0,1,2", "0-2" (默认: %(default)s)')
    parser.add_argument('--samples', type=str, default='all',
                       help='要处理的样本，支持格式: "all", "0", "0,1,2", "0-5" (默认: %(default)s)')
    
    # 随机种子配置
    parser.add_argument('--global-seed', type=int, default=3571,
                       help='全局随机种子，设为-1表示完全随机 (默认: %(default)s)')
    
    # 固定维度配置（这些将被固定，不变化）
    parser.add_argument('--d-fixed', type=int, default=None,
                       help='固定特征维度d，范围[0, d_model)，通常640 (默认: 随机选择)')
    parser.add_argument('--d-prime-fixed', type=int, default=None,
                       help='固定特征维度d\'，范围[0, d_model)，通常640 (默认: 随机选择)')
    parser.add_argument('--t-fixed', type=int, default=None,
                       help='固定时间索引t，范围[0, t)，通常15 (默认: 随机选择)')
    parser.add_argument('--p-fixed', type=int, default=None,
                       help='固定空间索引p，范围[0, p)，通常100 (默认: 随机选择)')

    # 输出路径配置
    parser.add_argument('--add-seed-suffix', action='store_true', default=False,
                       help='是否在输出目录后添加随机种子后缀 (默认: False)')
    
    return parser.parse_args()

def compute_K_simplified(At, As, W_sigma_t, W_sigma_s, Wdd1, Wdd2, Wdd3, 
                         random_seed=None, d_fixed=None, d_prime_fixed=None, 
                         t_fixed=None, p_fixed=None):
    """
    简化计算K张量为2维张量，固定(t,p,d,d')让(t',p')变化：
    K_t_prime_p_prime: (t', p') - 对应固定(t,p,d,d')下的核函数值
    
    Args:
        At, As, W_sigma_t, W_sigma_s, Wdd1, Wdd2, Wdd3: 张量数据
        random_seed: 随机种子
        d_fixed: 指定的特征维度 d，None表示随机选择
        d_prime_fixed: 指定的特征维度 d'，None表示随机选择
        t_fixed: 指定的时间索引 t，None表示随机选择
        p_fixed: 指定的空间索引 p，None表示随机选择
    
    原公式：
    K(t,p,d;t',p',d') = A_TT(p',t,t') * W^σ^T(t',p) * A_PP(t,p,p') * W^σ^S(t,p') * W_dd^1(d',d)
                       + I_TT(t,t') * A_PP(t,p,p') * W^σ^S(t,p') * W_dd^2(d',d)
                       + I_PP(p',p) * A_TT(p,t,t') * W^σ^T(t',p) * W_dd^3(d',d)
    """
    # 设置随机种子以保证可重复性
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # 转换为torch张量
    At_torch = torch.from_numpy(At).float()  # (p, t, t) - 注意：现在A_TT是3维的
    As_torch = torch.from_numpy(As).float()  # (t, p, p) - 注意：现在A_PP是3维的
    
    # 从W_sigma_t获取维度信息：(t, p)
    t_max, p_max = W_sigma_t.shape
    d_model = Wdd1.shape[0]  # 从Wdd矩阵获取d_model维度，通常是640
    
    # 权重矩阵转为torch
    Wdd1_torch = torch.from_numpy(Wdd1).float()
    Wdd2_torch = torch.from_numpy(Wdd2).float()
    Wdd3_torch = torch.from_numpy(Wdd3).float()
    W_sigma_t_torch = torch.from_numpy(W_sigma_t).float()  # (t, p)
    W_sigma_s_torch = torch.from_numpy(W_sigma_s).float()  # (t, p)
    
    # 选择固定的维度索引：如果指定了就用指定值，否则随机选择
    d_fixed_val = d_fixed if d_fixed is not None else np.random.randint(0, d_model)
    d_prime_fixed_val = d_prime_fixed if d_prime_fixed is not None else np.random.randint(0, d_model)
    t_fixed_val = t_fixed if t_fixed is not None else np.random.randint(0, t_max)
    p_fixed_val = p_fixed if p_fixed is not None else np.random.randint(0, p_max)
    
    # 检查索引有效性
    if not (0 <= d_fixed_val < d_model):
        raise ValueError(f"d_fixed_val={d_fixed_val} 超出范围 [0, {d_model})")
    if not (0 <= d_prime_fixed_val < d_model):
        raise ValueError(f"d_prime_fixed_val={d_prime_fixed_val} 超出范围 [0, {d_model})")
    if not (0 <= t_fixed_val < t_max):
        raise ValueError(f"t_fixed_val={t_fixed_val} 超出范围 [0, {t_max})")
    if not (0 <= p_fixed_val < p_max):
        raise ValueError(f"p_fixed_val={p_fixed_val} 超出范围 [0, {p_max})")
    
    print(f"    使用固定索引: t={t_fixed_val}{'(fixed)' if t_fixed is not None else '(random)'}, " +
          f"p={p_fixed_val}{'(fixed)' if p_fixed is not None else '(random)'}, " +
          f"d={d_fixed_val}{'(fixed)' if d_fixed is not None else '(random)'}, " +
          f"d'={d_prime_fixed_val}{'(fixed)' if d_prime_fixed is not None else '(random)'}")
    
    # 计算固定特征维度下的权重标量
    w1_scalar = Wdd1_torch[d_prime_fixed_val, d_fixed_val].item()  # 标量
    w2_scalar = Wdd2_torch[d_prime_fixed_val, d_fixed_val].item()
    w3_scalar = Wdd3_torch[d_prime_fixed_val, d_fixed_val].item()
    
    # ===== 计算 K_t_prime_p_prime: (t', p') =====
    # 固定 t=t_fixed_val, p=p_fixed_val, d=d_fixed_val, d'=d_prime_fixed_val
    # K(t,p,d;t',p',d') 的公式简化为关于(t',p')的函数
    
    # Term 1: A_TT(p',t,t') * W^σ^T(t',p) * A_PP(t,p,p') * W^σ^S(t,p') * w1
    # 简化为: A_TT(p',t_fixed_val,t') * W^σ^T(t',p_fixed_val) * A_PP(t_fixed_val,p_fixed_val,p') * W^σ^S(t_fixed_val,p') * w1
    # 需要计算: (p',t') * (t',) * (p',) * (p',) * scalar -> (t', p')
    
    # A_TT(p',t_fixed_val,t') -> (p', t')
    a_tt_term1 = At_torch[:, t_fixed_val, :]  # (p', t')
    # W^σ^T(t',p_fixed_val) -> (t',)
    w_sigma_t_p_fixed = W_sigma_t_torch[:, p_fixed_val]  # (t',)
    # A_PP(t_fixed_val,p_fixed_val,p') -> (p',)
    a_pp_term1 = As_torch[t_fixed_val, p_fixed_val, :]  # (p',)
    # W^σ^S(t_fixed_val,p') -> (p',)
    w_sigma_s_t_fixed = W_sigma_s_torch[t_fixed_val, :]  # (p',)
    
    # 计算Term1: (p',t') * (t',) * (p',) * (p',) -> (t', p')
    # 先处理时间维度：(p',t') * (t',) -> (p',t')
    term1_pt_part = a_tt_term1 * w_sigma_t_p_fixed[None, :]  # (p', t')
    # 再处理空间维度：(p',t') * (p',) * (p',) -> (p',t')
    term1_pt_part = term1_pt_part * (a_pp_term1[:, None] * w_sigma_s_t_fixed[:, None])  # (p', t')
    # 转置为 (t', p')
    term1_t_p = term1_pt_part.T * w1_scalar  # (t', p')
    
    # Term 2: I_TT(t,t') * A_PP(t,p,p') * W^σ^S(t,p') * w2
    # I_TT(t_fixed_val,t') = 1 only when t'=t_fixed_val
    # 简化为: delta(t', t_fixed_val) * A_PP(t_fixed_val,p_fixed_val,p') * W^σ^S(t_fixed_val,p') * w2
    term2_t_p = torch.zeros(t_max, p_max)
    term2_t_p[t_fixed_val, :] = a_pp_term1 * w_sigma_s_t_fixed * w2_scalar
    
    # Term 3: I_PP(p',p) * A_TT(p,t,t') * W^σ^T(t',p) * w3
    # I_PP(p', p_fixed_val) = 1 only when p'=p_fixed_val
    # 简化为: delta(p', p_fixed_val) * A_TT(p_fixed_val,t_fixed_val,t') * W^σ^T(t',p_fixed_val) * w3
    term3_t_p = torch.zeros(t_max, p_max)
    a_tt_term3 = At_torch[p_fixed_val, t_fixed_val, :]  # (t',) - A_TT(p_fixed_val,t_fixed_val,t')
    term3_t_p[:, p_fixed_val] = a_tt_term3 * w_sigma_t_p_fixed * w3_scalar
    
    K_t_prime_p_prime = term1_t_p + term2_t_p + term3_t_p
    
    # 返回K张量和随机数索引
    fixed_indices = {
        'd_fixed_val': d_fixed_val,
        'd_prime_fixed_val': d_prime_fixed_val,
        't_fixed_val': t_fixed_val,
        'p_fixed_val': p_fixed_val,
        'd_fixed': d_fixed is not None,
        'd_prime_fixed': d_prime_fixed is not None,
        't_fixed': t_fixed is not None,
        'p_fixed': p_fixed is not None
    }
    
    return (K_t_prime_p_prime.numpy(), fixed_indices,
            term1_t_p.numpy(), term2_t_p.numpy(), term3_t_p.numpy())

def process_sample(sample_file, output_dir, global_seed=None, 
                  d_fixed=None, d_prime_fixed=None, t_fixed=None, p_fixed=None):
    """
    处理单个样本，计算简化的K张量
    
    Args:
        sample_file: 样本文件路径
        output_dir: 输出目录
        global_seed: 全局随机种子
        d_fixed: 指定的特征维度 d，None表示随机选择
        d_prime_fixed: 指定的特征维度 d'，None表示随机选择
        t_fixed: 指定的时间索引 t，None表示随机选择
        p_fixed: 指定的空间索引 p，None表示随机选择
    """
    sample_name = os.path.basename(sample_file).replace('_weights.npz', '')
    print(f"Processing {sample_name}...")
    
    # 加载权重数据
    data = np.load(sample_file)
    
    # 处理每一层
    K_t_prime_p_prime_layers = []
    fixed_indices_layers = []
    # 保存三个项的列表 - K(t',p')
    term1_t_p_layers = []
    term2_t_p_layers = []
    term3_t_p_layers = []
    # 保存1+2项的和
    term12_sum_t_p_layers = []
    
    for layer in range(6):
        print(f"  Computing simplified K tensors for layer {layer+1}...")
        
        # 获取当前层的数据
        Wdd1 = data[f'layer_{layer}_Wdd1']
        Wdd2 = data[f'layer_{layer}_Wdd2']
        Wdd3 = data[f'layer_{layer}_Wdd3']
        W_sigma_t = data[f'layer_{layer}_W_sigma_t']
        W_sigma_s = data[f'layer_{layer}_W_sigma_s']
        At = data[f'layer_{layer}_At']
        As = data[f'layer_{layer}_As']
        
        # 使用全局随机种子（所有层使用相同种子）
        random_seed = global_seed
        
        print(f"    Layer {layer+1} random seed: {random_seed}")
        
        # 计算简化的K张量
        result = compute_K_simplified(
            At, As, W_sigma_t, W_sigma_s, Wdd1, Wdd2, Wdd3, 
            random_seed=random_seed, d_fixed=d_fixed, d_prime_fixed=d_prime_fixed,
            t_fixed=t_fixed, p_fixed=p_fixed)
        
        K_t_prime_p_prime, fixed_indices = result[:2]
        term1_t_p, term2_t_p, term3_t_p = result[2:5]
        
        K_t_prime_p_prime_layers.append(K_t_prime_p_prime)
        fixed_indices_layers.append(fixed_indices)
        # 保存K(t',p')的三个项
        term1_t_p_layers.append(term1_t_p)
        term2_t_p_layers.append(term2_t_p)
        term3_t_p_layers.append(term3_t_p)
        # 计算并保存1+2项的和
        term12_sum_t_p = term1_t_p + term2_t_p
        term12_sum_t_p_layers.append(term12_sum_t_p)
        
        print(f"    Layer {layer+1}: K_t_prime_p_prime shape = {K_t_prime_p_prime.shape}")
        print(f"    Layer {layer+1}: K(t',p') terms shape = {term1_t_p.shape} each")
        print(f"    Layer {layer+1}: term1+term2 sum shape = {term12_sum_t_p.shape}")
    
    # 保存结果
    output_file = os.path.join(output_dir, f'{sample_name}_K_simplified.npz')
    save_dict = {}
    for i in range(6):
        save_dict[f'layer_{i}_K_t_prime_p_prime'] = K_t_prime_p_prime_layers[i]
        # 保存K(t',p')计算公式的三个项
        save_dict[f'layer_{i}_term1_t_p'] = term1_t_p_layers[i]
        save_dict[f'layer_{i}_term2_t_p'] = term2_t_p_layers[i]
        save_dict[f'layer_{i}_term3_t_p'] = term3_t_p_layers[i]
        # 保存1+2项的和
        save_dict[f'layer_{i}_term12_sum_t_p'] = term12_sum_t_p_layers[i]
    
    np.savez(output_file, **save_dict)
    print(f"Saved simplified K tensors and individual terms to {output_file}")
    
    # 返回固定索引供主函数保存
    return fixed_indices_layers[0] if fixed_indices_layers else None

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

def main():
    """主函数 - 计算简化的K张量 - 固定(t,p,d,d')让(t',p')变化"""
    # 解析命令行参数
    args = parse_arguments()
    
    print("Starting simplified K tensor calculation...")
    print("Computing K_t_prime_p_prime: (t', p') with fixed (t,p,d,d')")
    print("=" * 60)
    
    # 处理随机种子
    global_seed = args.global_seed if args.global_seed != -1 else None
    
    # 设置输出目录
    if args.add_seed_suffix:
        if global_seed is not None:
            output_dir = f"{args.output_dir}_{global_seed}"
        else:
            output_dir = f"{args.output_dir}_random"
    else:
        output_dir = args.output_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 打印配置信息
    print("📋 当前配置:")
    print(f"  输入目录: {args.input_dir}")
    print(f"  输出目录: {output_dir}")
    print(f"  全局随机种子: {global_seed}")
    print(f"  数据集选择: {args.datasets}")
    print(f"  样本选择: {args.samples}")
    print(f"  固定维度配置:")
    print(f"    d_fixed: {args.d_fixed}")
    print(f"    d'_fixed: {args.d_prime_fixed}")
    print(f"    t_fixed: {args.t_fixed}")
    print(f"    p_fixed: {args.p_fixed}")
    print()
    
    # 随机种子说明
    if global_seed is not None:
        print("🎲 种子生成规则: 所有样本和层使用相同的全局种子")
    else:
        print("🎲 种子生成规则: 完全随机（每次运行结果不同）")
    print()
    
    # 获取所有数据集目录
    dataset_dirs = [d for d in os.listdir(args.input_dir) 
                   if os.path.isdir(os.path.join(args.input_dir, d)) and d.startswith('dataset_')]
    dataset_dirs.sort()
    
    if not dataset_dirs:
        print("❌ 未找到任何数据集目录！")
        return
    
    # 解析数据集范围
    available_dataset_ids = [int(d.split('_')[1]) for d in dataset_dirs]
    selected_dataset_ids = parse_range_string(args.datasets, available_dataset_ids)
    selected_dataset_dirs = [f'dataset_{id}' for id in selected_dataset_ids]
    
    print(f"📁 找到数据集: {[d.split('_')[1] for d in dataset_dirs]}")
    print(f"📁 将处理数据集: {[str(id) for id in selected_dataset_ids]}")
    print()
    
    for dataset_dir in selected_dataset_dirs:
        print(f"🔄 处理 {dataset_dir}...")
        dataset_input_path = os.path.join(args.input_dir, dataset_dir)
        
        # 为每个数据集创建输出目录
        dataset_output_dir = os.path.join(output_dir, dataset_dir)
        os.makedirs(dataset_output_dir, exist_ok=True)
        
        # 获取该数据集下的所有权重文件
        weight_files = [f for f in os.listdir(dataset_input_path) if f.endswith('_weights.npz')]
        weight_files.sort()
        
        if not weight_files:
            print(f"  ⚠️  {dataset_dir} 中未找到权重文件，跳过")
            continue
        
        # 解析样本范围
        available_sample_ids = [int(f.split('_')[1]) for f in weight_files 
                              if f.startswith('sample_') and f.endswith('_weights.npz')]
        selected_sample_ids = parse_range_string(args.samples, available_sample_ids)
        selected_weight_files = [f'sample_{id}_weights.npz' for id in selected_sample_ids]
        
        print(f"  📄 找到样本文件: {len(weight_files)} 个")
        print(f"  📄 将处理样本: {selected_sample_ids}")
        
        # 保存随机数（只需要处理第一个样本来获取随机数）
        random_indices = None
        
        for weight_file in tqdm(selected_weight_files, desc=f"处理 {dataset_dir}"):
            if weight_file not in weight_files:
                print(f"  ⚠️  样本文件 {weight_file} 不存在，跳过")
                continue
                
            weight_path = os.path.join(dataset_input_path, weight_file)
            sample_fixed_indices = process_sample(weight_path, dataset_output_dir, 
                         global_seed=global_seed,
                         d_fixed=args.d_fixed, d_prime_fixed=args.d_prime_fixed,
                         t_fixed=args.t_fixed, p_fixed=args.p_fixed)
            
            # 只保存第一个样本的固定索引（所有样本都相同）
            if random_indices is None:
                random_indices = sample_fixed_indices
        
        # 保存固定索引到dataset目录下的rand.txt
        if random_indices is not None:
            random_file = os.path.join(dataset_output_dir, 'rand.txt')
            with open(random_file, 'w') as f:
                f.write(f"Dataset: {dataset_dir}\n")
                f.write(f"Global Seed: {global_seed}\n")
                f.write(f"Configuration:\n")
                f.write(f"  d_fixed: {args.d_fixed}\n")
                f.write(f"  d'_fixed: {args.d_prime_fixed}\n")
                f.write(f"  t_fixed: {args.t_fixed}\n")
                f.write(f"  p_fixed: {args.p_fixed}\n")
                f.write("-" * 50 + "\n")
                f.write(f"Actual fixed indices used (same for all samples and layers):\n")
                f.write(f"t_fixed_val: {random_indices['t_fixed_val']} {'(fixed)' if random_indices['t_fixed'] else '(random)'}\n")
                f.write(f"p_fixed_val: {random_indices['p_fixed_val']} {'(fixed)' if random_indices['p_fixed'] else '(random)'}\n")
                f.write(f"d_fixed_val: {random_indices['d_fixed_val']} {'(fixed)' if random_indices['d_fixed'] else '(random)'}\n")
                f.write(f"d_prime_fixed_val: {random_indices['d_prime_fixed_val']} {'(fixed)' if random_indices['d_prime_fixed'] else '(random)'}\n")
                f.write("\n")
                f.write("Note: (t',p') vary while (t,p,d,d') are fixed for all samples and layers.\n")
            
            print(f"  💾 已保存固定索引到 {random_file}")
    
    print()
    print("✅ 简化K张量计算完成！")
    print("📊 保存的张量（每层包含）:")
    print("  - K_t_prime_p_prime: (t', p') 对应固定(t,p,d,d')下的核函数值")
    print("  - term1_t_p, term2_t_p, term3_t_p: K(t',p')的三个分项")
    print("  - term12_sum_t_p: term1和term2的和（前两项之和）")
    print("    其中 K_t_prime_p_prime = term1_t_p + term2_t_p + term3_t_p")
    print("    其中 term12_sum_t_p = term1_t_p + term2_t_p")
    print(f"📁 结果保存在: {output_dir}")

if __name__ == "__main__":
    main()
