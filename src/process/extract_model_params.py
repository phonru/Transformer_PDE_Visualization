#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从模型检查点中提取注意力层参数和LayerNorm层参数，并保存到指定目录
支持单独或同时提取两种类型的参数
"""
import torch
import os
import numpy as np
import argparse
from pathlib import Path
from collections import OrderedDict
import sys

# 添加项目根目录到 sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.time_tansformer_block_2d_Spatiotemporal_emb_ori import PDE_CNN_Spatiotemporal_Transformer

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='提取模型注意力层和LayerNorm层参数')
    
    # 基础配置
    parser.add_argument('--checkpoint_path', type=str, 
                       default='/data/1/pkq2/data3/latest_v1.tar',
                       help='模型检查点文件路径')
    parser.add_argument('--save_dir', type=str,
                       default='data/weight',
                       help='参数保存的根目录')
    parser.add_argument('--device', type=str, default='auto',
                       help='使用的设备 (auto/cpu/cuda:0/cuda:1/...)')
    
    # 提取控制开关
    parser.add_argument('--extract_attention', action='store_true',
                       help='提取注意力层参数')
    parser.add_argument('--extract_layernorm', action='store_true',
                       help='提取LayerNorm层参数')
    parser.add_argument('--extract_all', action='store_true',
                       help='提取所有参数（注意力+LayerNorm）')
    
    # 保存目录配置
    parser.add_argument('--attention_subdir', type=str, default='attn',
                       help='注意力参数保存子目录名')
    parser.add_argument('--layernorm_subdir', type=str, default='LN',
                       help='LayerNorm参数保存子目录名')
    
    # 其他选项
    parser.add_argument('--verbose', action='store_true',
                       help='详细输出模式')
    parser.add_argument('--save_summary', action='store_true', default=True,
                       help='保存参数汇总信息')
    
    args = parser.parse_args()
    
    # 处理extract_all选项
    if args.extract_all:
        args.extract_attention = True
        args.extract_layernorm = True
    
    # 如果没有指定任何提取选项，默认提取所有
    if not args.extract_attention and not args.extract_layernorm:
        args.extract_attention = True
        args.extract_layernorm = True
        print("未指定提取选项，默认提取所有参数")
    
    return args

def setup_device(device_arg):
    """设置设备"""
    if device_arg == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_arg)
    
    print(f"使用设备: {device}")
    return device

def load_model_and_checkpoint(checkpoint_path, device='cpu', verbose=False):
    """加载模型和检查点"""
    if verbose:
        print(f"正在加载检查点: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # 获取模型状态字典
    state_dict = checkpoint.get('state_dict', checkpoint)
    
    # 处理 state_dict，去掉可能的 module. 前缀
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k.replace('module.', '') if k.startswith('module.') else k
        new_state_dict[new_key] = v
    
    if verbose:
        print(f"检查点加载完成，包含 {len(new_state_dict)} 个参数")
    
    return new_state_dict

def extract_attention_params(model, verbose=False):
    """提取注意力层参数"""
    if verbose:
        print("\n开始提取注意力层参数...")
    
    attn_params = {}
    linear_names = ['Wq', 'Wk', 'Wv', 'Wo']
    
    # 针对新模型结构，提取 SpatiotemporalEncoder 中的注意力层
    if hasattr(model, 'st_encoder'):
        st_encoder = model.st_encoder
        for idx, layer in enumerate(st_encoder.layers):
            # temporal_attn
            if hasattr(layer, 'temporal_attn'):
                t_attn = layer.temporal_attn
                t_name = f'layer{idx+1}_temporal_attn'
                attn_params[t_name] = {}
                for i, linear in enumerate(t_attn.linears):
                    w = linear.weight.detach().cpu().numpy()
                    b = linear.bias.detach().cpu().numpy() if linear.bias is not None else None
                    attn_params[t_name][f'{linear_names[i]}_weight'] = w
                    if b is not None:
                        attn_params[t_name][f'{linear_names[i]}_bias'] = b
                
                if verbose:
                    print(f"提取 {t_name}")
            
            # spatial_attn
            if hasattr(layer, 'spatial_attn'):
                s_attn = layer.spatial_attn
                s_name = f'layer{idx+1}_spatial_attn'
                attn_params[s_name] = {}
                for i, linear in enumerate(s_attn.linears):
                    w = linear.weight.detach().cpu().numpy()
                    b = linear.bias.detach().cpu().numpy() if linear.bias is not None else None
                    attn_params[s_name][f'{linear_names[i]}_weight'] = w
                    if b is not None:
                        attn_params[s_name][f'{linear_names[i]}_bias'] = b
                
                if verbose:
                    print(f"提取 {s_name}")
    else:
        # 兼容其他情况，直接提取所有MultiHeadedAttention
        for name, module in model.named_modules():
            if module.__class__.__name__ == 'MultiHeadedAttention':
                attn_params[name] = {}
                for i, linear in enumerate(module.linears):
                    w = linear.weight.detach().cpu().numpy()
                    b = linear.bias.detach().cpu().numpy() if linear.bias is not None else None
                    attn_params[name][f'{linear_names[i]}_weight'] = w
                    if b is not None:
                        attn_params[name][f'{linear_names[i]}_bias'] = b
                
                if verbose:
                    print(f"提取 {name}")
    
    return attn_params

def save_attention_params(attn_params, save_dir, verbose=False, save_summary=True):
    """保存注意力层参数"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print(f"\n保存注意力参数到: {save_dir}")
    
    for layer_name, params in attn_params.items():
        layer_dir = save_dir / layer_name.replace('.', '_')
        layer_dir.mkdir(parents=True, exist_ok=True)
        
        for pname, value in params.items():
            npy_path = layer_dir / f'{pname}.npy'
            np.save(npy_path, value)
            
            if verbose:
                print(f"  保存 {layer_name}/{pname}: {value.shape}")
    
    # 保存汇总信息
    if save_summary:
        summary_path = save_dir / "attention_summary.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("注意力参数提取汇总\n")
            f.write("=" * 50 + "\n\n")
            
            for layer_name, params in attn_params.items():
                f.write(f"层名称: {layer_name}\n")
                for pname, value in params.items():
                    f.write(f"  {pname}: {value.shape}\n")
                    f.write(f"    统计: min={value.min():.6f}, max={value.max():.6f}, mean={value.mean():.6f}, std={value.std():.6f}\n")
                f.write("-" * 30 + "\n")
        
        if verbose:
            print(f"注意力参数汇总保存到: {summary_path}")

def extract_layernorm_params(state_dict, verbose=False):
    """从状态字典中提取所有LayerNorm层的参数"""
    layer_norm_params = {}
    
    if verbose:
        print("\n开始提取LayerNorm参数...")
    
    for param_name, param_tensor in state_dict.items():
        # 查找包含LayerNorm相关的参数
        if any(keyword in param_name for keyword in ['.norm.a_2', '.norm.b_2', 'norm.a_2', 'norm.b_2']):
            if verbose:
                print(f"发现LayerNorm参数: {param_name}")
                print(f"  形状: {param_tensor.shape}")
            
            # 将参数转换为numpy数组
            param_numpy = param_tensor.cpu().numpy()
            layer_norm_params[param_name] = param_numpy
    
    return layer_norm_params

def save_layernorm_params(layer_norm_params, save_dir, verbose=False, save_summary=True):
    """保存LayerNorm层参数"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print(f"\n保存LayerNorm参数到: {save_dir}")
    
    for param_name, param_numpy in layer_norm_params.items():
        # 将参数名中的点替换为下划线，便于文件命名
        clean_name = param_name.replace('.', '_')
        save_path = save_dir / f"{clean_name}.npy"
        
        # 保存参数
        np.save(save_path, param_numpy)
        
        if verbose:
            print(f"  保存 {param_name}: {param_numpy.shape}")
            print(f"    统计: min={param_numpy.min():.6f}, max={param_numpy.max():.6f}, mean={param_numpy.mean():.6f}, std={param_numpy.std():.6f}")
    
    # 保存汇总信息
    if save_summary:
        summary_path = save_dir / "layernorm_summary.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("LayerNorm参数提取汇总\n")
            f.write("=" * 50 + "\n\n")
            
            for param_name, param_array in layer_norm_params.items():
                f.write(f"参数名: {param_name}\n")
                f.write(f"形状: {param_array.shape}\n")
                f.write(f"统计信息:\n")
                f.write(f"  - 最小值: {param_array.min():.6f}\n")
                f.write(f"  - 最大值: {param_array.max():.6f}\n")
                f.write(f"  - 均值: {param_array.mean():.6f}\n")
                f.write(f"  - 标准差: {param_array.std():.6f}\n")
                f.write(f"  - 非零元素数量: {np.count_nonzero(param_array)}\n")
                f.write(f"  - 总元素数量: {param_array.size}\n")
                f.write("-" * 30 + "\n")
        
        if verbose:
            print(f"LayerNorm参数汇总保存到: {summary_path}")

def main():
    # 解析命令行参数
    args = parse_args()
    
    print("模型参数提取脚本")
    print("=" * 60)
    print(f"检查点路径: {args.checkpoint_path}")
    print(f"保存根目录: {args.save_dir}")
    print(f"提取注意力参数: {args.extract_attention}")
    print(f"提取LayerNorm参数: {args.extract_layernorm}")
    
    # 设置设备
    device = setup_device(args.device)
    
    try:
        # 创建模型实例 - 使用固定配置
        model = PDE_CNN_Spatiotemporal_Transformer(
            d_model=640,           # 固定模型维度
            img_size=32*5,         # 固定图像尺寸 160
            patch_size=16,         # 固定补丁尺寸
            device=device
        ).to(device)
        
        # 加载检查点
        state_dict = load_model_and_checkpoint(args.checkpoint_path, device, args.verbose)
        
        # 加载模型权重
        model.load_state_dict(state_dict, strict=False)
        if args.verbose:
            print(f'模型权重加载完成')
        
        # 提取注意力参数
        if args.extract_attention:
            print("\n" + "=" * 40)
            print("提取注意力层参数")
            print("=" * 40)
            
            attn_params = extract_attention_params(model, args.verbose)
            attn_save_dir = Path(args.save_dir) / args.attention_subdir
            save_attention_params(attn_params, attn_save_dir, args.verbose, args.save_summary)
            
            print(f"注意力参数提取完成！共提取 {len(attn_params)} 个注意力层")
            print(f"参数已保存到: {attn_save_dir}")
            
            # 打印每层参数形状汇总
            if args.verbose:
                print("\n注意力层参数形状汇总:")
                for layer_name, params in attn_params.items():
                    print(f'[{layer_name}]')
                    for pname, value in params.items():
                        print(f'  {pname}: {value.shape}')
        
        # 提取LayerNorm参数
        if args.extract_layernorm:
            print("\n" + "=" * 40)
            print("提取LayerNorm层参数")
            print("=" * 40)
            
            layer_norm_params = extract_layernorm_params(state_dict, args.verbose)
            ln_save_dir = Path(args.save_dir) / args.layernorm_subdir
            save_layernorm_params(layer_norm_params, ln_save_dir, args.verbose, args.save_summary)
            
            print(f"LayerNorm参数提取完成！共提取 {len(layer_norm_params)} 个LayerNorm参数")
            print(f"参数已保存到: {ln_save_dir}")
        
        print("\n" + "=" * 60)
        print("所有参数提取完成!")
        
        # 验证保存的文件
        if args.verbose:
            if args.extract_attention:
                attn_files = list((Path(args.save_dir) / args.attention_subdir).rglob("*.npy"))
                print(f"\n注意力参数文件数量: {len(attn_files)}")
            
            if args.extract_layernorm:
                ln_files = list((Path(args.save_dir) / args.layernorm_subdir).glob("*.npy"))
                print(f"LayerNorm参数文件数量: {len(ln_files)}")
        
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
