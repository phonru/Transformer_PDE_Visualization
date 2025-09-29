import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from tqdm import tqdm
import types
import argparse
import logging

# 添加项目根目录到 sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.fft_stretch_test import fourier_downsample_2d, fourier_upsample_2d
from src.utils.dataset_2d_multi_dataloader import InitDataset_sample
from model.time_tansformer_block_2d_Spatiotemporal_emb_ori import PDE_CNN_Spatiotemporal_Transformer


def parse_args():
    parser = argparse.ArgumentParser(description='Extract hidden states from transformer PDE model')
    
    # 可配置参数
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging for debugging')
    
    # 路径参数
    parser.add_argument('--checkpoint_path', type=str,
                        default='/data/1/pkq2/checkpoints/checkpoint_per10ep/seed_init/checkpoints/checkpoint_epoch_140.tar', 
                        help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='output/plots_pre',
                        help='Output directory for visualizations')
    parser.add_argument('--hidden_states_dir', type=str, default='data/hidden_states', 
                        help='Directory to save hidden states')
    
    # 数据集配置
    parser.add_argument('--dataset_files', type=str, nargs='+', 
                        default=[
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
                                ],  
                        help='dataset files')

    parser.add_argument('--dataset_sizes', type=int, nargs='+',
                       default=[240, 20, 68, 4, 19, 5, 42, 62, 35, 240, 12],
                       help='dataset sizes')

    parser.add_argument('--dataset_bs', type=int, nargs='+',
                       default=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                       help='dataset batch sizes')

    # 推理样本配置 (用字符串表示，格式如 "2:0,30" "3:0,2" "4:all")
    parser.add_argument('--processing_pairs', type=str, nargs='+',
                       default=["3:0,2", "4:all"],
                       help='List of dataset-sample pairs to process, e.g., "2:0,30" "3:0,2" "4:all". If not specified, will not process any samples.')
    
    return parser.parse_args()

def parse_processing_pairs(pairs_str_list):
    """
    解析 "dataset_idx:sample_indices" 格式的字符串列表。
    返回一个字典，键是数据集索引(int)，值是样本索引列表(list of int)或None(表示'all')。
    """
    processing_map = {}
    if not pairs_str_list:
        return processing_map
        
    for pair_str in pairs_str_list:
        try:
            ds_idx_str, sample_indices_str = pair_str.split(':', 1)
            ds_idx = int(ds_idx_str)
            
            if sample_indices_str.strip().lower() == 'all':
                processing_map[ds_idx] = None  # None 表示处理所有样本
            else:
                indices = [int(x.strip()) for x in sample_indices_str.split(',') if x.strip()]
                if ds_idx in processing_map:
                    # 如果已经存在，进行合并
                    if processing_map[ds_idx] is not None:
                        processing_map[ds_idx].extend(indices)
                else:
                    processing_map[ds_idx] = indices
        except ValueError:
            print(f"Warning: Invalid format for processing pair '{pair_str}'. Skipping.")
            continue
            
    # 对于有索引列表的，去重并排序
    for ds_idx in processing_map:
        if processing_map[ds_idx] is not None:
            processing_map[ds_idx] = sorted(list(set(processing_map[ds_idx])))
            
    return processing_map

# 数据处理函数，完全参考predict_2d_fft_block_mdr_Dpred.py
def time_shape_change(u, f, c, re, gt):
    b, c_in, t, h, w = f.shape
    u_gt = gt[:, :, 1:, :, :]
    u = u.transpose(1, 2)
    f = f[:, :, 1:, :, :].transpose(1, 2)
    c = c[:, :, :, 1:, :]
    c = torch.reshape(c, (b, -1, t - 1, w)).transpose(1, 2)
    re = re.unsqueeze(1).repeat(1, t - 1)
    return u, f, c, re, u_gt

def interpolate_u_f_c(u, f, u_gt, M_factor, time_step):
    u = u.permute(0, 2, 1, 3, 4).reshape((-1, 2, 32, 32))
    u_gt = u_gt.permute(0, 2, 1, 3, 4).reshape((-1, 2, 32, 32))
    f = f.permute(0, 2, 1, 3, 4).reshape((-1, 2, 32, 32))

    u = fourier_upsample_2d(u, M_factor)
    u_gt = fourier_upsample_2d(u_gt, M_factor)

    u = u.reshape((-1, time_step, 2, 32 * M_factor, 32 * M_factor)).permute(0, 2, 1, 3, 4)
    u_gt = u_gt.reshape((-1, time_step, 2, 32 * M_factor, 32 * M_factor)).permute(0, 2, 1, 3, 4)

    f = fourier_upsample_2d(f, M_factor)
    f = f.reshape((-1, time_step, 2, 32 * M_factor, 32 * M_factor)).permute(0, 2, 1, 3, 4)

    u_0 = u[:, :, None, :, 0, :]
    u_l = u[:, :, None, :, -1, :]
    v_0 = u[:, :, None, :, :, 0]
    v_l = u[:, :, None, :, :, -1]
    c = torch.cat((u_0, u_l, v_0, v_l), dim=2)
    return u, f, c, u_gt

def plot_prediction_vs_gt(prediction, ground_truth, save_dir, sample_idx, t_idx):
    os.makedirs(save_dir, exist_ok=True)
    components = ['u', 'v']
    xx, yy = np.meshgrid(np.linspace(0, 31, 32), np.linspace(0, 31, 32))
    for i, comp in enumerate(components):
        pred = prediction[i, t_idx, :, :].cpu().numpy()
        gt = ground_truth[i, t_idx, :, :].cpu().numpy()
        diff = np.abs(pred - gt)
        vmax = max(np.abs(pred).max(), np.abs(gt).max())
        diff_max = diff.max() if diff.max() > 0 else 1e-6
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        im0 = axes[0].contourf(xx, yy, gt, cmap='coolwarm', levels=50, vmin=-vmax, vmax=vmax)
        plt.colorbar(im0, ax=axes[0])
        axes[0].set_title('Ground Truth')
        im1 = axes[1].contourf(xx, yy, pred, cmap='coolwarm', levels=50, vmin=-vmax, vmax=vmax)
        plt.colorbar(im1, ax=axes[1])
        axes[1].set_title('Prediction')
        im2 = axes[2].contourf(xx, yy, diff, cmap='coolwarm', levels=50, vmin=0, vmax=diff_max)
        plt.colorbar(im2, ax=axes[2])
        axes[2].set_title('Abs. Error')
        for ax in axes:
            ax.set_aspect('equal')
            ax.axis('off')
        plt.tight_layout()
        fname = os.path.join(save_dir, f'sample{sample_idx}_ch{comp}_t{t_idx}.png')
        plt.savefig(fname)
        plt.close(fig)

def main():
    # 解析命令行参数
    args = parse_args()

    # 配置日志
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')

    # 固定的模型和数据参数（不可修改）
    d_model = 640
    patch_size = 16
    M_factor = 5
    N_original = 32
    time_step = 16
    cut_step = 1
    img_size = N_original * M_factor
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 解析要处理的数据集和样本对
    processing_map = parse_processing_pairs(args.processing_pairs)
    if not processing_map:
        logging.warning("No processing pairs specified via --processing_pairs. Exiting.")
        return

    device = torch.device(args.device)
    logging.info(f"Using device: {device}")
    
    # 使用训练脚本中的模型超参数设置
    model = PDE_CNN_Spatiotemporal_Transformer(d_model=d_model,
                                               img_size=img_size,
                                               patch_size=patch_size,
                                               device=device).to(device)

    # 递归查找所有MultiHeadedAttention实例
    attn_modules = []
    def find_attn(module):
        for name, child in module.named_children():
            if child.__class__.__name__ == 'MultiHeadedAttention':
                attn_modules.append(child)
            else:
                find_attn(child)
    find_attn(model)
    logging.info(f"Found {len(attn_modules)} MultiHeadedAttention modules.")

    # 递归查找所有SpatiotemporalEncoderLayer实例
    st_layer_modules = []
    def find_st_layer(module):
        for name, child in module.named_children():
            if child.__class__.__name__ == 'SpatiotemporalEncoderLayer':
                st_layer_modules.append(child)
            else:
                find_st_layer(child)
    find_st_layer(model)
    logging.info(f"Found {len(st_layer_modules)} SpatiotemporalEncoderLayer modules.")

    # monkey patch: 包裹forward，提取QKV和attn
    original_forwards = []
    # 命名区分：每层有temporal和spatial，按递归顺序分配
    attn_names = []
    for i, attn in enumerate(attn_modules):
        # 约定：偶数为temporal，奇数为spatial
        layer_idx = i // 2
        sub_name = 'temporal' if i % 2 == 0 else 'spatial'
        attn_names.append(f'attn_{layer_idx}_{sub_name}')
    for idx, attn in enumerate(attn_modules):
        orig_forward = attn.forward
        original_forwards.append(orig_forward)
        def make_new_forward(orig_forward, idx):
            def new_forward(self, query, key, value, mask=None):
                nbatches = query.size(0)
                h = self.h
                d_k = self.d_k
                V = self.linears[2](value).view(nbatches, -1, h, d_k).transpose(1, 2).detach().cpu()
                self._last_input = query.detach().cpu().squeeze(0)
                self._last_V = V.squeeze(0)
                out = orig_forward(query, key, value, mask)
                attn_weights = getattr(self, 'attn', None)
                if attn_weights is not None:
                    self._last_attn = attn_weights.detach().cpu().squeeze(0)
                else:
                    self._last_attn = None
                self._last_output = out.detach().cpu().squeeze(0)
                return out
            return types.MethodType(new_forward, attn)
        attn.forward = make_new_forward(orig_forward, idx)

    # monkey patch SpatiotemporalEncoderLayer: 提取第一个子层输入和第二个子层输出
    original_st_forwards = []
    for idx, st_layer in enumerate(st_layer_modules):
        orig_forward = st_layer.forward
        original_st_forwards.append(orig_forward)
        def make_new_st_forward(orig_forward, idx):
            def new_forward(self, x, temporal_mask):
                B, pn, T, D = x.shape
                
                # 保存第一个子层（时间注意力）的输入，去掉batch维度
                self._spatiotemporal_input = x.detach().cpu().squeeze(0)
                
                # 1. 时间注意力 (第一个子层)
                # Reshape: (B, pn, T, D) -> (B*pn, T, D)
                x_temp = x.reshape(B * pn, T, D)
                x_temp = self.sublayer[0](x_temp, lambda q: self.temporal_attn(q, q, q, temporal_mask))
                x = x_temp.view(B, pn, T, D)
                
                # 2. 空间注意力 (第二个子层)
                # Reshape: (B, pn, T, D) -> (B, T, pn, D) -> (B*T, pn, D)
                x_spatial = x.permute(0, 2, 1, 3).reshape(B * T, pn, D)
                # 空间上不需要mask
                x_spatial = self.sublayer[1](x_spatial, lambda q: self.spatial_attn(q, q, q, None)) 
                x = x_spatial.view(B, T, pn, D).permute(0, 2, 1, 3)
                
                # 保存第二个子层的输出，去掉batch维度
                self._spatiotemporal_output = x.detach().cpu().squeeze(0)
                
                # 3. 前馈网络 (第三个子层)
                x = self.sublayer[2](x, self.feed_forward)
                return x
            return types.MethodType(new_forward, st_layer)
        st_layer.forward = make_new_st_forward(orig_forward, idx)
    if os.path.isfile(args.checkpoint_path):
        checkpoint = torch.load(args.checkpoint_path, map_location=device, weights_only=False)
        state_dict = checkpoint['state_dict']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_key = k.replace('module.', '') if k.startswith('module.') else k
            new_state_dict[new_key] = v
        state_dict = new_state_dict
        model.load_state_dict(state_dict)
        epoch = checkpoint.get('epoch', 0)
        logging.info(f"Loaded checkpoint from {args.checkpoint_path} (epoch = {epoch})")
    else:
        logging.error(f"Checkpoint not found: {args.checkpoint_path}")
        return
    model.eval()

    # 根据 processing_map 遍历指定的数据集和样本
    for ds_idx, indices in processing_map.items():
        if ds_idx >= len(args.dataset_files):
            logging.warning(f"Dataset index {ds_idx} is out of range. Skipping.")
            continue

        fp = args.dataset_files[ds_idx]
        sz = args.dataset_sizes[ds_idx]
        
        try:
            dataset = InitDataset_sample(fp, sz)
            logging.info(f"Loaded dataset {ds_idx} from {fp} with {len(dataset)} samples.")
        except Exception as e:
            logging.error(f"Failed to load dataset {ds_idx} from {fp}. Error: {e}")
            continue
        
        # 确定要处理的样本索引
        if indices is None:  # 'all'
            sample_indices_to_process = list(range(len(dataset)))
        else:
            sample_indices_to_process = [i for i in indices if i < len(dataset)]
            if len(sample_indices_to_process) != len(indices):
                logging.warning(f"Some sample indices for dataset {ds_idx} were out of range and have been skipped.")

        for sample_idx in tqdm(sample_indices_to_process, desc=f"Processing dataset {ds_idx}"):
            try:
                u, f, c, re, u_gt = dataset[sample_idx]
            except IndexError:
                logging.warning(f"Sample index {sample_idx} out of bounds for dataset {ds_idx}. Skipping.")
                continue
            # 保证都是 torch.Tensor
            if not torch.is_tensor(u): u = torch.from_numpy(u)
            if not torch.is_tensor(f): f = torch.from_numpy(f)
            if not torch.is_tensor(c): c = torch.from_numpy(c)
            if not torch.is_tensor(re):
                if isinstance(re, np.generic) or np.isscalar(re):
                    re = torch.tensor(re)
                else:
                    re = torch.from_numpy(re)
            if not torch.is_tensor(u_gt): u_gt = torch.from_numpy(u_gt)
            # 增加batch维
            u = u.unsqueeze(0)
            f = f.unsqueeze(0)
            c = c.unsqueeze(0)
            re = re.unsqueeze(0)
            u_gt = u_gt.unsqueeze(0)
            u0, f0, c0, re0, u_gt0 = u.to(device), f.to(device), c.to(device), re.to(device), u_gt.to(device)
            u_large, f_large, c_large, u_gt_large = interpolate_u_f_c(u0, f0, u_gt0, M_factor, time_step)
            u_large, f_large, c_large, re_large, u_gt_large = time_shape_change(u_large, f_large, c_large, re0, u_gt_large)
            u_gt_original = u_gt0[:, :, 1:, :, :]
            u0_stretch = u_large[:, 0, :, :, :].squeeze(1)
            with torch.no_grad():
                pre_large = model(u0_stretch, f_large, c_large, re_large).permute(0, 2, 1, 3, 4)

            # 收集所有注意力层的输入输出等
            attn_data = {}
            for i, attn in enumerate(attn_modules):
                name = attn_names[i]
                attn_data[name] = {
                    'input': getattr(attn, '_last_input', None),
                    'V': getattr(attn, '_last_V', None),
                    'attn': getattr(attn, '_last_attn', None),
                    'output': getattr(attn, '_last_output', None),
                }
                if args.verbose:
                    logging.debug(f"[ds{ds_idx} sample{sample_idx}] Attn Layer '{name}':")
                    for k, v in attn_data[name].items():
                        if v is not None:
                            logging.debug(f"    - {k}: shape {tuple(v.shape)}")

            # 收集所有SpatiotemporalEncoderLayer的输入输出
            st_data = {}
            for i, st_layer in enumerate(st_layer_modules):
                name = f'st_layer_{i}'
                st_data[name] = {
                    'Spatiotemporal_input': getattr(st_layer, '_spatiotemporal_input', None),
                    'Spatiotemporal_output': getattr(st_layer, '_spatiotemporal_output', None),
                }
                if args.verbose:
                    logging.debug(f"[ds{ds_idx} sample{sample_idx}] ST Layer '{name}':")
                    for k, v in st_data[name].items():
                        if v is not None:
                            logging.debug(f"    - {k}: shape {tuple(v.shape)}")

            # 保存到hidden_states
            save_dir = os.path.join(args.hidden_states_dir, f'dataset_{ds_idx}')
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join (save_dir, f'sample_{sample_idx}.npz')
            # 展平成一层，key格式 attn_x_type/变量名 和 st_layer_x/变量名
            npz_dict = {}
            # 保存注意力层数据
            for name, d in attn_data.items():
                for k, v in d.items():
                    if v is not None:
                        npz_dict[f'{name}/{k}'] = v.numpy()
            # 保存Spatiotemporal层数据
            for name, d in st_data.items():
                for k, v in d.items():
                    if v is not None:
                        npz_dict[f'{name}/{k}'] = v.numpy()
            np.savez(save_path, **npz_dict)

            # 继续原有可视化
            pre_shape = pre_large.shape
            reshaped_pre = torch.transpose(pre_large, 1, 2).reshape(-1, pre_shape[1], pre_shape[3], pre_shape[4])
            reshaped_pre = fourier_downsample_2d(reshaped_pre, N_original, M_factor)
            prediction = torch.transpose(reshaped_pre.reshape(pre_shape[0], pre_shape[2], pre_shape[1], 32, 32), 1, 2)
            prediction = prediction.cpu().squeeze(0)
            ground_truth = u_gt_original.cpu().squeeze(0)
            num_timesteps = prediction.shape[1]
            for t_name, t_idx in zip(['initial', 'middle', 'final'], [0, (num_timesteps - 1) // 2, num_timesteps - 1]):
                save_dir_img = os.path.join(args.output_dir, f'dataset_{ds_idx}', f'sample_{sample_idx}')
                plot_prediction_vs_gt(prediction, ground_truth, save_dir_img, sample_idx, t_idx)
    
    logging.info(f'All tasks completed. Outputs are saved in {args.output_dir} and {args.hidden_states_dir}')

if __name__ == '__main__':
    main()
