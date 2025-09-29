import torch
import torch.nn as nn
import torch.fft as fft
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from src.utils.dataset_sep_2d import InitDataset_sample

# ==============================================================================
# 1. 傅里叶变换工具函数 (核心实现)
# ==============================================================================

def fourier_upsample_2d(field, M_factor):
    """
    通过傅里叶插值将二维场进行上采样。
    u(N, N) -> u_S(S, S) where S = N * M_factor

    :param field: 输入场，张量形状 (batch, channels, N, N)
    :param M_factor: 上采样倍数 M
    :return: 上采样后的场，张量形状 (batch, channels, S, S)
    """
    # 检查输入张量的 dtype 是否为 float16，如果是则转换为 float32
    if field.dtype == torch.float16:   
        field = field.to(torch.float32)
    N = field.shape[-1]
    S = N * M_factor

    # 1.1 对实数场进行傅里叶变换 (rfft2 更高效)
    # 形状: (batch, channels, N, N//2 + 1)
    field_fft = fft.rfft2(field,
                        #   norm='ortho'
                          )

    # 1.2 创建一个更大的、填满零的频域张量
    # 形状: (batch, channels, S, S//2 + 1)
    upsampled_fft = torch.zeros(field.shape[0], field.shape[1], S, S // 2 + 1,
                                dtype=field_fft.dtype, device=field.device)

    # 1.3 将低频系数复制到新张量的对应位置
    # 这是频域零填充的关键步骤
    # rfft2的输出，正频率在前半部分，负频率被折叠在后半部分
    n_h = N // 2
    
    # 复制第一象限和第四象限的频率 (正kx)
    upsampled_fft[..., :n_h, :n_h+1] = field_fft[..., :n_h, :]
    # 复制第二象限和第三象限的频率 (负kx, 在rfft中存储在数组后部)
    upsampled_fft[..., S-n_h:, :n_h+1] = field_fft[..., n_h:, :n_h+1]
    # upsampled_fft[..., S // 2, :N//2+1] = field_fft[..., n_h, :]   # Nyquist
    upsampled_fft[..., n_h, :n_h+1] = field_fft[..., n_h, :]   # Nyquist
    upsampled_fft *= M_factor**2     # 2-D 情况只乘一次

    # 1.4 逆傅里叶变换得到上采样后的空间场
    # s=(S, S) 指定了输出的空间尺寸
    upsampled_field = fft.irfft2(upsampled_fft, s=(S, S)
                                #  , norm='ortho'
                                 )
    
    return upsampled_field


def fourier_downsample_2d(field, N_original, M_factor):
    """
    通过傅里叶截断将二维场进行下采样 (上采样过程的逆操作)。
    u_S(S, S) -> u'(N, N)

    :param field: 输入的高分辨率场，张量形状 (batch, channels, S, S)
    :param N_original: 目标原始尺寸 N
    :return: 下采样/还原后的场，张量形状 (batch, channels, N, N)
    """
    # 检查输入张量的 dtype 是否为 float16，如果是则转换为 float32
    if field.dtype == torch.float16:   
        field = field.to(torch.float32)
    
    S = field.shape[-1]

    # 2.1 对高分辨率场进行傅里叶变换
    # 形状: (batch, channels, S, S//2 + 1)
    field_fft = fft.rfft2(field
                        #   , norm='ortho'
                          )

    # 2.2 创建一个小的、用于存放截断后频谱的张量
    # 形状: (batch, channels, N, N//2 + 1)
    downsampled_fft = torch.zeros(field.shape[0], field.shape[1], N_original, N_original // 2 + 1,
                                  dtype=field_fft.dtype, device=field.device)

    # 2.3 从高分辨率频谱中截取低频部分
    n_h = N_original // 2
    
    # 截取第一象限和第四象限
    downsampled_fft[..., :n_h, :] = field_fft[..., :n_h, :n_h+1]
    # 截取第二象限和第三象限
    downsampled_fft[..., n_h:, :n_h+1] = field_fft[..., S-n_h:, :n_h+1]
    # downsampled_fft[..., n_h, :] = field_fft[..., S // 2,  :N_original//2+1]   # Nyquist
    downsampled_fft[..., n_h, :] = field_fft[..., n_h,  :n_h+1]   # Nyquist
    
    downsampled_fft /= M_factor**2 

    # 2.4 逆傅里叶变换得到还原后的空间场
    downsampled_field = fft.irfft2(downsampled_fft, s=(N_original, N_original)
                                #    , norm='ortho'
                                   )
    
    return downsampled_field



if __name__ == '__main__':
    # -- 参数设置 --
    N_original = 32      # 原始物理场分辨率
    M_factor = 10         # 拉伸因子
    S_upsampled = N_original * M_factor # 拉伸后的分辨率

    batch_size = 2
    learning_rate = 1e-3
    epochs = 10

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # -- 数据准备 --
    train_dataset = InitDataset_sample('/data01/gxs/Kol_data/Kol_train_nonorm.h5')
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=16,
        shuffle=True,
        drop_last=True)    
    
    
    # 预处理：将原始高频场转换为拉伸后的平滑场，作为网络学习的目标
    u_original, _, _, _ = next(iter(train_loader))
    u_sha = u_original.shape
    u_original = u_original[:,1,1,:,:].reshape(-1, u_sha[1], u_sha[3], u_sha[4])
    u_target_stretched = fourier_upsample_2d(u_original, M_factor)
    print(type(fourier_upsample_2d))
    
    
    u_original = u_original.to(device)
    u_target_stretched = u_target_stretched.to(device)

    predicted_reconstructed = fourier_downsample_2d(u_target_stretched, N_original, M_factor)
    print(type(predicted_reconstructed))
    # 将数据移到CPU并转为numpy用于绘图
    u_orig_sample = u_original[0, 0].cpu().numpy()
    u_target_stretched_sample = u_target_stretched[0, 0].cpu().numpy()
    predicted_reconstructed_sample = predicted_reconstructed[0, 0].cpu().numpy()
    # u_original  (shape: B×1×N×N)
    #  u_target_stretched (shape: B×1×S×S)
    B, C, N, _ = u_original.shape
    M = M_factor    # 拉伸倍率
    S = N * M

    # ------------------ 方法 A：直接子采样检查 ------------------
    # 在上采样结果里，每隔 M 个像素抽样一次
    sampled_from_g = u_target_stretched[..., ::M, ::M]   # shape: B×1×N×N
    eq_A = torch.allclose(sampled_from_g, u_original, atol=1e-12)
    max_err = (sampled_from_g - u_original).abs().mean().item()
    print(f"[子采样一致性]  g[::M,::M] == f ?  {eq_A},  mean |err| = {max_err:e}")

    # ------------------ 方法 B：完整往返检查 ------------------
    u_roundtrip = fourier_downsample_2d(u_target_stretched, N_original, M_factor)
    
    
    eq_B = torch.allclose(u_roundtrip, u_original, atol=1e-12)
    max_err = (u_roundtrip - u_original).abs().max().item()
    print(f"[往返一致性]    down(up(f)) == f ?  {eq_B},  max |err| = {max_err:e}")

    # 若两项均 True 且 max_err ~ 1e-13 ～ 1e-15，则理论与实现完全一致。
    
    # ------------------ 绘图 ------------------
    fig, axes = plt.subplots(1, 4, figsize=(29, 6), constrained_layout=True)
    
    # im1 = axes[0].imshow(u_orig_sample, cmap='viridis', origin='lower', aspect='equal')
    im1 = axes[0].contourf(u_orig_sample, cmap='viridis', origin='lower', levels=20)
    axes[0].set_title(f'Original High-Freq Field (u)\n{N_original}x{N_original}', pad=10, fontsize=12)
    fig.colorbar(im1, ax=axes[0])
    
    # im2 = axes[1].imshow(u_target_stretched_sample, cmap='viridis', origin='lower', aspect='equal')
    im2 = axes[1].contourf(u_target_stretched_sample, cmap='viridis', origin='lower', levels=20)
    axes[1].set_title(f'Stretched Target Field (u_S)\n{S_upsampled}x{S_upsampled}', pad=10, fontsize=12)
    fig.colorbar(im2, ax=axes[1])

    # im3 = axes[2].imshow(predicted_reconstructed_sample, cmap='viridis', origin='lower', aspect='equal')
    im3 = axes[2].contourf(predicted_reconstructed_sample, cmap='viridis', origin='lower', levels=20)
    axes[2].set_title(f'Reconstructed Prediction (u\')\n{N_original}x{N_original}', pad=10, fontsize=12)
    fig.colorbar(im3, ax=axes[2])

    error = np.abs(u_orig_sample - predicted_reconstructed_sample)
    relative_error = error/ np.mean(np.abs(predicted_reconstructed_sample))
    im4 = axes[3].imshow(relative_error, cmap='hot', origin='lower')
    # axes[3].set_title('Absolute Error |u - u\'|')
    axes[3].set_title('Relative Error |u - u\'|/mean|u|')
    fig.colorbar(im4, ax=axes[3])
    
    # plt.tight_layout()
    plt.savefig('./draw/Freq_scale_test.png')
    
    # ==========================================================
    # 4. 额外：固定 y = y_idx，比较 x 方向剖面
    # ==========================================================
    # ① 选择一条横截线（y 方向下标），这里取原场中间那行
    y_idx_orig = N_original // 2           # 原网格行号
    y_idx_str  = y_idx_orig * M_factor     # 拉伸网格对应行号
    # y_idx_str  = y_idx_orig 
    
    # ② 取出三条曲线
    # x_orig      = np.arange(N_original)                 # 0,1,…,N-1
    # x_stretch   = np.linspace(0, N_original-1, S)       # S 个点均匀落在同一物理区间
    x_orig      = np.arange(0,S,M_factor)                 # 0,1,…,N-1
    x_stretch   = np.arange(0,  S)       # S 个点均匀落在同一物理区间
    # line_orig   = u_orig_sample[y_idx_orig, :]          # (N,)
    # line_stretch= u_target_stretched_sample[y_idx_str, :]  # (S,)
    # line_pred   = predicted_reconstructed_sample[y_idx_orig, :]  # (N,)


    line_orig   = u_orig_sample[:,y_idx_orig]          # (N,)
    line_stretch= u_target_stretched_sample[:,y_idx_str]  # (S,)
    line_pred   = predicted_reconstructed_sample[:,y_idx_orig]  # 
    
    # ③ 作图
    plt.figure(figsize=(9,4))
    plt.plot(x_stretch,  line_stretch,  '-',  lw=1.5,  color='k', label='Stretched (u_S)')
    plt.plot(x_orig,     line_orig,     'o',  ms=4,   label='Original u')
    plt.plot(x_orig,     line_pred,     'x',  ms=5,   label='Predicted u\'')
    plt.xlabel('x Directional Grid Index')
    plt.ylabel('Amplitude')
    plt.title(f'Satble y = {y_idx_orig} Profile comparison')
    plt.legend()
    plt.tight_layout()
    plt.savefig('./draw/profile_compare.png')
    
    
    field_fft = fft.rfft2(u_original[0, 0].cpu(),
                        #   norm='ortho'
                          )
    
    
    x_orig  = np.arange(32)
    
    plt.figure(figsize=(9,4))
    plt.plot(x_orig,  field_fft[:,0].real,  '-',  lw=1.5,  color='k', label='x=0')
    plt.xlabel('x Directional Grid Index')
    plt.ylabel('k')
    plt.title(f'x_k')
    plt.legend()
    plt.tight_layout()
    plt.savefig('./draw/x_k.png')