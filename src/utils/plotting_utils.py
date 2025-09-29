import yaml
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable

def load_config(config_path):
    """加载YAML配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def plot_from_config(data_matrix, config, output_path, title_text=None):
    """
    根据配置绘制单个矩阵的热力图（仅支持单图模式）。

    参数:
        data_matrix (np.ndarray): 2D 数据矩阵。
        config (dict): YAML 样式配置。
        output_path (str): 输出文件路径。
        title_text (str | None): 主脚本传入的标题文本。
    约束:
        - 当前实现只支持单图。如果未来需要网格布局，应另写封装函数。
    """
    cfg_fig = config.get('figure', {})
    cfg_title = config.get('title', {})
    cfg_axes = config.get('axes', {})
    cfg_cbar = config.get('colorbar', {})
    cfg_output = config.get('output', {})
    cfg_font = config.get('font', {})

    # --- 0. 设置字体 ---
    plt.rcParams['font.family'] = cfg_font.get('family', 'sans-serif')

    # --- 1. 创建图形和坐标轴 ---
    fig, ax = plt.subplots(figsize=cfg_fig.get('size', (5, 4)))
    
    # --- 2. 颜色范围计算 ---
    if cfg_cbar.get('adaptive_range', {}).get('enabled', True):
        percentiles = cfg_cbar.get('adaptive_range', {}).get('percentile', [0, 100])
        vmin = np.percentile(data_matrix, percentiles[0])
        vmax = np.percentile(data_matrix, percentiles[1])
        if vmax - vmin < 1e-9: # 避免范围过小
            vmin, vmax = np.min(data_matrix), np.max(data_matrix)
            if vmax - vmin < 1e-9:
                vmax = vmin + 1e-9
    else:
        fixed_range = cfg_cbar.get('fixed_range', {})
        vmin = fixed_range.get('vmin', 0)
        vmax = fixed_range.get('vmax', 1)

    # --- 3. 绘制热力图 ---
    cbar_mode = cfg_cbar.get('mode', 'normal')
    cbar_kws = {}
    cax = None

    if cbar_mode == 'professional':
        pos = cfg_cbar.get('professional', {}).get('position', [0.85, 0.15, 0.03, 0.7])
        cax = fig.add_axes(pos)
    else: # normal mode
        cbar_kws = cfg_cbar.get('normal', {})

    ax = sns.heatmap(data_matrix, ax=ax, cmap=config.get('colormap', 'viridis'), 
                vmin=vmin, vmax=vmax, 
                cbar=True, cbar_ax=cax, cbar_kws=cbar_kws,
                xticklabels=False, yticklabels=False)
    
    ax.set_aspect('equal') # 保持热力图为正方形

    # 添加四周黑色边框
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(0.5)

    # 获取 colorbar 对象
    cbar = ax.figure.axes[-1]

    # --- 4. 设置标题 ---
    if title_text:
        ax.set_title(title_text, fontsize=cfg_title.get('fontsize', 16), pad=cfg_title.get('pad', 10))

    # X轴
    cfg_xlabel = cfg_axes.get('xlabel', {})
    ax.set_xlabel(cfg_xlabel.get('text', ''), fontsize=cfg_xlabel.get('fontsize', 14), 
                  style=cfg_xlabel.get('style', 'normal'), labelpad=cfg_xlabel.get('pad', 5))
    if cfg_xlabel.get('position') == 'top':
        ax.xaxis.set_label_position('top')
        ax.xaxis.tick_top()

    # Y轴
    cfg_ylabel = cfg_axes.get('ylabel', {})
    ax.set_ylabel(cfg_ylabel.get('text', ''), fontsize=cfg_ylabel.get('fontsize', 14), 
                  style=cfg_ylabel.get('style', 'normal'), labelpad=cfg_ylabel.get('pad', 10),
                  rotation=cfg_ylabel.get('rotation', 90))

    # 刻度
    # ax.set_xticks([] if not cfg_axes.get('xticks', {}).get('visible', False) else None)
    # ax.set_yticks([] if not cfg_axes.get('yticks', {}).get('visible', False) else None)

    # --- 6. 配置颜色条 ---
    # cbar_mode = cfg_cbar.get('mode', 'normal')
    # if cbar_mode == 'professional':
    #     pos = cfg_cbar.get('professional', {}).get('position', [0.85, 0.15, 0.03, 0.7])
    #     cax = fig.add_axes(pos)
    #     cbar = plt.colorbar(im, cax=cax)
    # else: # normal mode
    #     cfg_normal = cfg_cbar.get('normal', {})
    #     cbar = plt.colorbar(im, ax=ax, shrink=cfg_normal.get('shrink', 1.0), 
    #                         aspect=cfg_normal.get('aspect', 20), pad=cfg_normal.get('pad', 0.05))


    cbar.tick_params(labelsize=cfg_cbar.get('label_fontsize', 12))
    
    # 刻度数量
    num_ticks = cfg_cbar.get('ticks', 3)
    tick_values = np.linspace(vmin, vmax, num_ticks)

    # 先固定刻度位置，防止后续被自动重计算
    cbar.set_yticks(tick_values)

    # 科学计数法测试: 零范围保护
    if vmin == vmax:
        tick_values = np.linspace(vmin, vmax + 1e-9, num_ticks)
        cbar.set_yticks(tick_values)

    # 科学计数法
    sci_cfg = cfg_cbar.get('scientific_notation', {})
    sci_enabled = sci_cfg.get('enabled', False)
    threshold_low = sci_cfg.get('threshold_low', 1e-3)
    threshold_high = sci_cfg.get('threshold_high', 1e4)
    decimals = int(sci_cfg.get('decimals', 2))
    exp_cfg = sci_cfg.get('exponent', {})

    use_scientific = False
    data_abs_max = max(abs(vmin), abs(vmax)) if not np.isnan(vmin) and not np.isnan(vmax) else 0
    if sci_enabled and data_abs_max != 0 and (data_abs_max < threshold_low or data_abs_max > threshold_high):
        use_scientific = True

    if use_scientific:
        # 计算公共指数（保证系数落在 0.x ~ 9.x 范围）
        # 小数位数由配置 scientific_notation.decimals 决定（与普通模式共享），不再强制为 1
        non_zero_vals = [abs(val) for val in tick_values if abs(val) > 0]
        if non_zero_vals:
            # 采用最大数量级，避免混合数量级时出现 10.x 的系数
            max_abs = max(non_zero_vals)
            common_exp = int(np.floor(np.log10(max_abs)))
        else:
            common_exp = 0
        scale = 10 ** common_exp

        # 初步计算系数
        coeffs = np.array(tick_values, dtype=float) / scale

        # 检查是否有因边界（比如 999 与 1001 混合）导致 coeff >= 10
        if np.any(np.abs(coeffs) >= 10):
            common_exp += 1
            scale *= 10
            coeffs = np.array(tick_values, dtype=float) / scale

        # 四舍五入后再次检查是否出现 10.0（例如 9.96 -> 10.0）
        rounded_coeffs = np.round(coeffs, decimals)
        if np.any(np.abs(rounded_coeffs) >= 10):
            # 统一再提升一次指数，重新计算，保证显示稳定
            common_exp += 1
            scale *= 10
            coeffs = np.array(tick_values, dtype=float) / scale
            rounded_coeffs = np.round(coeffs, decimals)

        # 生成标签字符串
        def fmt_coeff_from_array(idx):
            if abs(tick_values[idx]) == 0:
                return f"0.{''.join(['0']*decimals)}" if decimals > 0 else '0'
            return f"{rounded_coeffs[idx]:.{decimals}f}"
        labels = [fmt_coeff_from_array(i) for i in range(len(tick_values))]
        cbar.set_yticklabels(labels)
        exp_pos = exp_cfg.get('position', 'top')
        exp_x = exp_cfg.get('x', 0.5)
        exp_y = exp_cfg.get('y', 1.05)
        exp_fs = exp_cfg.get('fontsize', sci_cfg.get('fontsize', 12))
        if 'x' not in exp_cfg and 'y' not in exp_cfg:
            if exp_pos == 'top':
                exp_x, exp_y = 0.5, 1.05
            elif exp_pos == 'bottom':
                exp_x, exp_y = 0.5, -0.1
            elif exp_pos == 'right':
                exp_x, exp_y = 1.1, 0.5
            elif exp_pos == 'left':
                exp_x, exp_y = -0.2, 0.5
        cbar.text(exp_x, exp_y, f"×10$^{ {common_exp} }$".replace(' {','').replace(' }',''),
                  transform=cbar.transAxes, ha='center', va='center', fontsize=exp_fs)
    else:
        cbar.set_yticklabels([f"{val:.{decimals}f}" for val in tick_values])

    # 为 colorbar 添加黑色边框
    for spine in cbar.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(0.5)


    # --- 7. 调整布局和保存 ---
    plt.tight_layout(rect=[0, 0, cfg_fig.get('margins', {}).get('right', 0.85), 1])
    
    # 保存
    plt.savefig(output_path, 
                dpi=cfg_output.get('dpi', 300), 
                format=cfg_output.get('format', 'png'),
                bbox_inches=cfg_output.get('bbox_inches', 'tight'))
    plt.close(fig)
