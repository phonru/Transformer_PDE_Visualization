# 🚀 Transformer PDE Visualization
该项目用于对基于 Transformer 的 PDE（偏微分方程）建模实验产生的隐藏状态、注意力矩阵及其衍生核函数进行可视化。
> 如果你已熟悉项目，可直接跳到“快速开始”。

## 🧩 术语 / 符号表

| 名称 / 符号 | 说明 |
|-------------|------|
| At | 时间维度注意力矩阵 |
| As | 空间维度注意力矩阵 |
| As_reshaped | 按空间拓扑或排列重塑后的注意力矩阵，提升观察空间结构特征可读性 |
| HS_p | 处理中间特征 |
| K(t', p') | 简化核函数 |

## 🧪 环境需求

- Python = 3.12
依赖见 `requirements.txt`

## 🗂 数据目录约定

```
data/
	hidden_states/
		dataset_<id>/
			sample_<k>.npz        # 原始/抽取的隐藏状态
	processed/
		A_mean/                # 计算得到的平均注意力 (At / As)
		As_reshaped/           # 重排空间注意力
		HS_p/                  # 中间特征（供 K 或后续使用）
		K_simplified/          # 简化核函数及其分量输出
	weight/
		attn/                  # 注意力权重（如有提取）
		LN/                    # LayerNorm 参数（如提取）
```

一键创建（如为空项目初次使用）：
```bash
mkdir -p data/hidden_states \
	data/processed/{A_mean,As_reshaped,HS_p,K_simplified} \
	data/weight/{attn,LN}
```

## 🚀 快速开始

1. 全流程（注意力 + 核函数 + 可视化）：
```bash
bash scripts/run.sh
```

2. 数据集可视化：
```bash
bash scripts/run_dataset_visualization.sh
```

3. 多 checkpoint 批处理：
```bash
bash scripts/run_checkpoints.sh
```
4. `src/process/`，`src/visualization/`下的脚本均可单独运行，见下。

## 🧱 单独处理脚本（`src/process/`）

| 脚本 | 功能 | 主要输入 | 输出目录 |
|------|------|----------|----------|
| `extra_hidden_states.py` | 从模型 / checkpoint 抽取隐藏状态 | 模型权重 | `data/hidden_states/` |
| `process_hidden_states.py` | 生成核函数/后续步骤需要的中间张量 | hidden_states | `data/processed/HS_p/` |
| `calc_attn_mean.py` | 计算多层/多头注意力平均 (At / As) | hidden_states | `data/processed/A_mean/` |
| `calc_spatial_attention_reshaped.py` | 生成重排空间注意力矩阵 | A_mean / HS_p | `data/processed/As_reshaped/` |
| `calc_K_t_prime_p_prime.py` | 计算简化核函数 K(t', p') | HS_p / As | `data/processed/K_simplified/` |
| `extract_model_params.py` | 提取注意力权重 / LayerNorm 参数 | 模型权重 | `data/weight/` |

## 🎨 可视化脚本（`src/visualization/`）

示例：
```bash
python -m src.visualization.plot_As_matrices --config config/plot_As_config.yaml
```
其它：
- `plot_At_matrices.py`
- `plot_spatial_attention_reshaped.py`
- `plot_spatial_reshaped_with_contour.py`
- `plot_K_t_prime_p_prime.py`

## 📤 输出目录 (`output/`)

| 目录 | 内容 |
|------|------|
| `plots_As/` | 空间注意力（或其平均）图像 |
| `plots_At/` | 时间注意力图像 |
| `plots_As_reshaped/` | 重排空间注意力矩阵图像 |
| `plots_As_reshaped_with_contour/` | 带等高线的重排注意力图像 |
| `plots_K_simplified/` | 简化核函数及分量图像 |
| `plots_datasets/` | 原始数据序列可视化（全时序 / 单通道） |
| `plots_pre/` | 预测结果可视化 |


