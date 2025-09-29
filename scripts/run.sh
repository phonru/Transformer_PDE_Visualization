#!/bin/bash
# 快速运行流水线脚本：按依赖顺序执行各分析与可视化步骤
# 可注释掉不运行的脚本，可按需要增添配置
set -e

DATASET='1'
SAMPLES='0'   # 示例: "0,5,10" 或 all

echo "运行配置: dataset=$DATASET samples=$SAMPLES"

# 将逗号分隔转为空格列表供部分脚本 narg='*' 解析
DATASET_IDS="$DATASET"
SAMPLE_IDS="$SAMPLES"

# 1.提取隐状态 (必需)
python src/process/extra_hidden_states.py \
    --processing_pairs ${DATASET}:${SAMPLES}

# 2.提取模型参数 (可选，供后续步骤7使用)
python src/process/extract_model_params.py

# 3.计算平均注意力
python src/process/calc_attn_mean.py \
    --dataset_ids $DATASET_IDS \
    --sample_ids $SAMPLE_IDS

# 4.绘制At和As矩阵, 依赖1,3
python src/visualization/plot_At_matrices.py \
    --dataset_ids $DATASET_IDS \
    --sample_ids $SAMPLE_IDS \
    --data_types both
    # 可选mean, head_mean, both
python src/visualization/plot_As_matrices.py \
    --dataset_ids $DATASET_IDS \
    --sample_ids $SAMPLE_IDS \
    --data_types both
    # 可选mean, head_mean, both

# 5.计算重塑空间注意力矩阵，依赖1,3
python src/process/calc_spatial_attention_reshaped.py \
    --dataset_ids $DATASET_IDS \
    --sample_ids $SAMPLE_IDS \
    --data_types both \
    --process_all_layers

# 6.绘制重塑空间注意力矩阵，依赖5
python src/visualization/plot_spatial_attention_reshaped.py \
    --dataset_ids $DATASET_IDS \
    --sample_ids $SAMPLE_IDS \
    --data_types both \
    --plot_mode single_row
    # 可选single_row, single_col, comparison
python src/visualization/plot_spatial_reshaped_with_contour.py \
    --dataset-id $DATASET_IDS \
    --sample-id $SAMPLE_IDS \
    --reshape-type both \
    --draw-contour \
    # --no-contour

# 核函数部分
# 7.处理隐状态生成中间变量，依赖1,2
python src/process/process_hidden_states.py \
    --datasets $DATASET_IDS \
    --samples $SAMPLE_IDS

# 8.计算简化核函数 (K)，依赖7
python src/process/calc_K_t_prime_p_prime.py \
    --datasets $DATASET_IDS \
    --samples $SAMPLE_IDS \
    --global-seed 42

# 9.绘制核函数，依赖8
python src/visualization/plot_K_t_prime_p_prime.py \
    --datasets $DATASET_IDS \
    --samples $SAMPLE_IDS \
    --layers all

