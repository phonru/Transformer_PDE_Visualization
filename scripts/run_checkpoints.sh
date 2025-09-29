#!/bin/bash

# 用于运行多个检查点的注意力分析脚本，支持核函数计算和绘制
#
# 功能:
#   - 提取隐藏状态
#   - 计算注意力平均值
#   - 绘制注意力矩阵
#   - 计算和绘制重新排列的空间注意力矩阵 (分离式执行)
#   - 从检查点提取模型参数 (注意力层和LayerNorm层)
#   - 计算核函数 (可选)
#   - 绘制核函数热力图 (可选)
#   - 支持多个随机种子并行计算

set -e  # 遇到错误时停止

# =============================================================================
# 配置参数 - 可根据需要修改
# =============================================================================

# 基础路径配置
CHECKPOINT_DIR="/data/1/pkq2/checkpoints/checkpoint_per10ep/seed_init/checkpoints/"
OUTPUT_BASE_DIR="A_with_checkpoints"
# EPOCHS=(initial 0 10 20 30 40 50 60 70 80 90 100 110 120 130 140)
EPOCHS=(10) # 快速测试用

# 数据集和样本配置 - 只处理一个数据集
DATASET="0"  # 主要处理的数据集
# SAMPLES="0, 25, 50, 75, 100, 125, 190, 200, 220"  # 该数据集的样本
SAMPLES="100"  # 快速测试用

# 步骤控制开关 - 每个步骤都有独立的开关
STEP1_EXTRACT_HIDDEN_STATES=true      # 步骤1: 提取隐藏状态
STEP2_CALC_ATTENTION_MEAN=true        # 步骤2: 计算注意力平均值
STEP3_PLOT_SPATIAL_ATTENTION=true    # 步骤3: 绘制空间注意力矩阵
STEP4_PLOT_TEMPORAL_ATTENTION=true   # 步骤4: 绘制时间注意力矩阵
STEP5_CALC_RESHAPED_ATTENTION=true    # 步骤5: 计算重新排列的空间注意力矩阵
STEP6_PLOT_RESHAPED_ATTENTION=true    # 步骤6: 绘制重新排列的空间注意力矩阵
STEP7_EXTRACT_MODEL_PARAMS=true       # 步骤7: 从检查点提取模型参数
STEP8_PROCESS_HIDDEN_STATES=true      # 步骤8: 处理隐状态并计算权重 (原步骤9)
STEP9_CALC_KERNEL=true                # 步骤9: 计算核函数张量 (原步骤10)
STEP10_PLOT_KERNEL=true               # 步骤10: 绘制核函数热力图 (原步骤11)

# 每个步骤对应的脚本
STEP1_SCRIPT="src/process/extra_hidden_states.py"
STEP2_SCRIPT="src/process/calc_attn_mean.py"
STEP3_SCRIPT="src/visualization/plot_As_matrices.py"
STEP4_SCRIPT="src/visualization/plot_At_matrices.py"
STEP5_SCRIPT="src/process/calc_spatial_attention_reshaped.py"
STEP6_SCRIPT="src/visualization/plot_spatial_attention_reshaped.py"
STEP7_SCRIPT="src/process/extract_model_params.py"
STEP8_SCRIPT="src/process/process_hidden_states.py"
STEP9_SCRIPT="src/process/calc_K_t_prime_p_prime.py"
STEP10_SCRIPT="src/visualization/plot_K_t_prime_p_prime.py"

# 随机种子配置 - 支持多个种子并行计算
RANDOM_SEEDS=(3847 1234 5678 9999 42 3456 2490 8702 3867 1099 9736 2803 5869 7742 2753 8204 3662 799 843 123)  # 多个随机种子
# RANDOM_SEEDS=(3847)                   # 单个种子的例子
# RANDOM_SEEDS=($(seq 1000 100 1900))   # 生成连续种子 1000,1100,...,1900

# 核函数计算的具体参数
KERNEL_LAYERS="all"      # 要处理的层："all", "0", "0,1,2", "0-5"
KERNEL_D_FIXED=""        # 固定特征维度d (留空表示随机)
KERNEL_D_PRIME_FIXED=""  # 固定特征维度d' (留空表示随机)
KERNEL_T_FIXED=""        # 固定时间索引t (留空表示随机)
KERNEL_P_FIXED=""        # 固定空间索引p (留空表示随机)

# 绘图配置
PLOT_TIME_STEPS="all"    # 绘图的时间步："all", "0", "0,1,2", "0-5"
PLOT_SPACE_STEPS="all"   # 绘图的空间步："all", "0", "0,1,2", "0-5"

# =============================================================================
# =============================================================================

echo "开始处理时间: $(date)"
# 检查必需的脚本文件是否存在
check_scripts() {
    local missing_scripts=()
    
    # 检查各步骤所需的脚本文件
    [ "$STEP1_EXTRACT_HIDDEN_STATES" = true ] && [ ! -f "$STEP1_SCRIPT" ] && missing_scripts+=("$STEP1_SCRIPT")
    [ "$STEP2_CALC_ATTENTION_MEAN" = true ] && [ ! -f "$STEP2_SCRIPT" ] && missing_scripts+=("$STEP2_SCRIPT")
    [ "$STEP3_PLOT_SPATIAL_ATTENTION" = true ] && [ ! -f "$STEP3_SCRIPT" ] && missing_scripts+=("$STEP3_SCRIPT")
    [ "$STEP4_PLOT_TEMPORAL_ATTENTION" = true ] && [ ! -f "$STEP4_SCRIPT" ] && missing_scripts+=("$STEP4_SCRIPT")
    [ "$STEP5_CALC_RESHAPED_ATTENTION" = true ] && [ ! -f "$STEP5_SCRIPT" ] && missing_scripts+=("$STEP5_SCRIPT")
    [ "$STEP6_PLOT_RESHAPED_ATTENTION" = true ] && [ ! -f "$STEP6_SCRIPT" ] && missing_scripts+=("$STEP6_SCRIPT")
    [ "$STEP7_EXTRACT_MODEL_PARAMS" = true ] && [ ! -f "$STEP7_SCRIPT" ] && missing_scripts+=("$STEP7_SCRIPT")
    [ "$STEP8_PROCESS_HIDDEN_STATES" = true ] && [ ! -f "$STEP8_SCRIPT" ] && missing_scripts+=("$STEP8_SCRIPT")
    [ "$STEP9_CALC_KERNEL" = true ] && [ ! -f "$STEP9_SCRIPT" ] && missing_scripts+=("$STEP9_SCRIPT")
    [ "$STEP10_PLOT_KERNEL" = true ] && [ ! -f "$STEP10_SCRIPT" ] && missing_scripts+=("$STEP10_SCRIPT")
    
    if [ ${#missing_scripts[@]} -gt 0 ]; then
        echo "错误: 以下必需的脚本文件不存在:"
        for script in "${missing_scripts[@]}"; do
            echo "  - $script"
        done
        echo "请确保所有脚本文件都在正确的位置。"
        exit 1
    fi
    
    echo "所有必需的脚本文件检查完成"
}

echo "开始处理多个检查点的注意力分析..."
echo "检查点目录: $CHECKPOINT_DIR"
echo "输出基础目录: $OUTPUT_BASE_DIR"
echo "将处理的epochs: ${EPOCHS[@]}"
echo "处理数据集: $DATASET"
echo "样本: $SAMPLES"
echo ""
echo "步骤控制开关:"
echo "  步骤1  - 提取隐藏状态 ($STEP1_EXTRACT_HIDDEN_STATES): $STEP1_SCRIPT"
echo "  步骤2  - 计算注意力平均值 ($STEP2_CALC_ATTENTION_MEAN): $STEP2_SCRIPT"
echo "  步骤3  - 绘制空间注意力矩阵 ($STEP3_PLOT_SPATIAL_ATTENTION): $STEP3_SCRIPT"
echo "  步骤4  - 绘制时间注意力矩阵 ($STEP4_PLOT_TEMPORAL_ATTENTION): $STEP4_SCRIPT"
echo "  步骤5  - 计算重排空间注意力矩阵 ($STEP5_CALC_RESHAPED_ATTENTION): $STEP5_SCRIPT"
echo "  步骤6  - 绘制重排空间注意力矩阵 ($STEP6_PLOT_RESHAPED_ATTENTION): $STEP6_SCRIPT"
echo "  步骤7  - 提取模型参数 ($STEP7_EXTRACT_MODEL_PARAMS): $STEP7_SCRIPT"
echo "  步骤8  - 处理隐状态并计算权重 ($STEP8_PROCESS_HIDDEN_STATES): $STEP8_SCRIPT"
echo "  步骤9  - 计算核函数张量 ($STEP9_CALC_KERNEL): $STEP9_SCRIPT"
echo "  步骤10 - 绘制核函数热力图 ($STEP10_PLOT_KERNEL): $STEP10_SCRIPT"
echo ""
if [ "$STEP9_CALC_KERNEL" = true ] || [ "$STEP10_PLOT_KERNEL" = true ]; then
    echo "核函数计算参数:"
    echo "  随机种子: ${RANDOM_SEEDS[@]}"
    echo "  层: $KERNEL_LAYERS"
    echo "  固定维度 d: ${KERNEL_D_FIXED:-随机}"
    echo "  固定维度 d': ${KERNEL_D_PRIME_FIXED:-随机}"
    echo "  固定时间索引 t: ${KERNEL_T_FIXED:-随机}"
    echo "  固定空间索引 p: ${KERNEL_P_FIXED:-随机}"
    echo "绘图参数:"
    echo "  时间步: $PLOT_TIME_STEPS"
    echo "  空间步: $PLOT_SPACE_STEPS"
fi
echo ""

# 检查脚本文件
check_scripts

# 计算总任务数用于进度显示
total_epochs=${#EPOCHS[@]}
total_seeds=${#RANDOM_SEEDS[@]}
current_epoch=0

echo ""

# 为每个epoch处理
echo "开始进入epoch循环..."
for epoch in "${EPOCHS[@]}"; do
    current_epoch=$((current_epoch + 1))
    echo ""
    echo "============================================================"
    echo "处理 Epoch $epoch ($current_epoch/$total_epochs)"
    if [ "$PROCESS_KERNEL" = true ]; then
        echo "本epoch将处理 ${total_seeds} 个随机种子的核函数计算"
    fi
    echo "============================================================"
    
    CHECKPOINT_PATH="$CHECKPOINT_DIR/checkpoint_epoch_${epoch}.tar"
    
    # 检查检查点文件是否存在
    if [ ! -f "$CHECKPOINT_PATH" ]; then
        echo "警告: 检查点文件不存在: $CHECKPOINT_PATH"
        continue
    fi
    
    # 设置输出目录
    HIDDEN_STATES_DIR="$OUTPUT_BASE_DIR/hidden_states/ep${epoch}"
    PROCESS_DIR="$OUTPUT_BASE_DIR/process/ep${epoch}"
    
    echo "检查点文件: $CHECKPOINT_PATH"
    echo "隐藏状态目录: $HIDDEN_STATES_DIR"
    echo "处理结果目录: $PROCESS_DIR"
    echo ""
    
    # 步骤1: 提取隐藏状态
    if [ "$STEP1_EXTRACT_HIDDEN_STATES" = true ]; then
        echo "步骤1: 提取隐藏状态..."
        
        # 构建processing_pairs参数，格式 dataset_idx:sample_indices
        if [ "$SAMPLES" = "all" ]; then
            PROCESSING_PAIRS="${DATASET}:all"
        else
            PROCESSING_PAIRS="${DATASET}:${SAMPLES// /}"
        fi
        echo "  使用processing_pairs: $PROCESSING_PAIRS"
        python "$STEP1_SCRIPT" \
            --checkpoint_path "$CHECKPOINT_PATH" \
            --hidden_states_dir "$HIDDEN_STATES_DIR" \
            --processing_pairs "$PROCESSING_PAIRS"
        
        if [ $? -ne 0 ]; then
            echo "错误: 提取隐藏状态失败 (epoch $epoch)"
            continue
        fi
        echo "步骤1完成"
    else
        echo " 跳过步骤1: 提取隐藏状态"
    fi
    
    # 步骤2: 计算注意力平均值
    if [ "$STEP2_CALC_ATTENTION_MEAN" = true ]; then
        echo "步骤2: 计算注意力平均值..."
        
        echo "  处理数据集 $DATASET (样本: $SAMPLES)..."
        python "$STEP2_SCRIPT" \
            --hidden_states_dir "$HIDDEN_STATES_DIR" \
            --output_dir "$PROCESS_DIR/A_mean/data" \
            --dataset_ids "$DATASET" \
            --sample_ids "$SAMPLES"
        
        if [ $? -ne 0 ]; then
            echo "错误: 数据集 $DATASET 计算注意力平均值失败 (epoch $epoch)"
            continue
        fi
        echo "步骤2完成"
    else
        echo " 跳过步骤2: 计算注意力平均值"
    fi
    
    # 步骤3: 绘制空间注意力矩阵
    if [ "$STEP3_PLOT_SPATIAL_ATTENTION" = true ]; then
        echo "步骤3: 绘制空间注意力矩阵..."
        
        echo "  处理数据集 $DATASET..."
        python "$STEP3_SCRIPT" \
            --data_dir "$PROCESS_DIR/A_mean/data" \
            --output_dir "$PROCESS_DIR/A_mean/plots_Ap" \
            --dataset_ids "$DATASET" \
            --sample_ids "$SAMPLES" \
            --data_types both
        
        if [ $? -ne 0 ]; then
            echo "错误: 数据集 $DATASET 绘制空间注意力矩阵失败 (epoch $epoch)"
            continue
        fi
        echo "步骤3完成"
    else
        echo " 跳过步骤3: 绘制空间注意力矩阵"
    fi
    
    # 步骤4: 绘制时间注意力矩阵
    if [ "$STEP4_PLOT_TEMPORAL_ATTENTION" = true ]; then
        echo "步骤4: 绘制时间注意力矩阵..."
        
        echo "  处理数据集 $DATASET..."
        python "$STEP4_SCRIPT" \
            --data_dir "$PROCESS_DIR/A_mean/data" \
            --output_dir "$PROCESS_DIR/A_mean/plots_At" \
            --dataset_ids "$DATASET" \
            --sample_ids "$SAMPLES" \
            --data_types both
        
        if [ $? -ne 0 ]; then
            echo "错误: 数据集 $DATASET 绘制时间注意力矩阵失败 (epoch $epoch)"
            continue
        fi
        echo "步骤4完成"
    else
        echo " 跳过步骤4: 绘制时间注意力矩阵"
    fi
    
    # 步骤5: 计算重新排列的空间注意力矩阵
    if [ "$STEP5_CALC_RESHAPED_ATTENTION" = true ]; then
        echo "步骤5: 计算重新排列的空间注意力矩阵..."
        echo "  处理数据集 $DATASET..."
        python "$STEP5_SCRIPT" \
            --data_dir "$PROCESS_DIR/A_mean/data" \
            --output_dir "$PROCESS_DIR/A_mean/data_spatial_reshaped" \
            --dataset_ids "$DATASET" \
            --sample_ids "$SAMPLES" \
            --data_types both
        
        if [ $? -ne 0 ]; then
            echo "错误: 数据集 $DATASET 计算重新排列的空间注意力矩阵失败 (epoch $epoch)"
            continue
        fi
        echo "步骤5完成"
    else
        echo " 跳过步骤5: 计算重新排列的空间注意力矩阵"
    fi
    
    # 步骤6: 绘制重新排列的空间注意力矩阵
    if [ "$STEP6_PLOT_RESHAPED_ATTENTION" = true ]; then
        echo "步骤6: 绘制重新排列的空间注意力矩阵..."
        echo "  处理数据集 $DATASET..."
        python "$STEP6_SCRIPT" \
            --reshaped_data_dir "$PROCESS_DIR/A_mean/data_spatial_reshaped" \
            --output_dir "$PROCESS_DIR/A_mean/plots_spatial_reshaped" \
            --dataset_ids "$DATASET" \
            --sample_ids "$SAMPLES" \
            --plot_mode single_row \
            --data_types mean
        
        if [ $? -ne 0 ]; then
            echo "错误: 数据集 $DATASET 绘制重新排列的空间注意力矩阵失败 (epoch $epoch)"
            continue
        fi
        echo "步骤6完成"
    else
        echo " 跳过步骤6: 绘制重新排列的空间注意力矩阵"
    fi
    
    # 步骤7: 从当前检查点提取模型参数 (注意力和LayerNorm)
    if [ "$STEP7_EXTRACT_MODEL_PARAMS" = true ]; then
        echo "步骤7: 从检查点提取模型参数..."
        WEIGHTS_DIR="$PROCESS_DIR/weights"
        python "$STEP7_SCRIPT" \
            --checkpoint_path "$CHECKPOINT_PATH" \
            --save_dir "$WEIGHTS_DIR" \
            --extract_all \
            --attention_subdir "attn" \
            --layernorm_subdir "LN" \
            --verbose
        
        if [ $? -ne 0 ]; then
            echo "错误: 提取模型参数失败 (epoch $epoch)"
            continue
        fi
        echo "步骤7完成 (参数保存到: $WEIGHTS_DIR)"
    else
        echo " 跳过步骤7: 从检查点提取模型参数"
    fi
    
    # 步骤8: 处理隐状态并计算权重
    if [ "$STEP8_PROCESS_HIDDEN_STATES" = true ]; then
        echo "步骤8: 处理隐状态并计算权重..."
        
        # 确保权重目录存在
        WEIGHTS_DIR="$PROCESS_DIR/weights"
        if [ ! -d "$WEIGHTS_DIR/attn" ] || [ ! -d "$WEIGHTS_DIR/LN" ]; then
            echo "错误: 权重目录不存在，请先执行步骤7"
        else
            python "$STEP8_SCRIPT" \
                --hidden-states-dirs "$HIDDEN_STATES_DIR" \
                --output-dir "$PROCESS_DIR/HS_p/data" \
                --datasets "$DATASET" \
                --samples "$SAMPLES" \
                --attention-weights "$WEIGHTS_DIR/attn" \
                --layer-norm-weights "$WEIGHTS_DIR/LN"
            
            if [ $? -ne 0 ]; then
                echo "错误: 处理隐状态失败 (epoch $epoch)"
            else
                echo "步骤8完成"
            fi
        fi
    else
        echo " 跳过步骤8: 处理隐状态并计算权重"
    fi
    
    # 步骤9和10需要随机种子循环
    if [ "$STEP9_CALC_KERNEL" = true ] || [ "$STEP10_PLOT_KERNEL" = true ]; then
        echo "============================================================"
        echo "开始核函数计算和绘制 (Epoch $epoch) - 步骤9-10"
        echo "============================================================"
        
        # 为每个随机种子计算核函数
        current_seed=0
        for seed in "${RANDOM_SEEDS[@]}"; do
            current_seed=$((current_seed + 1))
            echo "------------------------------------------------------------"
            echo "处理随机种子: $seed (Epoch $epoch, 种子 $current_seed/$total_seeds)"
            echo "------------------------------------------------------------"
            
            # 步骤9: 计算核函数张量
            if [ "$STEP9_CALC_KERNEL" = true ]; then
                echo "步骤9: 计算核函数张量 (种子: $seed)..."
            
                # 检查输入数据是否存在
                if [ ! -d "$PROCESS_DIR/HS_p/data" ]; then
                    echo "错误: 输入数据目录不存在，请先执行步骤9"
                else
                    # 构建calc_K_ts_tp_2.py的参数
                    calc_k_args=(
                        --input-dir "$PROCESS_DIR/HS_p/data"
                        --output-dir "$PROCESS_DIR/K_simplified/data_${seed}"
                        --global-seed "$seed"
                        --datasets "$DATASET"
                        --samples "$SAMPLES"
                    )
                    
                    # 添加固定维度参数（如果设置了）
                    if [ -n "$KERNEL_D_FIXED" ]; then
                        calc_k_args+=(--d-fixed "$KERNEL_D_FIXED")
                    fi
                    if [ -n "$KERNEL_D_PRIME_FIXED" ]; then
                        calc_k_args+=(--d-prime-fixed "$KERNEL_D_PRIME_FIXED")
                    fi
                    if [ -n "$KERNEL_T_FIXED" ]; then
                        calc_k_args+=(--t-fixed "$KERNEL_T_FIXED")
                    fi
                    if [ -n "$KERNEL_P_FIXED" ]; then
                        calc_k_args+=(--p-fixed "$KERNEL_P_FIXED")
                    fi
                    
                    python "$STEP9_SCRIPT" "${calc_k_args[@]}"
                    
                    if [ $? -ne 0 ]; then
                        echo "错误: 计算核函数张量失败 (epoch $epoch, seed $seed)"
                    else
                        echo "步骤9完成 (种子: $seed)"
                    fi
                fi
            else
                echo "跳过步骤9: 计算核函数张量 (种子: $seed)"
            fi
            
            # 步骤10: 绘制核函数热力图
            if [ "$STEP10_PLOT_KERNEL" = true ]; then
                echo "步骤10: 绘制核函数热力图 (种子: $seed)..."
                
                # 检查输入数据是否存在
                if [ ! -d "$PROCESS_DIR/K_simplified/data_${seed}" ]; then
                    echo "错误: 核函数数据目录不存在，请先执行步骤10"
                else
                    # 构建plot_K_combined.py的参数
                    plot_k_args=(
                        --data-dir "$PROCESS_DIR/K_simplified/data_${seed}"
                        --output-dir "$PROCESS_DIR/K_simplified/plots_${seed}"
                        --datasets "$DATASET"
                        --samples "$SAMPLES"
                        --layers "$KERNEL_LAYERS"
                        --time-steps "$PLOT_TIME_STEPS"
                        --space-steps "$PLOT_SPACE_STEPS"
                        --plot-main
                        --plot-terms
                    )
                    
                    python "$STEP10_SCRIPT" "${plot_k_args[@]}"
                    
                    if [ $? -ne 0 ]; then
                        echo "警告: 绘制核函数热力图失败 (epoch $epoch, seed $seed)"
                    else
                        echo "步骤10完成 (种子: $seed)"
                    fi
                fi
            else
                echo "跳过步骤10: 绘制核函数热力图 (种子: $seed)"
            fi
            
            echo "随机种子 $seed 处理完成 (Epoch $epoch)"
        done
        
        echo "核函数处理完成 (Epoch $epoch)"
    else
        echo "跳过核函数处理 (步骤9-10)"
    fi
    
    echo "Epoch $epoch 处理完成!"
    echo ""
done

echo "所有检查点处理完成!"
echo "结果保存在: $OUTPUT_BASE_DIR"
echo "处理完成时间: $(date)"
