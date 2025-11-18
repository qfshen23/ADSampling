#!/bin/bash

# 批量计算多个数据集的groundtruth
# 使用方法: bash batch_compute_gt.sh

# 数据集列表（根据你的需求修改）
datasets=(
    "word2vec_normalized"
    "glove2m_normalized"
    "openai1536"
    "openai3072"
)

# 配置参数
DATA_DIR="/data/vector_datasets"
K=100  # 计算top-k最近邻
METRIC="IP"  # 距离度量: L2 或 IP

# 是否使用GPU（如果可用）
# USE_GPU="--gpu"  # 如果不想使用GPU，注释掉这一行或设置为空字符串

echo "开始批量计算groundtruth..."
echo "================================"

for dataset in "${datasets[@]}"; do
    echo ""
    echo "处理数据集: $dataset"
    echo "--------------------------------"
    
    # 检查数据集目录是否存在
    dataset_path="$DATA_DIR/$dataset"
    if [ ! -d "$dataset_path" ]; then
        echo "警告: 数据集目录不存在: $dataset_path"
        echo "跳过 $dataset"
        continue
    fi
    
    # 检查是否已经存在groundtruth文件
    gt_file="$dataset_path/${dataset}_groundtruth.ivecs"
    if [ -f "$gt_file" ]; then
        echo "警告: groundtruth文件已存在: $gt_file"
        read -p "是否覆盖? (y/n): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "跳过 $dataset"
            continue
        fi
    fi
    
    # 执行计算
    python compute_groundtruth.py \
        --dataset "$dataset" \
        --data_dir "$DATA_DIR" \
        --k $K \
        --metric "$METRIC" \
        $USE_GPU
    
    if [ $? -eq 0 ]; then
        echo "✓ $dataset 完成"
    else
        echo "✗ $dataset 失败"
    fi
    
    echo "--------------------------------"
done

echo ""
echo "================================"
echo "批量处理完成！"

