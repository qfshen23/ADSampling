#!/bin/bash

# 批量归一化多个数据集
# 使用方法: bash batch_normalize.sh

# 数据集列表（根据你的需求修改）
datasets=(
    "glove2m"
    "word2vec"
)

# 配置参数
DATA_DIR="/data/vector_datasets"
FILES="base query"  # 要归一化的文件类型

echo "开始批量归一化数据集..."
echo "========================================================================"
echo "数据目录: $DATA_DIR"
echo "处理文件: $FILES"
echo "========================================================================"

success_count=0
fail_count=0

for dataset in "${datasets[@]}"; do
    echo ""
    echo "========================================================================"
    echo "处理数据集: $dataset"
    echo "========================================================================"
    
    # 检查数据集目录是否存在
    dataset_path="$DATA_DIR/$dataset"
    if [ ! -d "$dataset_path" ]; then
        echo "❌ 错误: 数据集目录不存在: $dataset_path"
        echo "跳过 $dataset"
        ((fail_count++))
        continue
    fi
    
    # 检查是否已经存在归一化后的目录
    normalized_path="$DATA_DIR/${dataset}_normalized"
    if [ -d "$normalized_path" ]; then
        echo "⚠️  警告: 归一化目录已存在: $normalized_path"
        read -p "是否覆盖? (y/n): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "跳过 $dataset"
            ((fail_count++))
            continue
        fi
    fi
    
    # 执行归一化
    python normalized.py \
        --datasets "$dataset" \
        --source "$DATA_DIR" \
        --files $FILES
    
    if [ $? -eq 0 ]; then
        echo "✅ $dataset 归一化完成"
        ((success_count++))
    else
        echo "❌ $dataset 归一化失败"
        ((fail_count++))
    fi
    
    echo "------------------------------------------------------------------------"
done

echo ""
echo "========================================================================"
echo "📊 批量归一化完成!"
echo "========================================================================"
echo "✅ 成功: $success_count 个数据集"
echo "❌ 失败: $fail_count 个数据集"
echo "========================================================================"

