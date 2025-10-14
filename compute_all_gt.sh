# 构建程序（如果不存在）
if [ ! -f "compute_gt" ]; then
    echo "构建 compute_gt..."
    g++ -std=c++17 -O3 -fopenmp -o compute_gt src/compute_gt.cpp
    if [ $? -ne 0 ]; then
        echo "编译失败，请检查编译器"
        exit 1
    fi
fi

datasets=('spacev10m' 'bigann10m')
base_path=/data/vector_datasets

echo "开始处理 2 个数据集..."

for dataset in "${datasets[@]}"
do 
    echo ""
    echo "=== 处理 $dataset ==="
    
    base_file="$base_path/$dataset/${dataset}_base.bvecs"
    query_file="$base_path/$dataset/${dataset}_query.bvecs"
    output_file="$base_path/$dataset/${dataset}_groundtruth.ivecs"
    
    if [[ -f "$base_file" && -f "$query_file" ]]; then
        echo "开始计算 $dataset groundtruth..."
        time ./compute_gt -b "$base_file" -q "$query_file" -o "$output_file" -k 100 -t 32
        
        if [ $? -eq 0 ]; then
            echo "✓ $dataset 完成"
        else
            echo "✗ $dataset 失败"
        fi
    else
        echo "⚠ $dataset 数据文件不存在"
    fi
done
