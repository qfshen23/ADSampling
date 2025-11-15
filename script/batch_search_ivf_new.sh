#!/bin/bash

# 批量测试脚本 - search_ivf_new.sh
# 用法: ./batch_search_ivf_new.sh

cd ..
# 编译
echo "正在编译 search_ivf..."
g++ ./src/search_ivf.cpp -O3 -mavx -mavx512vpopcntdq -g -o ./src/search_ivf -I ./src/ -I /usr/include/eigen3 -fopenmp
if [ $? -ne 0 ]; then
    echo "编译失败!"
    exit 1
fi
echo "编译成功!"

path=/data/vector_datasets
index_path=/data/tmp/ivf
result_path=./results

# 确保结果目录存在
mkdir -p ${result_path}

K=10
k_overlap=64
randomize=0

# 定义数据集及其对应的参数
# 格式: 数据集名称 C值 CC值 ACTUAL_C值 test_params
declare -a configs=(
    "msong 1024 1024 1024 10:400,15:1500,20:2800,25:3500,30:5000,35:7500,40:9000,45:10000,50:12000,60:14000,80:15000"
    # SIFT 数据集配置
    # "sift 1024 256 256 10:400,15:1500,20:2800,25:3500,30:5000,35:7500,40:9000,45:10000,50:12000,60:14000,80:15000"
    # "sift 2048 512 512 10:400,15:1500,20:2800,25:3500,30:5000,35:7500,40:9000,45:10000,50:12000,60:14000,80:15000"
    # "sift 4096 1024 1024 10:200,15:800,20:1400,25:1800,30:2500,35:3800,40:4500,45:5000,50:6000,60:7000,80:7500"
    
    # # SIFT10M 数据集配置
    # "sift10m 2048 512 512 10:400,15:1500,20:2800,25:3500,30:5000,35:7500,40:9000,45:10000,50:12000,60:14000,80:15000"
    # "sift10m 4096 1024 1024 10:200,15:800,20:1400,25:1800,30:2500,35:3800,40:4500,45:5000,50:6000,60:7000,80:7500"
    
    # # GIST 数据集配置
    # "gist 1024 256 256 10:500,15:2000,20:3500,25:4500,30:6000,35:9000,40:11000,45:12000,50:14000,60:17000,80:20000"
    # "gist 2048 512 512 10:500,15:2000,20:3500,25:4500,30:6000,35:9000,40:11000,45:12000,50:14000,60:17000,80:20000"
    
    # # TINY5M 数据集配置
    # "tiny5m 2048 512 512 10:400,15:1500,20:2800,25:3500,30:5000,35:7500,40:9000,45:10000,50:12000,60:14000,80:15000"
    # "tiny5m 4096 1024 1024 10:200,15:800,20:1400,25:1800,30:2500,35:3800,40:4500,45:5000,50:6000,60:7000,80:7500"
)

echo "======================================"
echo "开始批量测试"
echo "======================================"
echo ""

# 遍历所有配置
for config in "${configs[@]}"
do
    # 解析配置
    read -r data C CC ACTUAL_C test_params <<< "$config"
    
    echo "======================================"
    echo "测试配置:"
    echo "  数据集: ${data}"
    echo "  C: ${C}"
    echo "  CC: ${CC}"
    echo "  ACTUAL_C: ${ACTUAL_C}"
    echo "  K: ${K}"
    echo "  k_overlap: ${k_overlap}"
    echo "  test_params: ${test_params}"
    echo "======================================"
    
    # 设置文件路径
    res="${result_path}/${data}_IVF${C}_${randomize}.log"
    index="${index_path}/${data}/${data}_ivf_${C}_${randomize}.index"
    query="${path}/${data}/${data}_query.fvecs"
    gnd="${path}/${data}/${data}_groundtruth.ivecs"
    trans="${path}/${data}/O.fvecs"
    diskK="${result_path}/${data}_IVF${C}_${randomize}_diskK.log"
    clusters_file="${path}/${data}/${data}_top_clusters_${ACTUAL_C}_of_${C}.ivecs"
    top_centroids_file="${path}/${data}/${data}_centroid_${C}.fvecs"
    
    # 检查必要文件是否存在
    if [ ! -f "${index}" ]; then
        echo "警告: 索引文件不存在: ${index}"
        echo "跳过此配置..."
        echo ""
        continue
    fi
    
    if [ ! -f "${query}" ]; then
        echo "警告: 查询文件不存在: ${query}"
        echo "跳过此配置..."
        echo ""
        continue
    fi
    
    if [ ! -f "${gnd}" ]; then
        echo "警告: Ground truth 文件不存在: ${gnd}"
        echo "跳过此配置..."
        echo ""
        continue
    fi
    
    if [ ! -f "${clusters_file}" ]; then
        echo "警告: Clusters 文件不存在: ${clusters_file}"
        echo "跳过此配置..."
        echo ""
        continue
    fi
    
    # 执行测试
    echo "开始测试..."
    echo "结果将保存到: ${res}"
    echo ""
    
    ./src/search_ivf -d ${randomize} \
                     -n ${data} \
                     -i ${index} \
                     -q ${query} \
                     -g ${gnd} \
                     -r ${res} \
                     -t ${trans} \
                     -k ${K} \
                     -a ${diskK} \
                     -o ${k_overlap} \
                     -f ${CC} \
                     -b ${clusters_file} \
                     -h ${top_centroids_file} \
                     -x ${ACTUAL_C} \
                     -z "${test_params}"
    
    if [ $? -eq 0 ]; then
        echo "测试完成: ${data} (C=${C})"
    else
        echo "测试失败: ${data} (C=${C})"
    fi
    echo ""
    echo "======================================"
    echo ""
done

echo "======================================"
echo "所有测试完成!"
echo "======================================"

