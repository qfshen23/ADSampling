#!/bin/bash

# 基于 search_ivf_new.sh 的批量测试增强版本
# 支持自定义 test_params

cd ..
g++ ./src/search_ivf.cpp -O3 -mavx -mavx512vpopcntdq -g -o ./src/search_ivf -I ./src/ -I /usr/include/eigen3 -fopenmp

if [ $? -ne 0 ]; then
    echo "编译失败!"
    exit 1
fi

path=/data/vector_datasets
index_path=/data/tmp/ivf
result_path=./results

mkdir -p ${result_path}

# 配置多个数据集和对应的测试参数
# 格式: "数据集:C:CC:ACTUAL_C:test_params"
declare -a configs=(
    "sift10m:2048:512:512:10:400,15:1500,20:2800,25:3500,30:5000,35:7500,40:9000"
    "sift10m:4096:1024:1024:10:200,15:800,20:1400,25:1800,30:2500,35:3800"
    "sift:1024:256:256:10:400,15:1500,20:2800,25:3500,30:5000,35:7500"
    "gist:1024:256:256:10:500,15:2000,20:3500,25:4500,30:6000"
)

K=1
k_overlap=64
randomize=0

echo "======================================"
echo "批量测试开始"
echo "======================================"

test_count=0
success_count=0

for config in "${configs[@]}"
do
    # 解析配置
    IFS=':' read -r data C CC ACTUAL_C test_params <<< "$config"
    
    ((test_count++))
    
    echo ""
    echo "======================================"
    echo "测试 #${test_count}: ${data} (C=${C})"
    echo "======================================"
    echo "  C: ${C}"
    echo "  CC: ${CC}"
    echo "  ACTUAL_C: ${ACTUAL_C}"
    echo "  test_params: ${test_params}"
    echo ""
    
    res="${result_path}/${data}_IVF${C}_${randomize}.log"
    index="${index_path}/${data}/${data}_ivf_${C}_${randomize}.index"
    query="${path}/${data}/${data}_query.fvecs"
    gnd="${path}/${data}/${data}_groundtruth.ivecs"
    trans="${path}/${data}/O.fvecs"
    diskK="${result_path}/${data}_IVF${C}_${randomize}_diskK.log"

    clusters_file="${path}/${data}/${data}_top_clusters_${ACTUAL_C}_of_${C}.ivecs"
    top_centroids_file="${path}/${data}/${data}_centroid_${C}.fvecs"

    echo "clusters_file: ${clusters_file}"
    echo "index: ${index}"
    
    # 检查文件是否存在
    if [ ! -f "${index}" ]; then
        echo "⚠ 警告: 索引文件不存在，跳过此配置"
        continue
    fi
    
    if [ ! -f "${query}" ]; then
        echo "⚠ 警告: 查询文件不存在，跳过此配置"
        continue
    fi
    
    if [ ! -f "${gnd}" ]; then
        echo "⚠ 警告: Ground truth 文件不存在，跳过此配置"
        continue
    fi
    
    if [ ! -f "${clusters_file}" ]; then
        echo "⚠ 警告: Clusters 文件不存在，跳过此配置"
        continue
    fi
    
    echo "开始测试..."
    
    # 运行测试，添加 -z 参数传递 test_params
    ./src/search_ivf \
        -d ${randomize} \
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
        echo "✓ 测试完成: ${data} (C=${C})"
        ((success_count++))
    else
        echo "✗ 测试失败: ${data} (C=${C})"
    fi
done

echo ""
echo "======================================"
echo "批量测试完成"
echo "======================================"
echo "总测试数: ${test_count}"
echo "成功: ${success_count}"
echo "失败: $((test_count - success_count))"
echo "======================================"

