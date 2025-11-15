#!/bin/bash

# 批量测试脚本 - 从配置文件读取参数
# 用法: ./batch_search_ivf_config.sh [config_file]
# 默认配置文件: ./test_params_config.txt

# 获取配置文件路径
CONFIG_FILE="${1:-./script/test_params_config.txt}"

if [ ! -f "${CONFIG_FILE}" ]; then
    echo "错误: 配置文件不存在: ${CONFIG_FILE}"
    echo "用法: $0 [config_file]"
    exit 1
fi

cd "$(dirname "$0")/.."

# 编译
echo "正在编译 search_ivf..."
g++ ./src/search_ivf.cpp -O3 -mavx -mavx512vpopcntdq -g -o ./src/search_ivf -I ./src/ -I /usr/include/eigen3 -fopenmp
if [ $? -ne 0 ]; then
    echo "编译失败!"
    exit 1
fi
echo "编译成功!"
echo ""

path=/data/vector_datasets
index_path=/data/tmp/ivf
result_path=./results

# 确保结果目录存在
mkdir -p ${result_path}

K=10
k_overlap=64
randomize=1

echo "======================================"
echo "批量测试配置"
echo "======================================"
echo "配置文件: ${CONFIG_FILE}"
echo "数据路径: ${path}"
echo "索引路径: ${index_path}"
echo "结果路径: ${result_path}"
echo "K: ${K}"
echo "k_overlap: ${k_overlap}"
echo "randomize: ${randomize}"
echo "======================================"
echo ""

# 统计信息
total_tests=0
successful_tests=0
failed_tests=0
skipped_tests=0

# 读取配置文件并执行测试
while IFS= read -r line || [ -n "$line" ]; do
    # 跳过空行和注释行
    if [[ -z "$line" ]] || [[ "$line" =~ ^[[:space:]]*# ]]; then
        continue
    fi
    
    # 解析配置
    read -r data C CC ACTUAL_C test_params <<< "$line"
    
    # 检查是否成功解析
    if [ -z "$data" ] || [ -z "$C" ] || [ -z "$test_params" ]; then
        echo "警告: 无法解析配置行: $line"
        echo ""
        continue
    fi
    
    ((total_tests++))
    
    echo "======================================"
    echo "测试 #${total_tests}"
    echo "======================================"
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
    top_centroids_file="${path}/${data}/O${data}_centroid_${C}.fvecs"
    
    # 检查必要文件是否存在
    missing_files=()
    [ ! -f "${index}" ] && missing_files+=("索引文件: ${index}")
    [ ! -f "${query}" ] && missing_files+=("查询文件: ${query}")
    [ ! -f "${gnd}" ] && missing_files+=("Ground truth: ${gnd}")
    [ ! -f "${clusters_file}" ] && missing_files+=("Clusters: ${clusters_file}")
    
    if [ ${#missing_files[@]} -gt 0 ]; then
        echo "警告: 以下文件不存在:"
        for file in "${missing_files[@]}"; do
            echo "  - ${file}"
        done
        echo "跳过此配置..."
        ((skipped_tests++))
        echo ""
        continue
    fi
    
    # 执行测试
    echo "开始测试..."
    echo "结果将保存到: ${res}"
    echo ""
    
    start_time=$(date +%s)
    
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
    
    exit_code=$?
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    
    if [ $exit_code -eq 0 ]; then
        echo "✓ 测试完成: ${data} (C=${C}) - 耗时: ${duration}秒"
        ((successful_tests++))
    else
        echo "✗ 测试失败: ${data} (C=${C}) - 错误码: ${exit_code}"
        ((failed_tests++))
    fi
    echo ""
    echo "======================================"
    echo ""
done < "${CONFIG_FILE}"

# 输出统计信息
echo ""
echo "======================================"
echo "测试总结"
echo "======================================"
echo "总测试数: ${total_tests}"
echo "成功: ${successful_tests}"
echo "失败: ${failed_tests}"
echo "跳过: ${skipped_tests}"
echo "======================================"

if [ $failed_tests -gt 0 ]; then
    exit 1
else
    exit 0
fi

