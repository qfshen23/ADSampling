#!/bin/bash

cd ..
# 编译 search_ivfpq，启用 SIMD 优化
g++ -O3 -mavx2 -mfma -mavx -msse4.2 ./src/search_ivfpq.cpp -o ./src/search_ivfpq \
    -I ./src/ \
    -I /usr/include/eigen3 \
    -lfaiss \
    -lopenblas \
    -lgomp \
    -fopenmp

# 配置路径
path=/data/vector_datasets
index_path=/data/tmp/ivfpq
result_path=./results

# 索引参数
NLIST=2048
M=16
NBITS=8

# 搜索参数
K=1
NPROBE_LIST=(5 10 15 20 25 30 35 40 50 60 70 80)

datasets=('sift10m')

for data in "${datasets[@]}"
do
    echo "=========================================="
    echo "Searching IVFPQ for dataset: ${data}"
    echo "=========================================="

    index="${index_path}/${data}/${data}_ivfpq_IVF${NLIST}_PQ${M}x${NBITS}.index"
    query="${path}/${data}/${data}_query.fvecs"
    gnd="${path}/${data}/${data}_groundtruth.ivecs"
    res="${result_path}/${data}_IVFPQ_IVF${NLIST}_PQ${M}x${NBITS}-top10.log"

    echo "Index: ${index}"
    echo "Query: ${query}"
    echo "Ground truth: ${gnd}"
    echo "Result: ${res}"
    echo ""

    # 追加模式：不清空结果文件
    # > ${res}
    
    # 添加时间戳和分隔符（方便区分不同次运行）
    echo "======================================" >> ${res}
    echo "Run at: $(date '+%Y-%m-%d %H:%M:%S')" >> ${res}
    echo "Dataset: ${data}, NLIST=${NLIST}, M=${M}, NBITS=${NBITS}" >> ${res}
    echo "======================================" >> ${res}

    # 遍历不同的 nprobe 值
    for nprobe in "${NPROBE_LIST[@]}"
    do
        echo "Testing nprobe=${nprobe}..."
        ./src/search_ivfpq \
            -i ${index} \
            -q ${query} \
            -g ${gnd} \
            -k ${K} \
            -p ${nprobe} \
            -r ${res}
        
        echo ""
    done

    echo "Search completed for ${data}!"
    echo "Results saved to: ${res}"
    echo ""
done

echo "All search tasks completed!"

