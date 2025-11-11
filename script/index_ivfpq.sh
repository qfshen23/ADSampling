#!/bin/bash

cd ..
# 编译 index_ivfpq，启用 SIMD 优化
g++ -O3 -mavx2 -mfma -mavx -msse4.2 ./src/index_ivfpq.cpp -o ./src/index_ivfpq \
    -I ./src/ \
    -I /usr/include/eigen3 \
    -lfaiss \
    -lopenblas \
    -lgomp \
    -fopenmp

# 配置参数
NLIST=2048      # IVF 聚类数
M=20            # PQ 子空间数
NBITS=8         # 编码位数
TRAIN_SIZE=0    # 0表示使用全部数据训练

datasets=('spacev10m')

for data in "${datasets[@]}"
do  
    echo "=========================================="
    echo "Building IVFPQ Index for dataset: ${data}"
    echo "=========================================="

    data_path=/data/vector_datasets/${data}
    index_path=/data/tmp/ivfpq/${data}

    if [ ! -d "$index_path" ]; then 
        mkdir -p "$index_path"
    fi

    data_file="${data_path}/${data}_base.fvecs"
    index_file="${index_path}/${data}_ivfpq_IVF${NLIST}_PQ${M}x${NBITS}.index"

    echo "Data file: ${data_file}"
    echo "Index file: ${index_file}"
    echo "Parameters: nlist=${NLIST}, M=${M}, nbits=${NBITS}"
    echo ""

    # 执行索引构建  
    ./src/index_ivfpq \
        -d ${data_file} \
        -i ${index_file} \
        -n ${NLIST} \
        -m ${M} \
        -b ${NBITS} \
        -t ${TRAIN_SIZE}

    echo ""
    echo "Index built successfully for ${data}!"
    echo ""
done

