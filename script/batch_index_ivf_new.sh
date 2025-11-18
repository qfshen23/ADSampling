#!/usr/bin/env bash
set -euo pipefail

# ========== 1) 编译 ==========
cd ..
g++ -fopenmp -O3 ./src/index_ivf.cpp -o ./src/index_ivf -I ./src/ -I /usr/include/eigen3

# ========== 2) 路径与基础配置 ==========
path=${PATH_VEC:-/data/vector_datasets}
index_path=${INDEX_PATH:-/data/tmp/ivf}

# 需要建立索引的多个数据集
datasets=(deep100m msmarco20m)

# 每个数据集对应的聚类数 C
declare -A C_BY_DATASET=(
  [msmarco20m]=4096
  [deep100m]=10240
  [bigann100m]=10240
)

# adaptive 模式列表: 0=IVF, 1=IVF++, 2=IVF+
# 根据 index_ivf_new.sh，通常只跑 adaptive=1
adaptive_list=(0)

# ========== 3) 为单个数据集建立索引 ==========
index_one_dataset () {
  local data="$1"
  local C="${C_BY_DATASET[$data]}"
  
  echo "========================================"
  echo "开始为数据集 ${data} 建立索引"
  echo "聚类数 C = ${C}"
  echo "========================================"
  
  local data_path="${path}/${data}"
  local dataset_index_path="${index_path}/${data}"
  
  # 确保索引目录存在
  if [ ! -d "$dataset_index_path" ]; then
    echo "创建索引目录: ${dataset_index_path}"
    mkdir -p "$dataset_index_path"
  fi
  
  for adaptive in "${adaptive_list[@]}"; do
    echo "----------------------------------------"
    echo "数据集: ${data}, Adaptive模式: ${adaptive}"
    
    # 根据 adaptive 模式选择数据文件和质心文件
    if [ $adaptive == "0" ]; then
      # raw vectors
      data_file="${data_path}/${data}_base.fvecs"
      centroid_file="${data_path}/${data}_centroid_${C}.fvecs"
    else
      # preprocessed vectors
      data_file="${data_path}/O${data}_base.fvecs"
      centroid_file="${data_path}/O${data}_centroid_${C}.fvecs"
    fi
    
    local training="${data_path}/${data}_groundtruth.ivecs"
    local index_file="${dataset_index_path}/${data}_ivf_${C}_${adaptive}.index"
    
    echo "数据文件: ${data_file}"
    echo "质心文件: ${centroid_file}"
    echo "索引文件: ${index_file}"
    
    # 检查输入文件是否存在
    if [ ! -f "$data_file" ]; then
      echo "警告: 数据文件不存在: ${data_file}"
      continue
    fi
    
    if [ ! -f "$centroid_file" ]; then
      echo "警告: 质心文件不存在: ${centroid_file}"
      continue
    fi
    
    # 执行索引构建
    echo "执行命令: ./src/index_ivf -d $data_file -c $centroid_file -i $index_file -a $adaptive"
    ./src/index_ivf -d "$data_file" -c "$centroid_file" -i "$index_file" -a "$adaptive"
    
    echo "✓ 完成: ${data} (adaptive=${adaptive})"
    echo
  done
  
  echo "✅ 数据集 ${data} 索引构建完成！"
  echo
}

# ========== 4) 主流程：遍历所有数据集 ==========
main () {
  echo "================================================"
  echo "批量 IVF 索引构建脚本"
  echo "数据集列表: ${datasets[*]}"
  echo "================================================"
  echo
  
  for ds in "${datasets[@]}"; do
    # 检查数据集是否在 C_BY_DATASET 中配置
    if [[ ! -v C_BY_DATASET[$ds] ]]; then
      echo "错误: 数据集 ${ds} 未在 C_BY_DATASET 中配置聚类数"
      continue
    fi
    
    index_one_dataset "${ds}"
  done
  
  echo "================================================"
  echo "✅ 所有数据集索引构建完成！"
  echo "================================================"
}

main "$@"

