#!/usr/bin/env bash
set -euo pipefail

# ========== 1) 编译 ==========
cd ..
g++ ./src/search_ivf.cpp -O3 -mavx -g -o ./src/search_ivf -I ./src/ -I /usr/include/eigen3 -fopenmp

# ========== 2) 路径与基础配置 ==========
path=${PATH_VEC:-/data/vector_datasets}
index_path=${INDEX_PATH:-/data/tmp/ivf}
result_path=${RESULT_PATH:-./results}  

# 需要跑的多个数据集
datasets=(glove2m)

# 每个数据集的聚类数 C（你的 index 文件名需要用到）
declare -A C_BY_DATASET=(
  [sift]=1024
  [gist]=1024
  [msong]=1024
  [glove2m]=1024
  [tiny5m]=2048
  [deep10m]=2048
  [bigann10m]=2048
  [sift10m]=2048
)

# cluster_ratio（同一 dataset+K 的 nprobe 列表对所有 ratio 相同）
cluster_ratios=(1.0)

# 只跑随机化=0（IVF）；1=IVF+, 2=IVF++（按你之前逻辑跳过）
randomize_list=(0)

# ========== 3) nprobe 列表：按 (dataset, K) 映射 ==========
# 返回“逗号分隔”的字符串，便于直接传给 -b
nprobe_list_for () {
  local ds="$1" k="$2"
  if [[ "${k}" -eq 10 ]]; then
    case "${ds}" in
      msong)   echo "5,10,15,20,25,30,35,40,45,50" ;;
      glove2m) echo "5,10,15,20,25,30,35,40,45,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200" ;;
      gist)   echo "10,15,20,25,30,35,45,50,60,70,90,110" ;;
      sift)   echo "5,10,15,20,25,30,35,40,45,50" ;;
      tiny5m)   echo "10,15,20,25,30,35,40,45,50,70,90,100,120,140,160" ;;
      deep10m)   echo "5,10,15,20,25,30,35,40,50,60,70,80" ;;
      bigann10m) echo "5,10,15,20,25,30,35,40,50,60,70,80,90" ;;
      sift10m)   echo "5,10,15,20,25,30,35,40,50,60,70,80" ;;
      *) echo "Unknown dataset for K=10: ${ds}" >&2; exit 1 ;;
    esac
  elif [[ "${k}" -eq 1 ]]; then
    case "${ds}" in
      glove2m) echo "5,10,15,20,25,30,35,40,45,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200" ;;
      msong)   echo "5,10,15,20,25,30,35,40,45,50" ;;
      gist)   echo "10,15,20,25,30,35,40,45,60,70,90,100,120" ;;
      sift)   echo "5,10,15,20,25,30,45,65,80" ;;
      deep10m)   echo "5,10,15,20,25,30,35,40,45,50,55,60,65" ;;
      bigann10m) echo "5,10,15,20,25,30,35,40,45,50,60,70" ;;
      sift10m)   echo "5,10,15,20,25,30,35,40,50,60,80" ;;
      *) echo "Unknown dataset for K=1: ${ds}" >&2; exit 1 ;;
    esac
  else
    echo "Unsupported K=${k}" >&2; exit 1
  fi
}

# ========== 4) 跑一组 (dataset, K) ==========
run_one_set () {
  local data="$1" K="$2"
  local C="${C_BY_DATASET[$data]}"
  local nprobe_list; nprobe_list="$(nprobe_list_for "${data}" "${K}")"

  for randomize in "${randomize_list[@]}"; do
    # 跳过非 0 的模式（保持和你原脚本一致）
    if [[ $randomize -ne 0 ]]; then
      echo "Skipping adaptive=${randomize} for dataset ${data} (K=${K})"
      continue
    fi

    local index="${index_path}/${data}/${data}_ivf_${C}_${randomize}.index"
    local query="${path}/${data}/${data}_query.fvecs"
    local gnd="${path}/${data}/${data}_groundtruth.ivecs"
    local trans="${path}/${data}/O.fvecs"

    for ratio in "${cluster_ratios[@]}"; do
      local nprobe_str; nprobe_str="$(echo "${nprobe_list}" | tr ',' '_')"
      local ratio_str;   ratio_str="$(echo "${ratio}" | tr '.' 'p')"

      local res="${result_path}/${data}_IVF${C}_${randomize}.log"
      local diskK="${result_path}/${data}_IVF${C}_${randomize}_diskK.log"

      echo "----------------------------------------------------------------"
      echo "[RUN] dataset=${data}  K=${K}  C=${C}  adaptive=${randomize}"
      echo "      cluster_ratio=${ratio}"
      echo "      nprobes=${nprobe_list}"
      echo "      res=${res}"
      echo "----------------------------------------------------------------"

      ./src/search_ivf -d "${randomize}" \
                       -n "${data}" \
                       -i "${index}" \
                       -q "${query}" \
                       -g "${gnd}" \
                       -r "${res}" \
                       -t "${trans}" \
                       -k "${K}" \
                       -a "${diskK}" \
                       -b "${nprobe_list}" \
                       -c "${ratio}"

      echo "[DONE] ${data} K=${K} ratio=${ratio}"
      echo
    done
  done
}

# ========== 5) 主流程：多数据集 × {K=1, K=10} ==========
main () {
  for ds in "${datasets[@]}"; do
    run_one_set "${ds}" 10
    run_one_set "${ds}" 1
  done
  echo "✅ All batch searches completed!"
}

main "$@"
