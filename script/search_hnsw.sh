cd ..

g++ ./src/search_hnsw.cpp -O3 -o ./src/search_hnsw -I ./src/ -I /usr/include/eigen3 -g -fopenmp

ef=500
M=32
datasets=('sift')
adaptive=0
topk_clusters=64

for data in "${datasets[@]}"
do  
    for K in 1024
    do

        if [ $adaptive -ne 0 ];then
            echo "Skipping adaptive=${adaptive} for dataset ${data}"
            continue
        fi  

        echo "Indexing - ${data}"

        data_path=/data/vector_datasets/${data}
        index_path=/data/tmp/hnsw/${data}
        result_path=./results

        if [ ! -d "$index_path" ]; then 
            mkdir -p "$index_path"
        fi

        if [ $adaptive == "0" ] # raw vectors 
        then
            data_file="${data_path}/${data}_base.fvecs"
        else                    # preprocessed vectors                  
            data_file="${data_path}/O${data}_base.fvecs"
        fi

        # 0 - IVF, 1 - IVF++, 2 - IVF+
        index_file="${index_path}/${data}_ef${ef}_M${M}.index"
        res="${result_path}/${data}_ef${ef}_M${M}_${adaptive}.log"
        query="${data_path}/${data}_query.fvecs"
        gnd="${data_path}/${data}_groundtruth_10000.ivecs"

        depth=1

        flags_file="${index_path}/${data}_ef${ef}_M${M}_arcflags${K}_dep${depth}.index"
        centroid_file="${data_path}/${data}_centroid_${K}.fvecs"
        cluster_ids_file="${data_path}/${data}_cluster_id_${K}.ivecs" 
        clusters_file="${data_path}/${data}_top_clusters_${K}.ivecs"

        # sudo perf stat -e branch-misses,branch-instructions  
        # sudo perf record -F 8000 
        ./src/search_hnsw -d ${adaptive} -n ${data} -i ${index_file} -q ${query} -g ${gnd} -r ${res} -f ${flags_file} -c ${centroid_file} -l ${cluster_ids_file} -b ${clusters_file} -h ${topk_clusters}
    done
done
