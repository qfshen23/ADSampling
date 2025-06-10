cd ..

g++ ./src/search_hnsw.cpp -O3 -o ./src/search_hnsw -I ./src/ -I /usr/include/eigen3 -fopenmp

ef=500
M=32
datasets=('sift')
adaptive=0
prune_prop=0
anchors=5   

for data in "${datasets[@]}"
do  
    echo "Searching - ${data}"

    data_path=/data/vector_datasets/${data}
    
    result_path=./results

    data_file="${data_path}/${data}_base.fvecs"

    if (( $(echo "$prune_prop > 0" | bc -l) )); then
        index_path=/data/tmp/hnsw_pruned/${data}
        index_file="${index_path}/${data}_ef${ef}_M${M}_p${prune_prop}_aall.index"
    else
        index_path=/data/tmp/hnsw/${data}
        index_file="${index_path}/${data}_ef${ef}_M${M}.index"
    fi

    # output index file
    echo "Index file: ${index_file}"

    res="${result_path}/${data}_ef${ef}_M${M}_${adaptive}.log"
    query="${data_path}/${data}_query.fvecs"
    gnd="${data_path}/${data}_groundtruth_10000.ivecs"
    trans="${data_path}/O.fvecs"

    ./src/search_hnsw -d ${adaptive} -n ${data} -i ${index_file} -q ${query} -g ${gnd} -r ${res} -t ${trans} -p ${prune_prop}
done
