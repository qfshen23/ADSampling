cd ..

g++ ./src/search_hnsw.cpp -O3 -o ./src/search_hnsw -I ./src/ -I /usr/include/eigen3 

ef=500
M=32
datasets=('msong')
adaptive=0

for data in "${datasets[@]}"
do  
    echo "Indexing - ${data}"

    data_path=/data/vector_datasets/${data}
    index_path=/data/tmp/hnsw/${data}
    result_path=./results

    if [ ! -d "$index_path" ]; then 
        mkdir -p "$index_path"
    fi


    data_file="${data_path}/${data}_base.fvecs"

    # 0 - IVF, 1 - IVF++, 2 - IVF+
    index_file="${index_path}/O${data}_ef${ef}_M${M}.index"
    res="${result_path}/${data}_ef${ef}_M${M}_${adaptive}.log"
    query="${data_path}/${data}_query.fvecs"
    gnd="${data_path}/${data}_groundtruth.ivecs"
    trans="${data_path}/O.fvecs"

    ./src/search_hnsw -d ${adaptive} -n ${data} -i ${index_file} -q ${query} -g ${gnd} -r ${res} -t ${trans}
done
