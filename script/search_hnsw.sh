cd ..

g++ ./src/search_hnsw.cpp -O3 -o ./src/search_hnsw -I ./src/ -I /usr/include/eigen3 

ef=500
M=32
datasets=('nuswide')
K=64

for data in "${datasets[@]}"
do  
    for adaptive in {0..2}
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
        index_file="${index_path}/O${data}_ef${ef}_M${M}.index"
        res="${result_path}/${data}_ef${ef}_M${M}_${adaptive}.log"
        query="${data_path}/${data}_query.fvecs"
        gnd="${data_path}/${data}_groundtruth.ivecs"

        depth=2

        flags_file="${index_path}/${data}_ef${ef}_M${M}_arcflags_dep${depth}.index"
        # centroid_file="${data_path}/${data}_centroid_${K}.fvecs"

        ./src/search_hnsw -d ${adaptive} -n ${data} -i ${index_file} -q ${query} -g ${gnd} -r ${res} # -f ${flags_file} # -c ${centroid_file}
    done
done
