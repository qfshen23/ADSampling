cd ..

g++ -o ./src/index_hnsw_arc_flags ./src/index_hnsw_arc_flags.cpp -I ./src/ -O3 -I /usr/include/eigen3 -fopenmp 

efConstruction=500
M=32
datasets=('sift')
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

        if [ ! -d "$index_path" ]; then 
            mkdir -p "$index_path"
        fi

        if [ $adaptive == "0" ] # raw vectors 
        then
            data_file="${data_path}/${data}_base.fvecs"
            centroid_file="${data_path}/${data}_centroid_${C}.fvecs"
        else                    # preprocessed vectors                  
            data_file="${data_path}/O${data}_base.fvecs"
            centroid_file="${data_path}/O${data}_centroid_${C}.fvecs"
        fi

        depth=1
        cluster_ids_file="${data_path}/${data}_cluster_id_${K}.ivecs" 
        index_file="${index_path}/O${data}_ef${efConstruction}_M${M}.index"
        flags_file="${index_path}/${data}_ef${efConstruction}_M${M}_arcflags_dep${depth}.index"
        echo $index_file
        echo $flags_file

        ./src/index_hnsw_arc_flags -d $data_file -i $index_file -e $efConstruction -m $M -t $depth -c $cluster_ids_file -f $flags_file
    done
done
