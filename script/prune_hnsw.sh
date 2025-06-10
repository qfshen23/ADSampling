cd ..

g++ -o ./src/prune_hnsw ./src/prune_hnsw.cpp -I ./src/ -O3 -I /usr/include/eigen3 -fopenmp

datasets=('sift')
prune_threshold=10.0
ef=500
M=32
anchors=5

for data in "${datasets[@]}"
do
    echo "Pruning - ${data}"

    data_path=/data/vector_datasets/${data}
    index_path=/data/tmp/hnsw/${data}
    output_path=/data/tmp/hnsw_pruned/${data}

    if [ ! -d "$output_path" ]; then
        mkdir -p "$output_path"
    fi

    data_file="${data_path}/${data}_base.fvecs"
    index_file="${index_path}/${data}_ef${ef}_M${M}.index"
    output_file="${output_path}/${data}_ef${ef}_M${M}_p${prune_threshold}_aall.index"

    echo "Input index: $index_file"
    echo "Output index: $output_file"

    ./src/prune_hnsw -d $data_file -i $index_file -o $output_file -p $prune_threshold
done
