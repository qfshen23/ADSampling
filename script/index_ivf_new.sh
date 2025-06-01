cd ..
g++ -fopenmp -O3 ./src/index_ivf.cpp -o ./src/index_ivf  -I ./src/ -I /usr/include/eigen3 

C=4096
datasets=('sift10m')
adaptive=0

for data in "${datasets[@]}"
do  
    echo "Indexing - ${data}"

    data_path=/data/vector_datasets/${data}
    index_path=/data/tmp/ivf/${data}

    if [ ! -d "$index_path" ]; then 
        mkdir -p "$index_path"
    fi

    data_file="${data_path}/${data}_base.fvecs"
    centroid_file="${data_path}/${data}_centroid_${C}.fvecs"

    training="${data_path}/${data}_groundtruth.ivecs"

    index_file="${index_path}/${data}_ivf_${C}_${adaptive}.index"

    echo $index_file

    # ./src/index_ivf -d $data_file -c $centroid_file -i $index_file -a $adaptive -t $training
    ./src/index_ivf -d $data_file -c $centroid_file -i $index_file -a $adaptive
done