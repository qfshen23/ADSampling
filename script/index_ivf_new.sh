cd ..
g++ -fopenmp -O3 ./src/index_ivf.cpp -o ./src/index_ivf  -I ./src/ -I /usr/include/eigen3 
C=4096
datasets=('gist')

prop=50

for data in "${datasets[@]}"
do  
    for adaptive in {0..2}
    do

        if [ $adaptive -ne 1 ];then
            echo "Skipping adaptive=${adaptive} for dataset ${data}"
            continue
        fi  

        echo "Indexing - ${data}"

        data_path=/data/vector_datasets/${data}
        index_path=/data/tmp/ivf/${data}

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

        # 0 - IVF, 1 - IVF++, 2 - IVF+
        index_file="${index_path}/${data}_ivf_${C}_${adaptive}_${prop}.index"

        echo $index_file

        ./src/index_ivf -d $data_file -c $centroid_file -i $index_file -a $adaptive
    done
done