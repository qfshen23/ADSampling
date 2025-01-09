
cd ..
g++ ./src/search_ivf.cpp -O3 -mavx -o ./src/search_ivf -I ./src/ -I /usr/include/eigen3

path=/data/vector_datasets
index_path=/data/tmp/ivf
result_path=./results
datasets=('sift' 'gist' 'deep1M')
C=4096
K=100

for data in "${datasets[@]}"
do

    for randomize in {0..2} # 0 - IVF, 1 - IVF++, 2 - IVF+
    do
        if [ $randomize == "0" ]
        then 
            echo "IVF"
            continue
        elif [ $randomize == "2" ]
        then 
            echo "IVF+"
            continue
        else
            echo "Running IVF++"
        fi

        if [ $randomize -ne 1 ];then
            echo "Skipping adaptive=${randomize} for dataset ${data}"
            continue
        fi

        res="${result_path}/${data}_IVF${C}_${randomize}.log"
        index="${index_path}/${data}/${data}_ivf_${C}_${randomize}.index"
        query="${path}/${data}/${data}_query.fvecs"
        gnd="${path}/${data}/${data}_groundtruth.ivecs"
        trans="${path}/${data}/O.fvecs"

        ./src/search_ivf -d ${randomize} -n ${data} -i ${index} -q ${query} -g ${gnd} -r ${res} -t ${trans} -k ${K}
 
    done

done
