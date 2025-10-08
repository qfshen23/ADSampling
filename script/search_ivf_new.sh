
cd ..
g++ ./src/search_ivf.cpp -O3 -mavx -g -o ./src/search_ivf -I ./src/ -I /usr/include/eigen3 -fopenmp

path=/data/vector_datasets
index_path=/data/tmp/ivf
result_path=./results 
datasets=('sift')
C=1024
K=1
prop=25

for data in "${datasets[@]}"
do
    for randomize in {0..2} # 0 - IVF, 1 - IVF++, 2 - IVF+
    do
        if [ $randomize -ne 2 ];then
            echo "Skipping adaptive=${randomize} for dataset ${data}"
            continue
        fi

        res="${result_path}/${data}_IVF${C}_${randomize}.log"
        # index="${index_path}/${data}/${data}_ivf_${C}_${randomize}_${prop}.index"
        index="${index_path}/${data}/${data}_ivf_${C}_${randomize}.index"
        # index="${index_path}/${data}/${data}_ivf_${C}_${randomize}_reorder.index"
        query="${path}/${data}/${data}_query.fvecs"
        gnd="${path}/${data}/${data}_groundtruth.ivecs"
        trans="${path}/${data}/O.fvecs"

        diskK="${result_path}/${data}_IVF${C}_${randomize}_diskK.log"

        ./src/search_ivf -d ${randomize} -n ${data} -i ${index} -q ${query} -g ${gnd} -r ${res} -t ${trans} -k ${K} -a ${diskK}
 
    done
done
