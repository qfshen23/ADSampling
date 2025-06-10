
cd ..
g++ ./src/search_ivf.cpp -O3 -mavx -g -o ./src/search_ivf -I ./src/ -I /usr/include/eigen3 -fopenmp

path=/data/vector_datasets
index_path=/data/tmp/ivf
result_path=./results   
datasets=('sift')
C=1024
randomize=0
K=10000

for data in "${datasets[@]}"
do
    res="${result_path}/${data}_IVF${C}_${randomize}.log"

    if [ ! -f "$res" ]; then
        touch "$res"
    fi

    index="${index_path}/${data}/${data}_ivf_${C}_${randomize}.index"
    query="${path}/${data}/${data}_query.fvecs"
    gnd="${path}/${data}/${data}_groundtruth_10000.ivecs"
    trans="${path}/${data}/O.fvecs"

    diskK="${result_path}/${data}_IVF${C}_${randomize}_diskK.log"
    > "$diskK"

    ./src/search_ivf -d ${randomize} -n ${data} -i ${index} -q ${query} -g ${gnd} -r ${res} -t ${trans} -k ${K} -a ${diskK}
done
