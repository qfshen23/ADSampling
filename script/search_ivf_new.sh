
cd ..
g++ ./src/search_ivf.cpp -O3 -mavx -g -o ./src/search_ivf -I ./src/ -I /usr/include/eigen3 -fopenmp

path=/data/vector_datasets
index_path=/data/tmp/ivf
result_path=./results
datasets=('gist')
C=1024
K=100
refine_num=70000
k_overlap=64

for data in "${datasets[@]}"
do
    for randomize in {0..2} # 0 - IVF, 1 - IVF++, 2 - IVF+
    do
        if [ $randomize -ne 0 ];then
            echo "Skipping adaptive=${randomize} for dataset ${data}"
            continue
        fi

        res="${result_path}/${data}_IVF${C}_${randomize}.log"
        index="${index_path}/${data}/${data}_ivf_${C}_${randomize}.index"
        query="${path}/${data}/${data}_query.fvecs"
        gnd="${path}/${data}/${data}_groundtruth.ivecs"
        trans="${path}/${data}/O.fvecs"
        diskK="${result_path}/${data}_IVF${C}_${randomize}_diskK.log"

        clusters_file="${path}/${data}/${data}_top_clusters_${C}.ivecs"

        echo "clusters_file: ${clusters_file}"

        ./src/search_ivf -d ${randomize} -n ${data} -i ${index} -q ${query} -g ${gnd} -r ${res} -t ${trans} -k ${K} -a ${diskK} -o ${k_overlap} -c ${refine_num} -b ${clusters_file}
 
    done
done
