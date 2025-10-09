cd ..
g++ ./src/search_ivf.cpp -O3 -mavx -g -o ./src/search_ivf -I ./src/ -I /usr/include/eigen3 -fopenmp

path=/data/vector_datasets
index_path=/data/tmp/ivf
result_path=./results
datasets=('sift')
C=1024
CC=256
K=1
refine_num=10000
k_overlap=64
randomize=1

for data in "${datasets[@]}"
do
    res="${result_path}/${data}_IVF${C}_${randomize}.log"
    index="${index_path}/${data}/${data}_ivf_${C}_${randomize}.index"
    query="${path}/${data}/${data}_query.fvecs"
    gnd="${path}/${data}/${data}_groundtruth.ivecs"
    trans="${path}/${data}/O.fvecs"
    diskK="${result_path}/${data}_IVF${C}_${randomize}_diskK.log"

    clusters_file="${path}/${data}/O${data}_top_clusters_${C}_of_${C}.ivecs"
    # clusters_file="${path}/${data}/${data}_top_clusters_${C}.ivecs"
    top_centroids_file="${path}/${data}/O${data}_centroid_${C}.fvecs"

    echo "clusters_file: ${clusters_file}"

    ./src/search_ivf -d ${randomize} -n ${data} -i ${index} -q ${query} -g ${gnd} -r ${res} -t ${trans} -k ${K} -a ${diskK} -o ${k_overlap} -c ${refine_num} -f ${CC} -b ${clusters_file} -h ${top_centroids_file}
done
