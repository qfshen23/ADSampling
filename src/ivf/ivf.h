/*
We highlight the function search which are closely related to our proposed algorithms.
We have included detailed comments in these functions. 

We explain the important variables for the enhanced IVF as follows.
1. d - It represents the number of initial dimensions.
    * For IVF  , d = D. 
    * For IVF+ , d = 0. 
    * For IVF++, d = delta_d. 
2. L1_data - The array to store the first d dimensions of a vector.
3. res_data - The array to store the remaining dimensions of a vector.
*/
#include <omp.h>
#include <limits>
#include <queue>
#include <vector>
#include <algorithm>
#include <map>
#include <mutex>
#include <random>

#include "adsampling.h"
#include "matrix.h"
#include "utils.h"

typedef uint64_t uint64_t;

class IVF{
public:
    size_t N;
    size_t D;
    size_t C;
    size_t d; // the dimensionality of first a few dimensions

    float* L1_data;
    float* res_data;
    float* centroids;

    size_t* start;
    size_t* len;
    size_t* id;

    // Top-k clusters data
    uint64_t** topk_clusters_;
    uint64_t* topk_clusters_flat_;
    size_t topk_clusters_k_;

    // Top centroids data
    float* top_centroids_;
    size_t top_centroids_num_;

    IVF();
    IVF(const Matrix<float> &X, const Matrix<float> &_centroids, int adaptive=0);
    IVF(const Matrix<float> &X, const Matrix<float> &_centroids, const Matrix<int> &groundtruth, int adaptive);
    ~IVF();

    ResultHeap search(float* query, size_t k, size_t nprobe, float distK = std::numeric_limits<float>::max(), size_t k_overlap = 0, size_t refine_num = 0) const;
    void save(char* filename);
    void load(char* filename);
    void loadTopkClusters(const char* filename, size_t k_overlap);
    void flattenTopkClusters();
    void loadTopkCentroids(const char* filename);
    void setTopkCentroidsNum(size_t num);
};

void IVF::setTopkCentroidsNum(size_t num){
    this->top_centroids_num_ = num;
}

IVF::IVF(){
    N = D = C = d = 0;
    start = len = id = NULL;
    L1_data = res_data = centroids = NULL;
    topk_clusters_ = NULL;
    topk_clusters_k_ = 0;
    top_centroids_ = NULL;
    top_centroids_num_ = 0;
}

IVF::IVF(const Matrix<float> &X, const Matrix<float> &_centroids, int adaptive) {
    
    N = X.n;
    D = X.d;
    C = _centroids.n;
    
    // Initialize topk_clusters_ member variables
    topk_clusters_ = NULL;
    topk_clusters_k_ = 0;
    
    // Initialize top_centroids_ member variables
    top_centroids_ = NULL;
    top_centroids_num_ = 0;
    
    assert(D >= 32);
    start = new size_t [C];
    len   = new size_t [C];
    id    = new size_t [N];

    int num_threads = 32;
    // thread_temp[t*C + c] 表示第 t 个线程负责的第 c 个簇
    std::vector<std::vector<size_t>> thread_temp(num_threads * C);

    // 并行处理每个点找所属簇
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();  // 当前线程 ID
        #pragma omp for
        for(int i = 0; i < X.n; i++){
            int belong = 0;
            float dist_min = X.dist(i, _centroids, 0);
            // 找到最小距离的簇
            for(int j = 1; j < C; j++){
                float dist = X.dist(i, _centroids, j);
                if(dist < dist_min){
                    dist_min = dist;
                    belong = j;
                }
            }
            // 将索引 i 放入 thread_temp[tid*C + belong]
            thread_temp[tid*C + belong].push_back(i);

            // （可选）输出进度：因为这里会被多线程调用，谨慎使用
            if(i % 50000 == 0) {
                #pragma omp critical
                {
                    std::cerr << "[Thread " << tid << "] Processing - " 
                              << i << " / " << X.n << std::endl;
                }
            }
        }
    }
    std::cerr << "Cluster Generated!" << std::endl;

    // 将所有线程的临时结果合并到一个最终的 temp 结构中
    // 先用单线程合并，以免再次加锁
    std::vector<std::vector<size_t>> temp(C); 
    for(int t = 0; t < num_threads; t++){
        for(int c = 0; c < (int)C; c++){
            // 合并 thread_temp[t*C + c] 到 temp[c]
            auto &src = thread_temp[t*C + c];
            temp[c].insert(temp[c].end(), src.begin(), src.end());
        }
    }
    // 至此，temp[c] 就包含了所有属于第 c 个簇的样本索引

    // ------------------------------
    //  2) 统计每个簇的长度，并写入 id[]
    // ------------------------------
    size_t sum = 0;
    for(int i = 0; i < (int)C; i++){
        len[i]   = temp[i].size();
        start[i] = sum;
        sum     += len[i];
    }

    // 并行写 id[]（可选，如果数据量很大，可以并行化）
    #pragma omp parallel for
    for(int i = 0; i < (int)C; i++){
        size_t offset = start[i];
        for(size_t j = 0; j < len[i]; j++){
            id[offset + j] = temp[i][j];
        }
    }

    // ------------------------------
    //  3) 根据 adaptive 参数确定分块维度 d
    // ------------------------------
    if(adaptive == 2)      d = 32;  // IVF++ - optimize cache (d = 32 by default)
    else if(adaptive == 0) d = D;    // IVF   - plain scan
    else                   d = 0;    // IVF+  - plain ADSampling

    L1_data   = new float[N * d + 10];       // +10 可防越界（原代码如此）
    res_data  = new float[N * (D - d) + 10];
    centroids = new float[C * D];

    // ------------------------------
    //  4) 并行填充 L1_data 和 res_data
    // ------------------------------
    // 这里可以并行，因为每个 i 都独立地写入
    #pragma omp parallel for
    for(int i = 0; i < (int)N; i++){
        int x = id[i];
        // 拷贝前 d 维到 L1_data，剩余 D-d 维到 res_data
        for(int j = 0; j < (int)D; j++){
            if(j < d) {
                L1_data[i * d + j] = X.data[x * D + j];
            } else {
                res_data[i * (D - d) + (j - d)] = X.data[x * D + j];
            }
        }
    }

    std::memcpy(centroids, _centroids.data, C * D * sizeof(float));

    temp.clear();
}

IVF::IVF(const Matrix<float> &X, const Matrix<float> &_centroids, const Matrix<int> &groundtruth, int adaptive) {
    N = X.n;
    D = X.d;
    C = _centroids.n;

    // Initialize topk_clusters_ member variables
    topk_clusters_ = NULL;
    topk_clusters_k_ = 0;

    // Initialize top_centroids_ member variables
    top_centroids_ = NULL;
    top_centroids_num_ = 0;

    assert(D >= 32);
    start = new size_t[C];
    len = new size_t[C];
    id = new size_t[N];

    // 用于统计每个 base vector 在 groundtruth 中的出现次数
    std::vector<int> base_vector_count(N, 0);

    // 统计出现次数
    for (int i = 0; i < groundtruth.n; i++) {
        for (int j = 0; j < groundtruth.d; j++) {
            int base_vector_id = groundtruth.data[i * groundtruth.d + j];
            if (base_vector_id >= 0 && base_vector_id < N) { // 确保索引合法
                base_vector_count[base_vector_id]++;
            }
        }
    }

    int num_threads = 32;
    std::vector<std::vector<size_t>> thread_temp(num_threads * C);

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        #pragma omp for
        for (int i = 0; i < X.n; i++) {
            int belong = 0;
            float dist_min = X.dist(i, _centroids, 0);
            for (int j = 1; j < C; j++) {
                float dist = X.dist(i, _centroids, j);
                if (dist < dist_min) {
                    dist_min = dist;
                    belong = j;
                }
            }
            thread_temp[tid * C + belong].push_back(i);
        }
    }

    std::cerr << "Cluster Generated!" << std::endl;

    std::vector<std::vector<size_t>> temp(C);
    for (int t = 0; t < num_threads; t++) {
        for (int c = 0; c < (int)C; c++) {
            auto &src = thread_temp[t * C + c];
            temp[c].insert(temp[c].end(), src.begin(), src.end());
        }
    }

    size_t sum = 0;
    for (int i = 0; i < (int)C; i++) {
        len[i] = temp[i].size();
        start[i] = sum;
        sum += len[i];
    }

    // 对每个 cluster 内的样本索引根据 base_vector_count 从大到小排序
    #pragma omp parallel for
    for (int i = 0; i < (int)C; i++) {
        std::sort(temp[i].begin(), temp[i].end(), [&](size_t a, size_t b) {
            return base_vector_count[a] > base_vector_count[b];
        });
    }

    // 写入 id[]
    #pragma omp parallel for
    for (int i = 0; i < (int)C; i++) {
        size_t offset = start[i];
        for (size_t j = 0; j < len[i]; j++) {
            id[offset + j] = temp[i][j];
        }
    }

    if (adaptive == 1)
        d = 32;
    else if (adaptive == 0)
        d = D;
    else
        d = 0;

    L1_data = new float[N * d + 10];
    res_data = new float[N * (D - d) + 10];
    centroids = new float[C * D];

    #pragma omp parallel for
    for (int i = 0; i < (int)N; i++) {
        int x = id[i];
        for (int j = 0; j < (int)D; j++) {
            if (j < d) {
                L1_data[i * d + j] = X.data[x * D + j];
            } else {
                res_data[i * (D - d) + (j - d)] = X.data[x * D + j];
            }
        }
    }

    std::memcpy(centroids, _centroids.data, C * D * sizeof(float));
    temp.clear();
}

IVF::~IVF(){
    if(id != NULL)delete [] id;
    if(len != NULL)delete [] len;
    if(start != NULL)delete [] start;
    if(L1_data != NULL)delete [] L1_data;
    if(res_data != NULL)delete [] res_data;
    if(centroids != NULL)delete [] centroids;
    
    // Clean up topk_clusters_ data
    if(topk_clusters_ != NULL) {
        for(size_t i = 0; i < N; i++) {
            if(topk_clusters_[i] != NULL) {
                delete [] topk_clusters_[i];
            }
        }
        delete [] topk_clusters_;
    }

    if(topk_clusters_flat_ != NULL) delete [] topk_clusters_flat_;
    
    // Clean up top_centroids_ data
    if(top_centroids_ != NULL) delete [] top_centroids_;
}

ResultHeap IVF::search(
    float* query, 
    size_t k, 
    size_t nprobe, 
    float distK, 
    size_t k_overlap, 
    size_t refine_num
) const {
    // StopW stopw;
    Result* centroid_dist = new Result[C];
    for(int i = 0; i < C; i++) {
        centroid_dist[i].first = sqr_dist(query, centroids + i * D, D);
        centroid_dist[i].second = i;
    }
    adsampling::dist_cnt += C;
    std::partial_sort(centroid_dist, centroid_dist + nprobe, centroid_dist + C);
    adsampling::tot_dimension += nprobe * D;
    // adsampling::time1 += stopw.getElapsedTimeMicro();

    Result* topk_centroids_dist = new Result[top_centroids_num_];
    for(int i = 0; i < top_centroids_num_; i++) {
        topk_centroids_dist[i].first = sqr_dist(query, top_centroids_ + i * D, D);
        topk_centroids_dist[i].second = i;
    }
    std::partial_sort(topk_centroids_dist, topk_centroids_dist + k_overlap, topk_centroids_dist + top_centroids_num_);

    size_t num_words = (top_centroids_num_ + 63) / 64;
    uint64_t* query_bitmasks = new uint64_t[num_words]();
    for (size_t i = 0; i < k_overlap; i++) {
        uint32_t centroid_id = topk_centroids_dist[i].second;
        size_t word = centroid_id / 64;
        size_t bit = centroid_id % 64;
        query_bitmasks[word] |= ((uint64_t)1 << bit);
    }
    
    size_t ncan = 0;
    for(int i=0;i<nprobe;i++)
        ncan += len[centroid_dist[i].second];

    // 预分配数组，避免重复分配
    Result* dist_candidates = new Result[refine_num * 10];
    size_t* overlap_ratios = new size_t[ncan];
    size_t* local_indices = new size_t[ncan];  // 存储local_idx以避免重复计算
    size_t* selected_indices = new size_t[refine_num * 10];  // 存储通过筛选的indices
    
    size_t cur = 0;
    size_t selected_count = 0;

    // StopW stopw2;
    
    // 合并overlap计算和第一次筛选，减少一次遍历
    for(int pi = 0; pi < nprobe; ++pi) {
        int cluster_id = centroid_dist[pi].second;
        for(size_t j = 0; j < len[cluster_id]; ++j) {
            size_t local_idx = start[cluster_id] + j;
            local_indices[cur] = local_idx;  // 存储local_idx
            uint64_t* vector_clusters = topk_clusters_flat_ + local_idx * num_words;

            size_t overlap_count = 0;
            
            // 优化位运算：处理完整的SIMD words
            size_t simd_words = (num_words / 4) * 4;
            
            // SIMD processing for aligned words
            for(size_t word_idx = 0; word_idx < simd_words; word_idx += 4) {
                uint64_t overlap0 = vector_clusters[word_idx] & query_bitmasks[word_idx];
                uint64_t overlap1 = vector_clusters[word_idx + 1] & query_bitmasks[word_idx + 1];
                uint64_t overlap2 = vector_clusters[word_idx + 2] & query_bitmasks[word_idx + 2];
                uint64_t overlap3 = vector_clusters[word_idx + 3] & query_bitmasks[word_idx + 3];
                
                overlap_count += __builtin_popcountll(overlap0);
                overlap_count += __builtin_popcountll(overlap1);
                overlap_count += __builtin_popcountll(overlap2);
                overlap_count += __builtin_popcountll(overlap3);
            }
            
            // 处理剩余的words
            for(size_t word_idx = simd_words; word_idx < num_words; word_idx++) {
                uint64_t overlap_bits = vector_clusters[word_idx] & query_bitmasks[word_idx];
                overlap_count += __builtin_popcountll(overlap_bits);
            }
            
            overlap_ratios[cur++] = overlap_count;
        }
    }

    // adsampling::time2 += stopw2.getElapsedTimeMicro();

    const int MAX_OVERLAP = 64;
    int bucket[MAX_OVERLAP + 1] = {0};
    // 统计每个 overlap_ratio 出现次数
    for (size_t i = 0; i < ncan; ++i)
        bucket[overlap_ratios[i]]++;

    // 累加找到第 refine_num 大的 overlap ratio
    int acc = 0, threshold = 0;
    for (int v = MAX_OVERLAP; v >= 0; --v) {
        acc += bucket[v];
        if (acc >= (int)refine_num) {
            threshold = v;
            break;
        }
    }

    // StopW stopw3;
    // 第二次遍历：只计算通过threshold的候选点距离，同时记录索引
    // 优化：提前终止条件，当找到足够多的候选点时停止
    for(size_t i = 0; i < ncan && selected_count < refine_num * 10; ++i) {
        if(overlap_ratios[i] >= threshold) {
            size_t local_idx = local_indices[i];
            selected_indices[selected_count] = i;  // 记录原始索引
            
            // 预取下一个通过threshold的向量数据
            if(selected_count + 1 < refine_num * 10) {
                for(size_t next_i = i + 1; next_i < ncan && next_i < i + 8; ++next_i) {
                    if(overlap_ratios[next_i] >= threshold) {
                        __builtin_prefetch(L1_data + local_indices[next_i] * d, 0, 1);
                        break;
                    }
                }
            }
            
            float dist = sqr_dist(query, L1_data + local_idx * d, d);
            dist_candidates[selected_count].first = dist;
            dist_candidates[selected_count].second = id[local_idx];
            selected_count++;
        }
    }

    adsampling::tot_dimension += selected_count * d;

    ResultHeap KNNs;

    if(d == D) {
        adsampling::dist_cnt += selected_count;

        // 只有当selected_count > k时才需要排序，否则直接全部加入
        if(selected_count > k) {
            std::partial_sort(
                dist_candidates,
                dist_candidates + k,
                dist_candidates + selected_count
            );
            for(size_t i = 0; i < k; i++) {
                KNNs.emplace(dist_candidates[i].first, dist_candidates[i].second);
            }
        } else {
            for(size_t i = 0; i < selected_count; i++) {
                KNNs.emplace(dist_candidates[i].first, dist_candidates[i].second);
            }
        }
    } else {
        // 优化两阶段处理：直接使用已存储的selected_indices
        for(size_t i = 0; i < selected_count; ++i) {
            size_t original_idx = selected_indices[i];
            size_t local_idx = local_indices[original_idx];
            size_t target_id = dist_candidates[i].second;
            
            float tmp_dist = adsampling::dist_comp(distK, res_data + local_idx * (D-d), query + d, dist_candidates[i].first, d);
            if(tmp_dist > 0){
                KNNs.emplace(tmp_dist, target_id);
                if(KNNs.size() > k) KNNs.pop();
            }
            if(KNNs.size() == k && KNNs.top().first < distK){
                distK = KNNs.top().first;
            }
        }
    }
    
    delete[] centroid_dist;
    delete[] overlap_ratios;
    delete[] local_indices;
    delete[] selected_indices;
    delete[] dist_candidates;
    delete[] query_bitmasks;
    delete[] topk_centroids_dist;
    // delete[] local_idx_arr;
    return KNNs;
}

void IVF::save(char * filename){
    std::ofstream output(filename, std::ios::binary);

    output.write((char *) &N, sizeof(size_t));
    output.write((char *) &D, sizeof(size_t));
    output.write((char *) &C, sizeof(size_t));
    output.write((char *) &d, sizeof(size_t));

    if(d > 0)output.write((char *) L1_data,  N * d       * sizeof(float));
    if(d < D)output.write((char *) res_data, N * (D - d) * sizeof(float));
    output.write((char *) centroids, C * D * sizeof(float));

    output.write((char *) start, C * sizeof(size_t));
    output.write((char *) len  , C * sizeof(size_t));
    output.write((char *) id   , N * sizeof(size_t));

    output.close();
}

void IVF::load(char * filename){
    std::ifstream input(filename, std::ios::binary);
    cerr << filename << endl;

    if (!input.is_open())
        throw std::runtime_error("Cannot open file");

    input.read((char *) &N, sizeof(size_t));
    input.read((char *) &D, sizeof(size_t));
    input.read((char *) &C, sizeof(size_t));
    input.read((char *) &d, sizeof(size_t));
    cerr << N << " " << D << " " << C << " " << d << endl;

    L1_data   = new float [N * d + 10];
    res_data  = new float [N * (D - d) + 10];
    centroids = new float [C * D];
    
    start = new size_t [C];
    len   = new size_t [C];
    id    = new size_t [N];

    if(d > 0)input.read((char *) L1_data,  N * d       * sizeof(float));
    if(d < D)input.read((char *) res_data, N * (D - d) * sizeof(float));
    input.read((char *) centroids, C * D * sizeof(float));

    input.read((char *) start, C * sizeof(size_t));
    input.read((char *) len  , C * sizeof(size_t));
    input.read((char *) id   , N * sizeof(size_t));
    
    input.close();
}

void IVF::loadTopkClusters(const char* filename, size_t k_overlap) {
    topk_clusters_k_ = k_overlap;

    std::ifstream input(filename, std::ios::binary);
    if (!input.is_open()) {
        throw std::runtime_error(std::string("Cannot open topk clusters file for reading: ") + filename);
    }

    int num_clusters = top_centroids_num_;
    size_t vec_index = 0;

    int cluster_flag_width = (num_clusters + 63) / 64;
    topk_clusters_ = new uint64_t*[N];
    for (size_t i = 0; i < N; i++) {
        topk_clusters_[i] = new uint64_t[cluster_flag_width];
    }

    while (input.read(reinterpret_cast<char*>(&num_clusters), sizeof(int))) {
        if (num_clusters <= 0) {
            throw std::runtime_error("Invalid number of clusters in topk cluster file");
        }

        if (vec_index >= N) {
            throw std::runtime_error("Top-k cluster file has more entries than base vectors");
        }

        std::vector<int> cluster_ids(num_clusters);
        input.read(reinterpret_cast<char*>(cluster_ids.data()), sizeof(int) * num_clusters);

        // Only take the first topk clusters
        size_t clusters_to_process = std::min(k_overlap, (size_t)num_clusters);
        
        // Map only the nearest topk cluster_ids to bitmask
        for (size_t i = 0; i < clusters_to_process; i++) {
            int cluster_id = cluster_ids[i];
            if (cluster_id < 0 || cluster_id >= num_clusters) {
                throw std::runtime_error("Cluster ID out of range");
            }
            size_t word = cluster_id / 64;
            size_t bit = cluster_id % 64;
            topk_clusters_[vec_index][word] |= ((uint64_t)1 << bit);
        }

        vec_index++;
    }

    if (vec_index != N) {
        std::cerr << "Warning: topk cluster file has " << vec_index 
                << " entries, but index has " << N << " base vectors." << std::endl;
    }

    input.close();
}

void IVF::flattenTopkClusters() {
    if (!topk_clusters_ || !id) return;
    size_t cluster_flag_width = (top_centroids_num_ + 63) / 64;
    topk_clusters_flat_ = new uint64_t[N * cluster_flag_width];
    std::fill(topk_clusters_flat_, topk_clusters_flat_ + N * cluster_flag_width, 0);
    for (size_t i = 0; i < N; ++i) {
        size_t sorted_idx = i;
        size_t vector_id = id[i];
        for (size_t w = 0; w < cluster_flag_width; ++w) {
            topk_clusters_flat_[sorted_idx * cluster_flag_width + w] = topk_clusters_[vector_id][w];
        }
    }
    // 删除原有
    for(size_t i = 0; i < N; ++i) delete[] topk_clusters_[i];
    delete[] topk_clusters_;
    topk_clusters_ = nullptr;
}

void IVF::loadTopkCentroids(const char* filename) {
    std::ifstream input(filename, std::ios::binary);
    if (!input.is_open()) {
        throw std::runtime_error(std::string("Cannot open top centroids file for reading: ") + filename);
    }

    // Read the dimension first (4 bytes for int)
    int dim;
    input.read(reinterpret_cast<char*>(&dim), sizeof(int));
    if (input.gcount() != sizeof(int)) {
        throw std::runtime_error("Failed to read dimension from centroids file");
    }
    
    // Count total number of vectors by reading file size
    input.seekg(0, std::ios::end);
    std::streamsize file_size = input.tellg();
    input.seekg(0, std::ios::beg);
    
    // Each vector: 4 bytes for dimension + dim * 4 bytes for floats
    size_t bytes_per_vector = sizeof(int) + D * sizeof(float);
    top_centroids_num_ = file_size / bytes_per_vector;
    
    if (file_size % bytes_per_vector != 0) {
        throw std::runtime_error("Invalid fvecs file format: file size not aligned with vector size");
    }

    // Allocate memory for centroids
    top_centroids_ = new float[top_centroids_num_ * D];
    
    // Read all vectors
    for (size_t i = 0; i < top_centroids_num_; i++) {
        // Read dimension (should be same as first one)
        int vec_dim;
        input.read(reinterpret_cast<char*>(&vec_dim), sizeof(int));
        if (vec_dim != dim) {
            throw std::runtime_error("Inconsistent dimension in fvecs file");
        }
        
        // Read vector data
        input.read(reinterpret_cast<char*>(top_centroids_ + i * D), 
                   D * sizeof(float));
    }
    
    input.close();
    
    std::cerr << "Loaded " << top_centroids_num_ << " centroids with dimension " 
              << D << std::endl;
}
