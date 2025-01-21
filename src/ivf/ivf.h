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

    IVF();
    IVF(const Matrix<float> &X, const Matrix<float> &_centroids, int adaptive=0);
    ~IVF();

    ResultHeap search(float* query, size_t k, size_t nprobe, float distK = std::numeric_limits<float>::max()) const;
    void save(char* filename);
    void load(char* filename);

};

IVF::IVF(){
    N = D = C = d = 0;
    start = len = id = NULL;
    L1_data = res_data = centroids = NULL;
}

IVF::IVF(const Matrix<float> &X, const Matrix<float> &_centroids, int adaptive) {
    
    N = X.n;
    D = X.d;
    C = _centroids.n;
    
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
    if(adaptive == 1)      d = 64;  // IVF++ - optimize cache (d = 32 by default)
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

    // ------------------------------
    //  5) 拷贝聚类中心
    // ------------------------------
    std::memcpy(centroids, _centroids.data, C * D * sizeof(float));

    // 不要忘了释放 thread_temp 原来的内存（vector 的自动清理也可以）
    // 如果你要继续使用 temp，就不要释放
    temp.clear();

    // std::vector<size_t> * temp = new std::vector<size_t> [C];

    // for(int i=0;i<X.n;i++){
    //     int belong = 0;
    //     float dist_min = X.dist(i, _centroids, 0);
    //     for(int j=1;j<C;j++){
    //         float dist = X.dist(i, _centroids, j);
    //         if(dist < dist_min){
    //             dist_min = dist;
    //             belong = j;
    //         }
    //     }
    //     if(i % 50000 == 0){
    //         std::cerr << "Processing - " << i << " / " << X.n  << std::endl;
    //     }
    //     temp[belong].push_back(i);
    // }
    // std::cerr << "Cluster Generated!" << std::endl;

    // size_t sum = 0;
    // for(int i=0;i<C;i++){
    //     len[i] = temp[i].size();
    //     start[i] = sum;
    //     sum += len[i];
    //     for(int j=0;j<len[i];j++){
    //         id[start[i] + j] = temp[i][j];
    //     }
    // }

    // if(adaptive == 1) d = 240;       // IVF++ - optimize cache (d = 32 by default)
    // else if(adaptive == 0) d = D;   // IVF   - plain scan
    // else d = 0;                     // IVF+  - plain ADSampling        

    // L1_data   = new float [N * d + 10];
    // res_data  = new float [N * (D - d) + 10];
    // centroids = new float [C * D];
    
    // for(int i=0;i<N;i++){
    //     int x = id[i];
    //     for(int j=0;j<D;j++){
    //         if(j < d) L1_data[i * d + j] = X.data[x * D + j];
    //         else res_data[i * (D-d) + j - d] = X.data[x * D + j];
    //     }
    // }

    // std::memcpy(centroids, _centroids.data, C * D * sizeof(float));
    // delete [] temp;
}

IVF::~IVF(){
    if(id != NULL)delete [] id;
    if(len != NULL)delete [] len;
    if(start != NULL)delete [] start;
    if(L1_data != NULL)delete [] L1_data;
    if(res_data != NULL)delete [] res_data;
    if(centroids != NULL)delete [] centroids;
}

ResultHeap IVF::search(float* query, size_t k, size_t nprobe, float distK) const{
    // unsigned int seed = 123456;
    // std::mt19937 rng(seed);
    random_device rd;
    std::mt19937 rng(rd());
    std::uniform_real_distribution<float> random_dist(0.0, 1.0);
    // the default value of distK is +inf 
    Result* centroid_dist = new Result [C];

    StopW stopw = StopW();
    // Find out the closest N_{probe} centroids to the query vector.
    for(int i=0;i<C;i++){
#ifdef COUNT_DIST_TIME
        StopW stopw = StopW();
#endif
        centroid_dist[i].first = sqr_dist(query, centroids + i * D, D);
#ifdef COUNT_DIST_TIME
        adsampling::distance_time += stopw.getElapsedTimeMicro();
#endif               
        centroid_dist[i].second = i;
    }

    // Find out the closest N_{probe} centroids to the query vector.
    std::partial_sort(centroid_dist, centroid_dist + nprobe, centroid_dist + C);

    adsampling::time1 += stopw.getElapsedTimeMicro();
    
    size_t ncan = 0;
    for(int i=0;i<nprobe;i++)
        ncan += len[centroid_dist[i].second];
    if(d == D) adsampling::tot_dimension += 1ll * ncan * D;

    float * dist = new float [ncan];
    Result * candidates = new Result [ncan];
    int * obj= new int [ncan];
    // bool * flag = new bool [ncan];
    
    size_t cur = 0;
    // for(int i=0;i<nprobe;i++){
    //     int cluster_id = centroid_dist[i].second;
    //     for(int j=0;j<len[cluster_id];j++){
    //         float r = random_dist(rng);
    //         flag[cur] = (r <= 0.75);
    //         cur ++;
    //     }    
    // }

    // std::vector<size_t> cand;
    // cand.reserve(ncan);
    // for (size_t i = 0; i < nprobe; i++) {
    //     int cluster_id = centroid_dist[i].second;
    //     for (size_t j = 0; j < len[cluster_id]; j++) {
    //         cand.push_back(start[cluster_id] + j);
    //     }
    // }

    // std::random_device rd;
    // std::mt19937 g(rd());
    // std::shuffle(cand.begin(), cand.end(), g);

    stopw = StopW();

    // size_t tmp_size = (size_t)(cand.size() * 0.75);
    // for (size_t i = 0; i < tmp_size; i++) {
    //     size_t can = cand[i];
    //     float tmp_dist = sqr_dist(query, L1_data + can * d, d);
    //     candidates[i].first = tmp_dist;
    //     candidates[i].second = id[can];
    // }


    // Scan a few initial dimensions and store the distances.
    // For IVF (i.e., apply FDScanning), it should be D. 
    // For IVF+ (i.e., apply ADSampling without optimizing data layout), it should be 0.
    // For IVF++ (i.e., apply ADSampling with optimizing data layout), it should be delta_d (i.e., 32). 
    cur = -1;
    for(int i=0;i<nprobe;i++){
        int cluster_id = centroid_dist[i].second;
        for(int j=0;j<len[cluster_id];j++){
             
            // float r = random_dist(rng);
            // bool not_process = r > 0.5;
            // // 
            // if(not_process) {
            //     continue;
            // }

            float r = random_dist(rng);
            // flag[cur] = (r <= 0.75);
            if(r >= 0.75) continue;

            cur++;

            size_t can = start[cluster_id] + j;

            // StopW stopw3 = StopW();
            
#ifdef COUNT_DIST_TIME
            StopW stopw = StopW();
#endif
            float tmp_dist = sqr_distt(query, L1_data + can * d, d);
#ifdef COUNT_DIST_TIME
            adsampling::distance_time += stopw.getElapsedTimeMicro();
#endif      

            // adsampling::time3 += stopw3.getElapsedTimeMicro();
            
            if(d > 0) dist[cur] = tmp_dist; // IVF++ or IVF
            else dist[cur] = 0; // IVF+  - plain ADSampling
            obj[cur] = can;
            candidates[cur].first = tmp_dist; // dist[cur];
            candidates[cur].second = id[can];
        }    
    }

    adsampling::cntt += cur;

    adsampling::time2 += stopw.getElapsedTimeMicro();

    ResultHeap KNNs;

    // d == D indicates FDScanning. 
    if(d == D){  // here, it should originally be d == D
        StopW stopw = StopW();
        
        std::partial_sort(candidates, candidates + k, candidates + cur);
        
        for(int i=0;i < k;i++){
            KNNs.emplace(candidates[i].first, candidates[i].second);
        }
        adsampling::time4 += stopw.getElapsedTimeMicro();
    } else if(d < D) {  // d < D indicates ADSampling with and without cache-level optimization
        auto cur_dist = dist;
        for(int i = 0;i < nprobe;i++){
            int cluster_id = centroid_dist[i].second;
            for(int j=0;j<len[cluster_id];j++){
                size_t can = start[cluster_id] + j;
                
#ifdef COUNT_DIST_TIME
                StopW stopw = StopW();
#endif
                float tmp_dist = adsampling::dist_comp(distK, res_data + can * (D-d), query + d, *cur_dist, d);
#ifdef COUNT_DIST_TIME
                adsampling::distance_time += stopw.getElapsedTimeMicro();
#endif                     
                
                
                if(tmp_dist > 0){
                    KNNs.emplace(tmp_dist, id[can]);
                    if(KNNs.size() > k) KNNs.pop();
                }
                if(KNNs.size() == k && KNNs.top().first < distK){
                    distK = KNNs.top().first;
                }
                cur_dist++;
                
            }
        }
    }

    delete [] centroid_dist;
    delete [] dist;
    delete [] candidates;
    delete [] obj;
    // delete [] flag;
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

