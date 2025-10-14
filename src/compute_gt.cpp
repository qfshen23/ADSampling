#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <immintrin.h>  // AVX/SSE
#include <omp.h>
#include <queue>
#include <getopt.h>

using namespace std;
using namespace std::chrono;

// 结果对结构
struct Result {
    float dist;
    int id;
    
    bool operator<(const Result& other) const {
        return dist > other.dist;  // 最小堆
    }
};

class GroundTruthComputer {
private:
    vector<float> base_data;
    vector<float> query_data;
    int nb, nq, dim;
    
public:
    // 读取bvecs格式文件
    bool read_bvecs(const string& filename, vector<float>& data, int& n, int& d) {
        ifstream file(filename, ios::binary);
        if (!file.is_open()) {
            cerr << "Cannot open file: " << filename << endl;
            return false;
        }
        
        // 获取文件大小
        file.seekg(0, ios::end);
        size_t file_size = file.tellg();
        file.seekg(0, ios::beg);
        
        // 读取第一个向量的维度
        int first_dim;
        file.read(reinterpret_cast<char*>(&first_dim), sizeof(int));
        if (file.gcount() != sizeof(int)) {
            cerr << "Failed to read dimension from " << filename << endl;
            return false;
        }
        
        d = first_dim;
        size_t bytes_per_vector = sizeof(int) + d * sizeof(unsigned char);
        n = file_size / bytes_per_vector;
        
        cout << "Reading " << n << " vectors of dimension " << d << " from " << filename << endl;
        
        // 重置文件指针
        file.seekg(0, ios::beg);
        
        // 预分配内存
        data.resize(n * d);
        vector<unsigned char> temp_vector(d);
        
        // 读取所有向量
        for (int i = 0; i < n; i++) {
            int vec_dim;
            file.read(reinterpret_cast<char*>(&vec_dim), sizeof(int));
            if (vec_dim != d) {
                cerr << "Inconsistent dimension at vector " << i << ": expected " << d << ", got " << vec_dim << endl;
                return false;
            }
            
            // 读取byte数据并转换为float
            file.read(reinterpret_cast<char*>(temp_vector.data()), d * sizeof(unsigned char));
            for (int j = 0; j < d; j++) {
                data[i * d + j] = static_cast<float>(temp_vector[j]);
            }
        }
        
        file.close();
        return true;
    }

    // 读取fvecs格式文件
    bool read_fvecs(const string& filename, vector<float>& data, int& n, int& d) {
        ifstream file(filename, ios::binary);
        if (!file.is_open()) {
            cerr << "Cannot open file: " << filename << endl;
            return false;
        }
        
        // 获取文件大小
        file.seekg(0, ios::end);
        size_t file_size = file.tellg();
        file.seekg(0, ios::beg);
        
        // 读取第一个向量的维度
        int first_dim;
        file.read(reinterpret_cast<char*>(&first_dim), sizeof(int));
        if (file.gcount() != sizeof(int)) {
            cerr << "Failed to read dimension from " << filename << endl;
            return false;
        }
        
        d = first_dim;
        size_t bytes_per_vector = sizeof(int) + d * sizeof(float);
        n = file_size / bytes_per_vector;
        
        cout << "Reading " << n << " vectors of dimension " << d << " from " << filename << endl;
        
        // 重置文件指针
        file.seekg(0, ios::beg);
        
        // 预分配内存
        data.resize(n * d);
        
        // 读取所有向量
        for (int i = 0; i < n; i++) {
            int vec_dim;
            file.read(reinterpret_cast<char*>(&vec_dim), sizeof(int));
            if (vec_dim != d) {
                cerr << "Inconsistent dimension at vector " << i << endl;
                return false;
            }
            
            file.read(reinterpret_cast<char*>(&data[i * d]), d * sizeof(float));
        }
        
        file.close();
        return true;
    }
    
    // 写入ivecs格式文件
    bool write_ivecs(const string& filename, const vector<vector<int>>& results) {
        ofstream file(filename, ios::binary);
        if (!file.is_open()) {
            cerr << "Cannot open file for writing: " << filename << endl;
            return false;
        }
        
        for (const auto& result : results) {
            int k = result.size();
            file.write(reinterpret_cast<const char*>(&k), sizeof(int));
            file.write(reinterpret_cast<const char*>(result.data()), k * sizeof(int));
        }
        
        file.close();
        return true;
    }
    
    // SIMD加速的L2距离计算 (AVX2)
    inline float simd_l2_distance_avx2(const float* a, const float* b, int dimension) {
        __m256 sum = _mm256_setzero_ps();
        int simd_end = dimension - (dimension % 8);
        
        // 处理8个float的倍数部分
        for (int i = 0; i < simd_end; i += 8) {
            __m256 va = _mm256_loadu_ps(&a[i]);
            __m256 vb = _mm256_loadu_ps(&b[i]);
            __m256 diff = _mm256_sub_ps(va, vb);
            sum = _mm256_fmadd_ps(diff, diff, sum);
        }
        
        // 水平求和
        float result[8];
        _mm256_storeu_ps(result, sum);
        float total = result[0] + result[1] + result[2] + result[3] + 
                     result[4] + result[5] + result[6] + result[7];
        
        // 处理剩余元素
        for (int i = simd_end; i < dimension; i++) {
            float diff = a[i] - b[i];
            total += diff * diff;
        }
        
        return total;
    }
    
    // SSE版本（兼容性更好）
    inline float simd_l2_distance_sse(const float* a, const float* b, int dimension) {
        __m128 sum = _mm_setzero_ps();
        int simd_end = dimension - (dimension % 4);
        
        // 处理4个float的倍数部分
        for (int i = 0; i < simd_end; i += 4) {
            __m128 va = _mm_loadu_ps(&a[i]);
            __m128 vb = _mm_loadu_ps(&b[i]);
            __m128 diff = _mm_sub_ps(va, vb);
            sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));
        }
        
        // 水平求和
        float result[4];
        _mm_storeu_ps(result, sum);
        float total = result[0] + result[1] + result[2] + result[3];
        
        // 处理剩余元素
        for (int i = simd_end; i < dimension; i++) {
            float diff = a[i] - b[i];
            total += diff * diff;
        }
        
        return total;
    }
    
    // 普通版本
    inline float l2_distance(const float* a, const float* b, int dimension) {
        float sum = 0.0f;
        for (int i = 0; i < dimension; i++) {
            float diff = a[i] - b[i];
            sum += diff * diff;
        }
        return sum;
    }
    
    // 选择最佳的距离计算函数
    inline float compute_distance(const float* a, const float* b, int dimension) {
        #ifdef __AVX2__
            return simd_l2_distance_avx2(a, b, dimension);
        #elif defined(__SSE__)
            return simd_l2_distance_sse(a, b, dimension);
        #else
            return l2_distance(a, b, dimension);
        #endif
    }
    
    // 使用优先队列的top-k算法
    vector<int> find_topk_heap(const float* query, int k) {
        priority_queue<Result> heap;
        
        // 先填满堆
        for (int i = 0; i < min(k, nb); i++) {
            float dist = compute_distance(query, &base_data[i * dim], dim);
            heap.push({dist, i});
        }
        
        // 处理剩余的点
        for (int i = k; i < nb; i++) {
            float dist = compute_distance(query, &base_data[i * dim], dim);
            if (dist < heap.top().dist) {
                heap.pop();
                heap.push({dist, i});
            }
        }
        
        // 提取结果并排序
        vector<Result> results;
        while (!heap.empty()) {
            results.push_back(heap.top());
            heap.pop();
        }
        
        // 按距离从小到大排序
        sort(results.begin(), results.end(), [](const Result& a, const Result& b) {
            return a.dist < b.dist;
        });
        
        vector<int> indices;
        for (const auto& r : results) {
            indices.push_back(r.id);
        }
        
        return indices;
    }
    
    // 使用部分排序的top-k算法（对于小k值更高效）
    vector<int> find_topk_partial_sort(const float* query, int k) {
        vector<Result> all_results;
        all_results.reserve(nb);
        
        // 计算所有距离
        for (int i = 0; i < nb; i++) {
            float dist = compute_distance(query, &base_data[i * dim], dim);
            all_results.push_back({dist, i});
        }
        
        // 部分排序
        nth_element(all_results.begin(), all_results.begin() + k - 1, all_results.end(),
                   [](const Result& a, const Result& b) { return a.dist < b.dist; });
        
        // 对前k个元素排序
        sort(all_results.begin(), all_results.begin() + k,
             [](const Result& a, const Result& b) { return a.dist < b.dist; });
        
        vector<int> indices;
        for (int i = 0; i < k; i++) {
            indices.push_back(all_results[i].id);
        }
        
        return indices;
    }
    
    // 分块处理版本（内存友好）
    vector<int> find_topk_chunked(const float* query, int k, int chunk_size = 10000) {
        priority_queue<Result> global_heap;
        
        for (int start = 0; start < nb; start += chunk_size) {
            int end = min(start + chunk_size, nb);
            
            // 处理当前块
            for (int i = start; i < end; i++) {
                float dist = compute_distance(query, &base_data[i * dim], dim);
                
                if (global_heap.size() < k) {
                    global_heap.push({dist, i});
                } else if (dist < global_heap.top().dist) {
                    global_heap.pop();
                    global_heap.push({dist, i});
                }
            }
        }
        
        // 提取结果并排序
        vector<Result> results;
        while (!global_heap.empty()) {
            results.push_back(global_heap.top());
            global_heap.pop();
        }
        
        sort(results.begin(), results.end(), [](const Result& a, const Result& b) {
            return a.dist < b.dist;
        });
        
        vector<int> indices;
        for (const auto& r : results) {
            indices.push_back(r.id);
        }
        
        return indices;
    }
    
    // 检测文件格式
    bool is_bvecs_format(const string& filename) {
        return filename.find(".bvecs") != string::npos;
    }

public:
    bool load_data(const string& base_file, const string& query_file) {
        cout << "Loading base vectors..." << endl;
        bool success;
        if (is_bvecs_format(base_file)) {
            success = read_bvecs(base_file, base_data, nb, dim);
        } else {
            success = read_fvecs(base_file, base_data, nb, dim);
        }
        
        if (!success) {
            return false;
        }
        
        cout << "Loading query vectors..." << endl;
        int query_dim;
        if (is_bvecs_format(query_file)) {
            success = read_bvecs(query_file, query_data, nq, query_dim);
        } else {
            success = read_fvecs(query_file, query_data, nq, query_dim);
        }
        
        if (!success) {
            return false;
        }
        
        if (dim != query_dim) {
            cerr << "Dimension mismatch: base=" << dim << ", query=" << query_dim << endl;
            return false;
        }
        
        cout << "Data loaded successfully!" << endl;
        cout << "Base: " << nb << " vectors, Query: " << nq << " vectors, Dim: " << dim << endl;
        
        return true;
    }
    
    bool compute_groundtruth(const string& output_file, int k = 100, int num_threads = 0) {
        if (num_threads <= 0) {
            num_threads = omp_get_max_threads();
        }
        omp_set_num_threads(num_threads);
        
        cout << "Computing groundtruth with k=" << k << " using " << num_threads << " threads..." << endl;
        
        // 检测SIMD支持
        #ifdef __AVX2__
            cout << "Using AVX2 SIMD acceleration" << endl;
        #elif defined(__SSE__)
            cout << "Using SSE SIMD acceleration" << endl;
        #else
            cout << "Using standard computation (no SIMD)" << endl;
        #endif
        
        vector<vector<int>> results(nq);
        
        auto start_time = high_resolution_clock::now();
        
        // 选择算法
        bool use_heap = (k <= nb / 100);  // 对于小k值使用堆
        cout << "Using " << (use_heap ? "heap-based" : "partial-sort") << " algorithm" << endl;
        
        #pragma omp parallel for schedule(dynamic, 1)
        for (int i = 0; i < nq; i++) {
            const float* query = &query_data[i * dim];
            
            if (use_heap) {
                results[i] = find_topk_heap(query, k);
            } else {
                results[i] = find_topk_partial_sort(query, k);
            }
            
            // 进度报告
            if (i % 100 == 0) {
                #pragma omp critical
                {
                    cout << "Processed " << i << "/" << nq << " queries" << endl;
                }
            }
        }
        
        auto end_time = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(end_time - start_time);
        
        cout << "Computation completed in " << duration.count() << " ms" << endl;
        cout << "Average time per query: " << (double)duration.count() / nq << " ms" << endl;
        
        cout << "Writing results to " << output_file << "..." << endl;
        return write_ivecs(output_file, results);
    }
};

void print_usage(const char* program_name) {
    cout << "Usage: " << program_name << " [options]" << endl;
    cout << "Options:" << endl;
    cout << "  -b, --base <file>      Base vectors file (fvecs format)" << endl;
    cout << "  -q, --query <file>     Query vectors file (fvecs format)" << endl;
    cout << "  -o, --output <file>    Output groundtruth file (ivecs format)" << endl;
    cout << "  -k, --topk <int>       Number of nearest neighbors (default: 100)" << endl;
    cout << "  -t, --threads <int>    Number of threads (default: auto)" << endl;
    cout << "  -h, --help             Show this help message" << endl;
}

int main(int argc, char* argv[]) {
    string base_file, query_file, output_file;
    int k = 100;
    int num_threads = 0;
    
    const struct option long_options[] = {
        {"base", required_argument, 0, 'b'},
        {"query", required_argument, 0, 'q'},
        {"output", required_argument, 0, 'o'},
        {"topk", required_argument, 0, 'k'},
        {"threads", required_argument, 0, 't'},
        {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0}
    };
    
    int option_index = 0;
    int c;
    
    while ((c = getopt_long(argc, argv, "b:q:o:k:t:h", long_options, &option_index)) != -1) {
        switch (c) {
            case 'b':
                base_file = optarg;
                break;
            case 'q':
                query_file = optarg;
                break;
            case 'o':
                output_file = optarg;
                break;
            case 'k':
                k = atoi(optarg);
                break;
            case 't':
                num_threads = atoi(optarg);
                break;
            case 'h':
                print_usage(argv[0]);
                return 0;
            default:
                print_usage(argv[0]);
                return 1;
        }
    }
    
    if (base_file.empty() || query_file.empty() || output_file.empty()) {
        cerr << "Error: Base file, query file, and output file are required!" << endl;
        print_usage(argv[0]);
        return 1;
    }
    
    if (k <= 0) {
        cerr << "Error: k must be positive!" << endl;
        return 1;
    }
    
    cout << "=== High-Performance Groundtruth Computation ===" << endl;
    cout << "Base file: " << base_file << endl;
    cout << "Query file: " << query_file << endl;
    cout << "Output file: " << output_file << endl;
    cout << "k: " << k << endl;
    cout << "Threads: " << (num_threads > 0 ? to_string(num_threads) : "auto") << endl;
    cout << endl;
    
    GroundTruthComputer computer;
    
    if (!computer.load_data(base_file, query_file)) {
        cerr << "Failed to load data!" << endl;
        return 1;
    }
    
    if (!computer.compute_groundtruth(output_file, k, num_threads)) {
        cerr << "Failed to compute groundtruth!" << endl;
        return 1;
    }
    
    cout << "Groundtruth computation completed successfully!" << endl;
    return 0;
}
