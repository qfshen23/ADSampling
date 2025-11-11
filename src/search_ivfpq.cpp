#define EIGEN_DONT_PARALLELIZE
// #define EIGEN_DONT_VECTORIZE  // 注释掉以启用 SIMD

#include <iostream>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <vector>
#include <getopt.h>
#include <omp.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/IndexFlat.h>
#include <faiss/index_io.h>

#include "matrix.h"
#include "utils.h"

using namespace std;

const int MAXK = 100;

void test(const Matrix<float> &Q, const Matrix<unsigned> &G, faiss::IndexIVFPQ &index, int k, int nprobe){
    float sys_t, usr_t, usr_t_sum = 0, total_time = 0;
    struct rusage run_start, run_end;

    size_t nq = Q.n;
    size_t d = Q.d;

    // 设置 nprobe
    index.nprobe = nprobe;
    
    // 是否开启 SIMD 和预计算表，-1表示不开启，1表示使用预计算表
    index.use_precomputed_table = 1;
    
    cout << "========== Search Parameters ==========" << endl;
    cout << "Number of queries: " << nq << endl;
    cout << "k (top-k): " << k << endl;
    cout << "nprobe: " << nprobe << endl;
    cout << "=======================================" << endl;

    // 预热（可选）
    if(nq > 0) {
        int warmup_n = min(10, (int)nq);
        for(int i = 0; i < warmup_n; i++) {
            vector<faiss::idx_t> I_warmup(k);
            vector<float> D_warmup(k);
            index.search(1, Q.data + i * d, k, D_warmup.data(), I_warmup.data());
        }
    }

    // 逐查询搜索和计时（与 IVF 保持一致）
    int correct = 0;
    for(size_t i = 0; i < nq; i++){
        vector<faiss::idx_t> I(k);
        vector<float> D(k);
        
        // 单个查询计时
        GetCurTime(&run_start);
        index.search(1, Q.data + i * d, k, D.data(), I.data());
        GetCurTime(&run_end);
        GetTime(&run_start, &run_end, &usr_t, &sys_t);
        total_time += usr_t * 1e6;
        
        // 计算 Recall
        for(int j = 0; j < k; j++){
            faiss::idx_t predicted_id = I[j];
            // 检查是否在ground truth中
            for(int g = 0; g < k && g < (int)G.d; g++){
                if(predicted_id == (faiss::idx_t)G.data[i * G.d + g]){
                    correct++;
                    break;
                }
            }
        }
    }

    float time_us_per_query = total_time / nq;
    float recall = 1.0f * correct / (nq * k);
    float qps = 1e6 / time_us_per_query;

    // 输出结果
    cout << "========== Search Results ==========" << endl;
    cout << "nprobe = " << nprobe << ", k = " << k << endl;
    cout << "Recall@" << k << " = " << (recall * 100.0) << "%" << endl;
    cout << "Time per query = " << time_us_per_query << " us" << endl;
    cout << "QPS = " << qps << " query/s" << endl;
    cout << "====================================" << endl;
}

int main(int argc, char * argv[]) {

    const struct option longopts[] ={
        // General Parameter
        {"help",                        no_argument,       0, 'h'}, 

        // Query Parameter 
        {"K",                           required_argument, 0, 'k'},
        {"nprobe",                      required_argument, 0, 'p'},

        // Indexing Path 
        {"index_path",                  required_argument, 0, 'i'},
        {"query_path",                  required_argument, 0, 'q'},
        {"groundtruth_path",            required_argument, 0, 'g'},
        {"result_path",                 required_argument, 0, 'r'},
    };

    int ind;
    int iarg = 0;
    opterr = 1;    // getopt error message (off: 0)

    char index_path[256] = "";
    char query_path[256] = "";
    char groundtruth_path[256] = "";
    char result_path[256] = "";
    
    int k = 10;
    int nprobe = 10;

    while(iarg != -1) {
        iarg = getopt_long(argc, argv, "k:p:i:q:g:r:h", longopts, &ind);
        switch (iarg){
            case 'k':
                if(optarg) k = atoi(optarg);
                break;  
            case 'p':
                if(optarg) nprobe = atoi(optarg);
                break;
            case 'i':
                if(optarg) strcpy(index_path, optarg);
                break;
            case 'q':
                if(optarg) strcpy(query_path, optarg);
                break;
            case 'g':
                if(optarg) strcpy(groundtruth_path, optarg);
                break;
            case 'r':
                if(optarg) strcpy(result_path, optarg);
                break;
        }
    }

    // 检查参数
    if(strlen(index_path) == 0 || strlen(query_path) == 0 || strlen(groundtruth_path) == 0) {
        cerr << "Error: index_path, query_path and groundtruth_path are required!" << endl;
        cerr << "Usage: " << argv[0] << " -i <index_path> -q <query_path> -g <groundtruth_path> [-k K] [-p nprobe] [-r result_path]" << endl;
        return 1;
    }

    // 设置单线程
    omp_set_num_threads(1);

    // 重定向输出到结果文件（如果指定）
    if(strlen(result_path) > 0) {
        if(freopen(result_path, "a", stdout) == NULL) {
            cerr << "Error: cannot open file " << result_path << endl;
            exit(1);
        }
    }

    // 加载索引
    faiss::Index* base_index = faiss::read_index(index_path);
    faiss::IndexIVFPQ* ivfpq_index = dynamic_cast<faiss::IndexIVFPQ*>(base_index);
    
    if(ivfpq_index == nullptr) {
        cerr << "Error: The loaded index is not an IVFPQ index!" << endl;
        delete base_index;
        return 1;
    }
    

    // 读取查询和ground truth
    Matrix<float> Q(query_path);
    Matrix<unsigned> G(groundtruth_path);

    // 执行测试
    test(Q, G, *ivfpq_index, k, nprobe);

    // 清理
    delete base_index;

    return 0;
}

