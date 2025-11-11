#include <iostream>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <getopt.h>
#include <omp.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/IndexFlat.h>
#include <faiss/index_io.h>

#include "matrix.h"
#include "utils.h"

using namespace std;

int main(int argc, char * argv[]) {

    const struct option longopts[] ={
        // General Parameter
        {"help",                        no_argument,       0, 'h'}, 

        // Index Parameter
        {"nlist",                       required_argument, 0, 'n'},
        {"M",                           required_argument, 0, 'm'},
        {"nbits",                       required_argument, 0, 'b'},

        // Indexing Path 
        {"data_path",                   required_argument, 0, 'd'},
        {"index_path",                  required_argument, 0, 'i'},
        {"train_size",                  required_argument, 0, 't'},
    };

    int ind;
    int iarg = 0;
    opterr = 1;    //getopt error message (off: 0)

    char index_path[256] = "";
    char data_path[256] = "";

    int nlist = 1024;      // IVF聚类数量
    int M = 16;            // PQ子空间数量
    int nbits = 8;         // 每个子空间的编码位数
    size_t train_size = 0; // 训练数据量，0表示使用全部数据

    while(iarg != -1){
        iarg = getopt_long(argc, argv, "n:m:b:d:i:t:h", longopts, &ind);
        switch (iarg){
            case 'n': 
                if(optarg){
                    nlist = atoi(optarg);
                }
                break;
            case 'm': 
                if(optarg){
                    M = atoi(optarg);
                }
                break;
            case 'b': 
                if(optarg){
                    nbits = atoi(optarg);
                }
                break;
            case 'd':
                if(optarg){
                    strcpy(data_path, optarg);
                }
                break;
            case 'i':
                if(optarg){
                    strcpy(index_path, optarg);
                }
                break;
            case 't':
                if(optarg){
                    train_size = atoi(optarg);
                }
                break;
        }
    }

    // 检查参数
    if(strlen(data_path) == 0 || strlen(index_path) == 0) {
        cerr << "Error: data_path and index_path are required!" << endl;
        cerr << "Usage: " << argv[0] << " -d <data_path> -i <index_path> [-n nlist] [-m M] [-b nbits] [-t train_size]" << endl;
        return 1;
    }

    cout << "========== IVFPQ Indexing Parameters ==========" << endl;
    cout << "Data path: " << data_path << endl;
    cout << "Index path: " << index_path << endl;
    cout << "nlist (IVF clusters): " << nlist << endl;
    cout << "M (PQ subspaces): " << M << endl;
    cout << "nbits: " << nbits << endl;
    cout << "train_size: " << (train_size > 0 ? to_string(train_size) : "all") << endl;
    cout << "===============================================" << endl;

    // 设置单线程
    omp_set_num_threads(1);
    
    // 读取数据
    cout << "Loading base vectors from: " << data_path << endl;
    Matrix<float> X(data_path);
    size_t nb = X.n;
    size_t d = X.d;
    cout << "Loaded base vectors: n=" << nb << ", d=" << d << endl;

    // 检查参数合法性
    if(d % M != 0) {
        cerr << "Error: dimension d=" << d << " must be divisible by M=" << M << endl;
        return 1;
    }

    // 创建 IVFPQ 索引
    cout << "Creating IVFPQ index..." << endl;
    faiss::IndexFlatL2 quantizer(d);  // 粗量化器
    faiss::IndexIVFPQ index(&quantizer, d, nlist, M, nbits);
    
    // 禁用SIMD
    index.use_precomputed_table = 0;  // 不使用预计算表
    
    // 准备训练数据
    float* train_data = X.data;
    size_t train_n = nb;
    
    if(train_size > 0 && train_size < nb) {
        train_n = train_size;
        cout << "Using " << train_n << " vectors for training (subsampled)" << endl;
        // 简单采样前train_size个向量
        // 如果需要随机采样，可以添加随机选择逻辑
    } else {
        cout << "Using all " << train_n << " vectors for training" << endl;
    }

    // 训练索引
    cout << "Training IVFPQ index..." << endl;
    StopW stopw = StopW();
    index.train(train_n, train_data);
    double train_time = stopw.getElapsedTimeMicro() / 1e6;
    cout << "Training completed in " << train_time << " seconds" << endl;

    // 添加所有向量
    cout << "Adding vectors to index..." << endl;
    stopw.reset();
    index.add(nb, X.data);
    double add_time = stopw.getElapsedTimeMicro() / 1e6;
    cout << "Adding completed in " << add_time << " seconds" << endl;
    cout << "Total vectors in index: " << index.ntotal << endl;

    // 保存索引
    cout << "Saving index to: " << index_path << endl;
    faiss::write_index(&index, index_path);
    cout << "Index saved successfully!" << endl;

    cout << "========== Indexing Summary ==========" << endl;
    cout << "Train time: " << train_time << " seconds" << endl;
    cout << "Add time: " << add_time << " seconds" << endl;
    cout << "Total time: " << (train_time + add_time) << " seconds" << endl;
    cout << "======================================" << endl;

    return 0;
}

