#include <iostream>
#include <fstream>
#include <queue>
#include <getopt.h>
#include <unordered_set>

#include "matrix.h"
#include "utils.h"
#include "hnswlib/hnswlib.h"

using namespace std;
using namespace hnswlib;

std::vector<uint32_t> read_ivecs(const std::string &filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    std::vector<uint32_t> cluster_ids;
    int d;
    while (file.read(reinterpret_cast<char*>(&d), sizeof(int))) {
        if (d != 1) {
            throw std::runtime_error("Expected dimension 1 for cluster IDs");
        }

        uint32_t cluster_id;
        if (!file.read(reinterpret_cast<char*>(&cluster_id), sizeof(int))) {
            break;
        }
        cluster_ids.push_back(cluster_id);
    }

    file.close();
    return cluster_ids;
}

int main(int argc, char * argv[]) {

    const struct option longopts[] ={
        // General Parameter
        {"help",                        no_argument,       0, 'h'}, 

        // Index Parameter
        {"efConstruction",              required_argument, 0, 'e'}, 
        {"M",                           required_argument, 0, 'm'}, 

        // Indexing Path 
        {"data_path",                   required_argument, 0, 'd'},
        {"index_path",                  required_argument, 0, 'i'},
        {"depth",                       required_argument, 0, 't'},
        {"cluster_ids_path",            required_argument, 0, 'c'},
        {"flags_path",                  required_argument, 0, 'f'},
    };

    int ind;
    int iarg = 0;
    opterr = 1;    //getopt error message (off: 0)

    char index_path[256] = "";
    char data_path[256] = "";
    char cluster_ids_path[256] = "";
    char flags_path[256] = "";
    size_t efConstruction = 0;
    size_t M = 0;
    size_t depth = 0;
    while(iarg != -1){
        iarg = getopt_long(argc, argv, "e:d:i:m:t:c:f:", longopts, &ind);
        switch (iarg){
            case 'e': 
                if(optarg){
                    efConstruction = atoi(optarg);
                }
                break;
            case 'm': 
                if(optarg){
                    M = atoi(optarg);
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
                    depth = atoi(optarg);
                }
                break;
            case 'c':
                if(optarg){
                    strcpy(cluster_ids_path, optarg);
                }
                break;
            case 'f':
                if(optarg){
                    strcpy(flags_path, optarg);
                }
                break;
        }
    }
    
    Matrix<float> *X = new Matrix<float>(data_path);
    size_t D = X->d;
    size_t N = X->n;
    size_t report = 50000;

    std::vector<uint32_t> cluster_ids = read_ivecs(cluster_ids_path);
    
    L2Space l2space(D);
    HierarchicalNSW<float> *appr_alg = new HierarchicalNSW<float>(&l2space, index_path, false);

    std::cout << "Computing cluster flags" << std::endl;
    appr_alg->computeClusterFlags(cluster_ids, depth);

    std::cout << "Saving index" << std::endl;
    appr_alg->saveFlags(flags_path);
    return 0;
}
