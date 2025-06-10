#include <iostream>
#include <fstream>
#include <ctime>
#include <cmath>
#include <matrix.h>
#include <utils.h>
#include <hnswlib/hnswlib.h>
#include <getopt.h>

using namespace std;
using namespace hnswlib;

int main(int argc, char * argv[]) {
    const struct option longopts[] = {
        {"help", no_argument, 0, 'h'},
        {"data_path", required_argument, 0, 'd'},
        {"index_path", required_argument, 0, 'i'},
        {"output_path", required_argument, 0, 'o'},
        {"prune_threshold", required_argument, 0, 'p'},
    };

    int ind;
    int iarg = 0;
    opterr = 1;

    char data_path[256] = "";
    char index_path[256] = "";
    char output_path[256] = "";
    float prune_threshold = 0.5; // Default pruning threshold

    while(iarg != -1) {
        iarg = getopt_long(argc, argv, "d:i:o:p:", longopts, &ind);
        switch (iarg) {
            case 'd':
                if(optarg) strcpy(data_path, optarg);
                break;
            case 'i':
                if(optarg) strcpy(index_path, optarg);
                break;
            case 'o':
                if(optarg) strcpy(output_path, optarg);
                break;
            case 'p':
                if(optarg) prune_threshold = atof(optarg);
                break;
        }
    }

    if (strlen(data_path) == 0 || strlen(index_path) == 0 || strlen(output_path) == 0) {
        cout << "Usage: " << argv[0] << " -d <data_path> -i <index_path> -o <output_path> [-p <prune_threshold>]" << endl;
        return 1;
    }

    // Read base vectors to get dimension
    Matrix<float> *X = new Matrix<float>(data_path);
    size_t D = X->d;
    delete X;
    X = nullptr;

    // Load the index
    L2Space l2space(D);
    HierarchicalNSW<float> *hnsw = new HierarchicalNSW<float>(&l2space, index_path, false);

    // Prune and bucket the nodes
    auto res = hnsw->prune_and_bucket_points(prune_threshold);

    hnsw->prune_index_structure_keep_id(res.pruned_nodes);

    // Save the pruned index
    std::string output_path_str(output_path);
    hnsw->saveIndexWithBuckets(res, output_path_str);

    delete hnsw;
    return 0;
}
