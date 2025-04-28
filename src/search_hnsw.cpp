

#define EIGEN_DONT_PARALLELIZE
#define EIGEN_DONT_VECTORIZE
#define COUNT_DIMENSION
#define COUNT_DIST_TIME

#include <iostream>
#include <fstream>

#include <ctime>
#include <cmath>
#include <matrix.h>
#include <utils.h>
#include <hnswlib/hnswlib.h>
#include <adsampling.h>

#include <getopt.h>

using namespace std;
using namespace hnswlib;

const int MAXK = 100;

long double rotation_time=0;

static void get_gt(unsigned int *massQA, float *massQ, size_t vecsize, size_t qsize, L2Space &l2space,
       size_t vecdim, vector<std::priority_queue<std::pair<float, labeltype >>> &answers, size_t k, size_t subk, HierarchicalNSW<float> &appr_alg) {

    (vector<std::priority_queue<std::pair<float, labeltype >>>(qsize)).swap(answers);
    DISTFUNC<float> fstdistfunc_ = l2space.get_dist_func();
    for (int i = 0; i < qsize; i++) {
        for (int j = 0; j < subk; j++) {
            answers[i].emplace(appr_alg.fstdistfunc_(massQ + i * vecdim, appr_alg.getDataByInternalId(massQA[k * i + j]), appr_alg.dist_func_param_), massQA[k * i + j]);
        }
    }
}


int recall(std::priority_queue<std::pair<float, labeltype >> &result, std::priority_queue<std::pair<float, labeltype >> &gt){
    unordered_set<labeltype> g;
    int ret = 0;
    while (gt.size()) {
        g.insert(gt.top().second);
        gt.pop();
    }
    while (result.size()) {
        if (g.find(result.top().second) != g.end()) {
            ret++;
        }
        result.pop();
    }    
    return ret;
}


static void test_approx(float *massQ, size_t vecsize, size_t qsize, HierarchicalNSW<float> &appr_alg, size_t vecdim,
            vector<std::priority_queue<std::pair<float, labeltype >>> &answers, size_t k, int adaptive) {
    size_t correct = 0;
    size_t total = 0;
    long double total_time = 0;

    adsampling::clear();

    for (int i = 0; i < qsize; i++) {

#ifndef WIN32
        float sys_t, usr_t, usr_t_sum = 0;  
        struct rusage run_start, run_end;
        GetCurTime( &run_start);
#endif

        std::priority_queue<std::pair<float, labeltype >> result = appr_alg.searchKnn(massQ + vecdim * i, k, adaptive);  

#ifndef WIN32
        GetCurTime( &run_end);
        GetTime( &run_start, &run_end, &usr_t, &sys_t);
        total_time += usr_t * 1e6;
#endif
        std::priority_queue<std::pair<float, labeltype >> gt(answers[i]);
        total += gt.size();
        int tmp = recall(result, gt);
        correct += tmp;   
    }
    long double time_us_per_query = total_time / qsize + rotation_time;
    long double recall = 1.0f * correct / total;
    cout << "---------------ADSampling HNSW------------------------" << endl;
    cout << "ef = " << appr_alg.ef_ << " k = " << k << endl;
    cout << "Recall = " << recall * 100.0 << "%\t" << endl;
    cout << "QPS = " << 1e6 / (time_us_per_query) << " query/s" << endl;
    cout << "Total number of DCO = " << adsampling::tot_full_dist << endl;
    cout << "Step 1 propotion = " << adsampling::time1 / total_time * 100 << "%" << endl;
    cout << "Step 2 propotion = " << adsampling::time2 / total_time * 100 << "%" << endl;
    cout << "Step 3 propotion = " << adsampling::time3 / total_time * 100 << "%" << endl;
    cout << "Pre-processing propotion = " << adsampling::time4 / total_time * 100 << "%" << endl;
    // cout << "Time1 = " << adsampling::time1 << " us" << endl;
    // cout << "Time2 = " << adsampling::time2 << " us" << endl;
    // cout << "Distance time = " << adsampling::distance_time << " us" << endl;
    // cout << "Pruning time = " << adsampling::time3 << " us" << endl;
    // cout << "Total time = " << total_time << " us" << endl;
    // cout << "Total distance computation = " << adsampling::tot_dist_calculation << endl;
    
    // cout << "Total prune = " << adsampling::tot_prune << endl;
    // cout << "Total tt = " << adsampling::tt / qsize << endl;
    // cout << appr_alg.ef_ << " " << recall * 100.0 << " " << time_us_per_query << " " << adsampling::tot_dimension + adsampling::tot_full_dist * vecdim << endl;
    return ;
}

static void test_vs_recall(float *massQ, size_t vecsize, size_t qsize, HierarchicalNSW<float> &appr_alg, size_t vecdim,
               vector<std::priority_queue<std::pair<float, labeltype >>> &answers, size_t k, int adaptive) {
    vector<size_t> efs;
    efs.push_back(100);
    // efs.push_back(200);
    // efs.push_back(400);
    // efs.push_back(600);
    // efs.push_back(800);
    // efs.push_back(1000);
    // efs.push_back(1500);
    // efs.push_back(2000);
    for (size_t ef : efs) {
        appr_alg.setEf(ef);
        test_approx(massQ, vecsize, qsize, appr_alg, vecdim, answers, k, adaptive);
    }
}

int main(int argc, char * argv[]) {

    const struct option longopts[] ={
        // General Parameter
        {"help",                        no_argument,       0, 'h'}, 

        // Query Parameter 
        {"randomized",                  required_argument, 0, 'd'},
        {"k",                           required_argument, 0, 'k'},
        {"epsilon0",                    required_argument, 0, 'e'},
        {"gap",                         required_argument, 0, 'p'},

        // Indexing Path 
        {"dataset",                     required_argument, 0, 'n'},
        {"index_path",                  required_argument, 0, 'i'},
        {"query_path",                  required_argument, 0, 'q'},
        {"groundtruth_path",            required_argument, 0, 'g'},
        {"result_path",                 required_argument, 0, 'r'},
        {"transformation_path",         required_argument, 0, 't'},
    };

    int ind;
    int iarg = 0;
    opterr = 1;    //getopt error message (off: 0)

    char index_path[256] = "";
    char query_path[256] = "";
    char distances_path[256] = "";
    char centroids_path[256] = "";
    char groundtruth_path[256] = "";
    char result_path[256] = "";
    char dataset[256] = "";
    char transformation_path[256] = "";

    int randomize = 0;
    int subk=100;

    while(iarg != -1){
        iarg = getopt_long(argc, argv, "d:i:q:g:r:t:n:k:e:p:c:l:", longopts, &ind);
        switch (iarg){
            case 'd':
                if(optarg)randomize = atoi(optarg);
                break;
            case 'k':
                if(optarg)subk = atoi(optarg);
                break;
            case 'e':
                if(optarg) adsampling::epsilon0 = atof(optarg);
                break;
            case 'p':
                if(optarg) adsampling::delta_d = atoi(optarg);
                break;
            case 'i':
                if(optarg)strcpy(index_path, optarg);
                break;
            case 'q':
                if(optarg)strcpy(query_path, optarg);
                break;
            case 'g':
                if(optarg)strcpy(groundtruth_path, optarg);
                break;
            case 'r':
                if(optarg)strcpy(result_path, optarg);
                break;
            case 't':
                if(optarg)strcpy(transformation_path, optarg);
                break;
            case 'n':
                if(optarg)strcpy(dataset, optarg);
                break;
            case 'c':
                if(optarg)strcpy(distances_path, optarg);
                break;
            case 'l':
                if(optarg)strcpy(centroids_path, optarg);
                break;
        }
    }   

    Matrix<unsigned> G(groundtruth_path);
    Matrix<float> Q(query_path);

    freopen(result_path,"a",stdout);
    if(randomize){
        Matrix<float> P(transformation_path);
        StopW stopw = StopW();
        Q = mul(Q, P);
        rotation_time = stopw.getElapsedTimeMicro() / Q.n;
        adsampling::D = Q.d;
    }
    
    L2Space l2space(Q.d);
    HierarchicalNSW<float> *appr_alg = new HierarchicalNSW<float>(&l2space, index_path, distances_path, centroids_path, false);

    size_t k = G.d;

    vector<std::priority_queue<std::pair<float, labeltype >>> answers;

    get_gt(G.data, Q.data, appr_alg->max_elements_, Q.n, l2space, Q.d, answers, k, subk, *appr_alg);
    test_vs_recall(Q.data, appr_alg->max_elements_, Q.n, *appr_alg, Q.d, answers, subk, randomize);

    return 0;
}
