

#define EIGEN_DONT_PARALLELIZE
#define EIGEN_DONT_VECTORIZE
// #define COUNT_DIMENSION
// #define COUNT_DIST_TIME

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
    // Use vectors instead of priority queues for better performance
    vector<labeltype> gt_labels;
    vector<labeltype> result_labels;
    
    // Reserve space to avoid reallocations
    gt_labels.reserve(gt.size());
    result_labels.reserve(result.size());
    
    // Extract labels from priority queues
    while (!gt.empty()) {
        gt_labels.push_back(gt.top().second);
        gt.pop();
    }
    
    while (!result.empty()) {
        result_labels.push_back(result.top().second);
        result.pop();
    }
    
    // Sort both vectors for efficient intersection
    sort(gt_labels.begin(), gt_labels.end());
    sort(result_labels.begin(), result_labels.end());
    
    // Count intersection using two pointers approach
    int ret = 0;
    size_t i = 0, j = 0;
    
    while (i < gt_labels.size() && j < result_labels.size()) {
        if (gt_labels[i] == result_labels[j]) {
            ret++;
            i++;
            j++;
        } else if (gt_labels[i] < result_labels[j]) {
            i++;
        } else {
            j++;
        }
    }
    
    return ret;
}


static void test_approx(float *massQ, size_t vecsize, size_t qsize, HierarchicalNSW<float> &appr_alg, size_t vecdim,
            vector<std::priority_queue<std::pair<float, labeltype >>> &answers, size_t k, int adaptive) {
    size_t correct = 0;
    size_t total = 0;
    long double total_time = 0;

    adsampling::clear();

    for (int i = 0; i < min(qsize, (size_t)1000); i++) {
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
    cout << "Total distance calculation = " << adsampling::tot_dist_calculation << endl;
    // cout << "Time1 = " << adsampling::time1 << " us" << endl;
    // cout << "Time2 = " << adsampling::time2 << " us" << endl;
    cout << "Distance time = " << adsampling::distance_time << " us" << endl;
    cout << "Total time = " << total_time << " us" << endl;
    cout << "First ef candidates = " << adsampling::first_ef_candidates << endl;

    // Save visited array to disk
    std::ofstream va_file("/home/qfshen/workspace/vdb/adsampling/data/sift_visited_array_10K.bin", std::ios::binary);
    if (va_file.is_open()) {
        size_t num_queries = adsampling::visited_array.size();
        va_file.write(reinterpret_cast<char*>(&num_queries), sizeof(size_t));
        
        for (const auto& query_array : adsampling::visited_array) {
            size_t array_size = query_array.size();
            va_file.write(reinterpret_cast<char*>(&array_size), sizeof(size_t));
            va_file.write(reinterpret_cast<const char*>(query_array.data()), array_size * sizeof(tableint));
        }
        va_file.close();
    }

    // Save lower bounds to disk
    // std::ofstream lb_file("/home/qfshen/workspace/vdb/adsampling/data/sift_lower_bounds_10K.bin", std::ios::binary);
    // if (lb_file.is_open()) {
    //     size_t num_vectors = adsampling::lower_bounds.size();
    //     lb_file.write(reinterpret_cast<char*>(&num_vectors), sizeof(size_t));
        
    //     for (const auto& vec : adsampling::lower_bounds) {
    //         size_t vec_size = vec.size();
    //         lb_file.write(reinterpret_cast<char*>(&vec_size), sizeof(size_t));
    //         lb_file.write(reinterpret_cast<const char*>(vec.data()), vec_size * sizeof(float));
    //     }
    //     lb_file.close();
    // }
    // cout << appr_alg.ef_ << " " << recall * 100.0 << " " << time_us_per_query << " " << adsampling::tot_dimension + adsampling::tot_full_dist * vecdim << endl;
}

static void test_vs_recall(float *massQ, size_t vecsize, size_t qsize, HierarchicalNSW<float> &appr_alg, size_t vecdim,
               vector<std::priority_queue<std::pair<float, labeltype >>> &answers, size_t k, int adaptive) {
    vector<size_t> efs;
    efs.push_back(40);
    // efs.push_back(50);
    // efs.push_back(60);
    // efs.push_back(70);
    // efs.push_back(80);
    // efs.push_back(90);
    // efs.push_back(100);
    // efs.push_back(120);
    // efs.push_back(140);
    // efs.push_back(160);
    // efs.push_back(180);
    // efs.push_back(200);
    // efs.push_back(100);
    // efs.push_back(200);
    // efs.push_back(400);
    // efs.push_back(201000);
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

        // Indexing Path 
        {"dataset",                     required_argument, 0, 'n'},
        {"index_path",                  required_argument, 0, 'i'},
        {"query_path",                  required_argument, 0, 'q'},
        {"groundtruth_path",            required_argument, 0, 'g'},
        {"result_path",                 required_argument, 0, 'r'},
        {"transformation_path",         required_argument, 0, 't'},
        {"prune_prop",                  required_argument, 0, 'p'},
    };

    int ind;
    int iarg = 0;
    opterr = 1;    //getopt error message (off: 0)

    char index_path[256] = "";
    char query_path[256] = "";
    char groundtruth_path[256] = "";
    char result_path[256] = "";
    char dataset[256] = "";
    char transformation_path[256] = "";
    float prune_prop = 0.0;

    int randomize = 0;
    int subk = 10000;

    while(iarg != -1) {
        iarg = getopt_long(argc, argv, "d:i:q:g:r:t:n:k:e:p:", longopts, &ind);
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
            case 'p':
                if(optarg) prune_prop = atof(optarg);
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
    HierarchicalNSW<float> *appr_alg = nullptr;
    if(abs(prune_prop) > 0.00001) {
        appr_alg = new HierarchicalNSW<float>(&l2space, index_path, true);
    } else {
        appr_alg = new HierarchicalNSW<float>(&l2space, index_path, false);
    }

    // appr_alg->printLayerStats();

    // auto indegree_histogram = appr_alg->get_indegree_histogram();

    // std::ofstream hist_file("msong_indegree_histogram.txt");
    // hist_file << indegree_histogram.size() << std::endl;
    // for (const auto& pair : indegree_histogram) {
    //     hist_file << pair.first << " " << pair.second << std::endl;
    // }
    // hist_file.close();
    // return 0;


    size_t k = 10000;

    vector<std::priority_queue<std::pair<float, labeltype >>> answers;
    get_gt(G.data, Q.data, appr_alg->max_elements_, Q.n, l2space, Q.d, answers, k, subk, *appr_alg);
    test_vs_recall(Q.data, appr_alg->max_elements_, Q.n, *appr_alg, Q.d, answers, subk, randomize);
    return 0;
}
