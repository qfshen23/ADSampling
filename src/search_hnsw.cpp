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
            vector<std::priority_queue<std::pair<float, labeltype >>> &answers, size_t k, int adaptive, unsigned *cluster_ids) {
    size_t correct = 0;
    size_t total = 0;
    long double total_time = 0;

    // Create vectors to store hop counts for each rank
    static vector<int> hop_sums(10000, 0);
    static vector<int> hop_counts(10000, 0);

    adsampling::clear();

    for (int i = 0; i < 10000; i++) {
        adsampling::visited_hop.clear();
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

        int gt_rank = 9999;
        while (!gt.empty() && gt_rank >= 0) {
            labeltype gt_id = gt.top().second;
            gt.pop();

            // If this gt point was visited during search
            auto hop_it = adsampling::visited_hop.find(gt_id);
            if (hop_it != adsampling::visited_hop.end() && hop_it->second < 99) {
                hop_sums[gt_rank] += hop_it->second;
                hop_counts[gt_rank]++;
            }
            gt_rank --;
        }

        std::priority_queue<std::pair<float, labeltype >> gt2(answers[i]);
        int tmp = recall(result, gt2);
        correct += tmp;
    }

    // Print average hops for each rank
    std::ofstream hop_stats("sift_hop_stats.txt");
    if (hop_stats.is_open()) {
        for (int rank = 0; rank < 10000; rank++) {
            if (hop_counts[rank] > 0) {
                float avg_hop = (float)hop_sums[rank] / hop_counts[rank];
                hop_stats << rank << "\t" << avg_hop << endl;
            }
        }
        hop_stats.close();
    } else {
        cerr << "Failed to open hop_stats.txt for writing" << endl;
    }

    long double time_us_per_query = total_time / qsize + rotation_time;
    long double recall = 1.0f * correct / total;
    cout << "---------------ADSampling HNSW------------------------" << endl;
    cout << "ef = " << appr_alg.ef_ << " k = " << k << endl;
    cout << "Recall = " << recall * 100.0 << "%\t" << endl;
    cout << "QPS = " << 1e6 / (time_us_per_query) << " query/s" << endl;
    cout << "Total full distance = " << adsampling::tot_full_dist << endl;
    cout << "Pruned by flags = " << adsampling::pruned_by_flags << endl;
    cout << "Avg gt in same cluster = " << adsampling::avg_gt_in_same_cluster / qsize << endl;
    cout << "Avg query in same cluster = " << adsampling::avg_query_in_same_cluster / qsize << endl;

    // Write entry_ids to a local file
    // std::ofstream entry_ids_file("sift_entry_ids.txt");
    // if (entry_ids_file.is_open()) {
    //     for (size_t i = 0; i < appr_alg.entry_ids_.size(); ++i) {
    //         entry_ids_file << appr_alg.entry_ids_[i] << endl;
    //     }
    //     entry_ids_file.close();
    //     cout << "Entry IDs written to entry_ids.txt (" << appr_alg.entry_ids_.size() << " entries)" << endl;
    // } else {
    //     cerr << "Failed to open entry_ids.txt for writing" << endl;
    // }

    // for (int i = 0; i < 8; i++) {
    //     cout << "hit_by_pruned[" << i << " hop] = " << adsampling::hit_by_pruned[i] * 1.0 / qsize << endl;
    // }

    // cout << "Groundtruth k-NNs distributed across cluster ranks (rank = distance from entry cluster):" << endl;
    // for (int i = 0; i < 20; ++i) {
    //     cout << "rank " << i << ": " << adsampling::gt_cluster_rank_count[i] << " ";
    // }
    // cout << endl;

    // cout << "Groundtruth k-NNs across cluster ranks (rank = distance from query's nearest centroid):" << endl;
    // for (int i = 0; i < 20; ++i) {
    //     cout << "rank " << i << ": " << adsampling::gt_query_cluster_rank_count[i] * 1.0 / qsize << " ";
    // }
    // cout << endl;

    // cout << "Pruned vertices cluster rank distribution:" << endl;
    // for (size_t i = 0; i < appr_alg.getNumCentroids() / 10; ++i) {
    //     cout << "rank " << i << ": " << adsampling::pruned_cluster_rank_count[i] * 1.0 / qsize << " ";
    // }
    // cout << endl;

    // cout << "Pruned ∩ GT (flag closest cluster) rank distribution:" << endl;
    // for (int i = 0; i < 20; ++i) {
    //     cout << "rank " << i << ": " << adsampling::pruned_in_gt_flag_rank_count[i] * 1.0 / qsize << " ";
    // }
    // cout << endl;


    // cout << "Bucket count for each query-gt kNN = " << endl;
    // for (int i = 0; i < 100; i++) {
    //     cout << adsampling::avg_query_in_same_cluster_vec[i] << " ";
    // }
    // cout << endl;
    // cout << "Time1 = " << adsampling::time1 << " us" << endl;
    // cout << "Time2 = " << adsampling::time2 << " us" << endl;
    // cout << "Distance time = " << adsampling::distance_time << " us" << endl;
    // cout << "Avg flags = " << adsampling::avg_flat / adsampling::tot_full_dist << endl;
    // cout << "Avg hop = " << adsampling::avg_hop / qsize << endl;
    // cout << "Total time = " << total_time << " us" << endl;
    // cout << appr_alg.ef_ << " " << recall * 100.0 << " " << time_us_per_query << " " << adsampling::tot_dimension + adsampling::tot_full_dist * vecdim << endl;
    return ;
}

static void test_vs_recall(float *massQ, size_t vecsize, size_t qsize, HierarchicalNSW<float> &appr_alg, size_t vecdim,
               vector<std::priority_queue<std::pair<float, labeltype >>> &answers, size_t k, int adaptive, unsigned *cluster_ids) {
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
        test_approx(massQ, vecsize, qsize, appr_alg, vecdim, answers, k, adaptive, cluster_ids);
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
        {"cluster_ids_path",            required_argument, 0, 'l'},
        // {"flags_path",                  required_argument, 0, 'f'},
        // {"centroid_path",               required_argument, 0, 'c'},
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
    char flags_path[256] = "";
    char centroid_path[256] = "";
    char cluster_ids_path[256] = "";
    int randomize = 0;
    char topk_clusters_path[256] = "";
    int topk_clusters = 0;
    int subk=10000;

    while(iarg != -1){
        iarg = getopt_long(argc, argv, "d:i:q:g:r:t:n:k:e:p:f:c:l:b:h:", longopts, &ind);
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
            case 'f':
                if(optarg)strcpy(flags_path, optarg);
                break;
            case 'c':
                if(optarg)strcpy(centroid_path, optarg);
                break;
            case 'l':
                if(optarg)strcpy(cluster_ids_path, optarg);
                break;
            case 'b':
                if(optarg)strcpy(topk_clusters_path, optarg);
                break;
            case 'h':
                if(optarg)topk_clusters = atoi(optarg);
                break;
        }
    }   

    Matrix<unsigned> G(groundtruth_path);
    Matrix<float> Q(query_path);
    Matrix<unsigned> L(cluster_ids_path);

    if(freopen(result_path,"a",stdout) == NULL){
        cout << "Failed to open result file" << endl;
    }
    
    if(randomize){
        Matrix<float> P(transformation_path);
        StopW stopw = StopW();
        Q = mul(Q, P);
        rotation_time = stopw.getElapsedTimeMicro() / Q.n;
        adsampling::D = Q.d;
    }
    
    L2Space l2space(Q.d);
    // HierarchicalNSW<float> *appr_alg = new HierarchicalNSW<float>(&l2space, index_path, false);

    HierarchicalNSW<float> *appr_alg = new HierarchicalNSW<float>(&l2space, index_path, centroid_path, false);

    // Load flags if provided
    if (flags_path != "") {
        appr_alg->loadFlags(flags_path);
    }

    if (topk_clusters_path != "") {
        appr_alg->loadTopkClusters(topk_clusters_path, topk_clusters);
    }

    size_t k = G.d;

    vector<std::priority_queue<std::pair<float, labeltype >>> answers;

    get_gt(G.data, Q.data, appr_alg->max_elements_, Q.n, l2space, Q.d, answers, k, subk, *appr_alg);
    test_vs_recall(Q.data, appr_alg->max_elements_, Q.n, *appr_alg, Q.d, answers, subk, randomize, L.data);

    /*
    if (L.data != nullptr && appr_alg->hasCentroids()) {
        std::cout << "Analyzing inter-cluster and intra-cluster edges..." << std::endl;

        std::vector<size_t> inter_cluster_edges(appr_alg->maxlevel_ + 1, 0);
        std::vector<size_t> intra_cluster_edges(appr_alg->maxlevel_ + 1, 0);

        for (size_t i = 0; i < appr_alg->cur_element_count; ++i) {
            if (appr_alg->isMarkedDeleted(i)) continue;

            uint32_t cluster_i = L.data[i];

            for (int level = 0; level <= appr_alg->element_levels_[i]; ++level) {
                hnswlib::linklistsizeint* ll = appr_alg->get_linklist_at_level(i, level);
                int num_neighbors = appr_alg->getListCount(ll);
                hnswlib::tableint* neighbors = (hnswlib::tableint*)(ll + 1);

                for (int j = 0; j < num_neighbors; ++j) {
                    hnswlib::tableint nb = neighbors[j];
                    if (appr_alg->isMarkedDeleted(nb)) continue;

                    uint32_t cluster_j = L.data[nb];
                    if (cluster_i == cluster_j)
                        intra_cluster_edges[level]++;
                    else
                        inter_cluster_edges[level]++;
                }
            }
        }

        std::cout << "[Edge Analysis per Level - Percentage]" << std::endl;
        std::cout << "Level\tIntra%\tInter%\tTotal" << std::endl;
        for (int l = 0; l <= appr_alg->maxlevel_; ++l) {
            size_t intra = intra_cluster_edges[l];
            size_t inter = inter_cluster_edges[l];
            size_t total = intra + inter;
            double intra_percent = total > 0 ? (100.0 * intra) / total : 0.0;
            double inter_percent = total > 0 ? (100.0 * inter) / total : 0.0;
            std::cout << l << "\t" << intra_percent << "%\t" << inter_percent << "%\t" << total << std::endl;
        }
    }
    */

    return 0;
}
