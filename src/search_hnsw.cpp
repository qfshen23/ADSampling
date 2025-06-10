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

    adsampling::clear();

    for (int i = 0; i < 1000; i++) {

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

        // if(i == 0) {
        //     appr_alg.collectLayer1Vertices();
            
        //     // Write layer1 vertices to a local file
        //     std::ofstream layer1_file("sift_layer1_vertices.txt");
        //     if (layer1_file.is_open()) {
        //         for (size_t j = 0; j < appr_alg.layer1_vertices_.size(); ++j) {
        //             layer1_file << appr_alg.layer1_vertices_[j] << endl;
        //         }
        //         layer1_file.close();
        //     }
        // }

        /*
        // Check if we have cluster IDs available
        if (cluster_ids != nullptr) {
            // Get the entry point's cluster ID
            tableint entry_id = appr_alg.entry_id_;

            // Count how many ground truth points share a cluster with the entry point
            int same_cluster_count = 0;
            int total_gt_points = answers[i].size();
            
            // Get the entry point's cluster ID
            unsigned entry_cluster_id = cluster_ids[entry_id];
            
            // Create a copy of answers[i] to iterate through
            std::priority_queue<std::pair<float, labeltype>> gt_copy = answers[i];

            int idx = 99;
            
            while (!gt_copy.empty()) {
                labeltype gt_label = gt_copy.top().second;
                
                // Get the ground truth point's cluster ID
                unsigned gt_cluster_id = cluster_ids[gt_label];
                
                // Check if the ground truth point is in the same cluster as the entry point
                if (gt_cluster_id == entry_cluster_id) {
                    same_cluster_count++;
                    adsampling::avg_gt_in_same_cluster_vec[idx]++;
                }
                
                gt_copy.pop();
                idx--;
            }
            
            float percentage = (total_gt_points > 0) ? (100.0f * same_cluster_count / total_gt_points) : 0.0f;
            adsampling::avg_gt_in_same_cluster += percentage;
        }

        {
            int same_cluster_count = 0;
            int total_gt_points = answers[i].size();
            // check the query in the same cluster as the gt kNNs
            auto query_cluster_ids = appr_alg.query_nearest_centroids_;
            std::priority_queue<std::pair<float, labeltype>> gt_copy = answers[i];

            int idx = 99;
                
            while (!gt_copy.empty()) {
                labeltype gt_label = gt_copy.top().second;
                
                // Get the ground truth point's cluster ID
                unsigned gt_cluster_id = cluster_ids[gt_label];
                
                // Check if the ground truth point is in the same cluster as the entry point
                for (auto query_cluster_id : query_cluster_ids) {
                    if (gt_cluster_id == query_cluster_id) {
                        same_cluster_count++;
                        adsampling::avg_query_in_same_cluster_vec[idx]++;
                        break;
                    }
                }
                
                gt_copy.pop();
                idx--;
            }
            
            float percentage = (total_gt_points > 0) ? (100.0f * same_cluster_count / total_gt_points) : 0.0f;
            adsampling::avg_query_in_same_cluster += percentage;
        }

        
        if(i % 10 == 0) {
            // Convert result and answers[i] to vectors for analyzePrunedCandidates
            std::vector<tableint> result_vec;
            std::vector<tableint> gt_vec;
            // Copy result priority queue to vector
            std::priority_queue<std::pair<float, labeltype>> result_copy = result;
            while (!result_copy.empty()) {
                result_vec.push_back(result_copy.top().second);
                result_copy.pop();
            }
            // Copy ground truth priority queue to vector
            std::priority_queue<std::pair<float, labeltype>> gt_copy = answers[i];
            while (!gt_copy.empty()) {
                gt_vec.push_back(gt_copy.top().second);
                gt_copy.pop();
            }
            appr_alg.analyzePrunedCandidates(appr_alg, 8, result_vec, gt_vec);
            if(i % 50 == 0) {
                std::cerr << "processed query " << i << std::endl;
            }
        }

        if (cluster_ids != nullptr && appr_alg.hasCentroids()) {
            tableint entry_id = appr_alg.entry_id_;
            unsigned entry_cluster_id = cluster_ids[entry_id];
            const float* entry_centroid = appr_alg.getCentroid(entry_cluster_id);
            size_t num_centroids = appr_alg.getNumCentroids();
            size_t dim = appr_alg.getCentroidDim();

            // Step 1: compute distance from entry_cluster to all other centroids
            std::vector<std::pair<float, uint32_t>> cluster_distances;
            for (size_t c = 0; c < num_centroids; ++c) {
                float dist = appr_alg.fstdistfunc_(entry_centroid, appr_alg.getCentroid(c), appr_alg.dist_func_param_);
                cluster_distances.emplace_back(dist, c);
            }

            // Step 2: sort cluster ids by distance to entry_cluster
            std::sort(cluster_distances.begin(), cluster_distances.end());

            // Step 3: build a rank map: cluster_id -> rank (0 = closest to entry cluster)
            std::unordered_map<uint32_t, size_t> cluster_rank;
            for (size_t i = 0; i < cluster_distances.size(); ++i) {
                cluster_rank[cluster_distances[i].second] = i;
            }

            // Step 4: count how many gt labels fall into each cluster_rank
            std::vector<int> rank_buckets(20, 0);  // assume 100 clusters max (adjust if needed)
            std::priority_queue<std::pair<float, labeltype>> gt_copy = answers[i];
            while (!gt_copy.empty()) {
                labeltype gt_label = gt_copy.top().second;
                gt_copy.pop();
                uint32_t gt_cluster_id = cluster_ids[gt_label];
                size_t rank = cluster_rank[gt_cluster_id];
                if (rank < rank_buckets.size()) {
                    rank_buckets[rank]++;
                }
            }

            // Step 5: accumulate into global stats (per rank bucket)
            for (size_t r = 0; r < rank_buckets.size(); ++r) {
                adsampling::gt_cluster_rank_count[r] += rank_buckets[r];
            }
        }
        

        if(i == 5770 or i == 6710 or i == 7605 or i == 9825) {
            // Calculate and display distances between query vector and nearest 20 clusters
            if (cluster_ids != nullptr && appr_alg.hasCentroids()) {
                std::cout << "Query " << i << " - Distances to nearest 20 clusters:" << std::endl;
                
                const float* entry_point_vector = (float*)appr_alg.getDataByInternalId(appr_alg.entry_id_);
                size_t num_centroids = appr_alg.getNumCentroids();
                
                // Calculate distances from query vector to all centroids
                std::vector<std::pair<float, uint32_t>> cluster_distances;
                for (size_t c = 0; c < num_centroids; ++c) {
                    const float* centroid = appr_alg.getCentroid(c);
                    float dist = appr_alg.fstdistfunc_(entry_point_vector, centroid, appr_alg.dist_func_param_);
                    cluster_distances.emplace_back(dist, c);
                }
                
                // Sort clusters by distance (ascending)
                std::sort(cluster_distances.begin(), cluster_distances.end());
                
                // Display the nearest 20 clusters (or fewer if there aren't 20)
                size_t display_count = std::min(size_t(20), cluster_distances.size());
                for (size_t j = 0; j < display_count; ++j) {
                    std::cout << "Cluster #" << cluster_distances[j].second 
                              << ": Distance = " << cluster_distances[j].first << std::endl;
                }
            }
        }  

        */
        
        /*
        if (cluster_ids != nullptr && appr_alg.hasCentroids()) {
            // query 的最近 centroid
            auto& query_nearest_centroids = appr_alg.query_nearest_centroids_;
            if (!query_nearest_centroids.empty()) {
                const float* query_centroid = appr_alg.getCentroid(query_nearest_centroids[0]);
                size_t num_centroids = appr_alg.getNumCentroids();
                size_t dim = appr_alg.getCentroidDim();

                // Step 1: compute distance from query's centroid to all centroids
                std::vector<std::pair<float, uint32_t>> cluster_distances;
                for (size_t c = 0; c < num_centroids; ++c) {
                    float dist = appr_alg.fstdistfunc_(query_centroid, appr_alg.getCentroid(c), appr_alg.dist_func_param_);
                    cluster_distances.emplace_back(dist, c);
                }

                // Step 2: sort clusters by distance
                std::sort(cluster_distances.begin(), cluster_distances.end());

                // Step 3: build a rank map
                std::unordered_map<uint32_t, size_t> cluster_rank;
                for (size_t i = 0; i < cluster_distances.size(); ++i) {
                    cluster_rank[cluster_distances[i].second] = i;
                }

                // Step 4: count how many gt labels fall into each cluster_rank
                std::vector<int> rank_buckets(20, 0);  // adjust if >100 clusters
                std::priority_queue<std::pair<float, labeltype>> gt_copy = answers[i];
                while (!gt_copy.empty()) {
                    labeltype gt_label = gt_copy.top().second;
                    gt_copy.pop();
                    uint32_t gt_cluster_id = cluster_ids[gt_label];
                    size_t rank = cluster_rank[gt_cluster_id];
                    if (rank < rank_buckets.size()) {
                        rank_buckets[rank]++;
                    }
                }

                // Step 5: accumulate
                for (size_t r = 0; r < rank_buckets.size(); ++r) {
                    adsampling::gt_query_cluster_rank_count[r] += rank_buckets[r];
                }
            }
        }
        */

        /*
        if (cluster_ids != nullptr && appr_alg.hasCentroids()) {
                const auto& pruned_vertices = appr_alg.pruned_vertices_;
                if (!pruned_vertices.empty()) {
                    // 用 entry_id 所在 cluster 排距离（你也可以改为 query_nearest_centroids_[0]）
                    tableint entry_id = appr_alg.entry_id_;
                    unsigned entry_cluster_id = cluster_ids[entry_id];
                    const float* entry_centroid = appr_alg.getCentroid(entry_cluster_id);
                    size_t num_centroids = appr_alg.getNumCentroids();
                    size_t dim = appr_alg.getCentroidDim();

                    // Step 1: compute distance from entry_cluster to all clusters
                    std::vector<std::pair<float, uint32_t>> cluster_distances;
                    for (size_t c = 0; c < num_centroids; ++c) {
                        float dist = appr_alg.fstdistfunc_(entry_centroid, appr_alg.getCentroid(c), appr_alg.dist_func_param_);
                        cluster_distances.emplace_back(dist, c);
                    }

                    std::sort(cluster_distances.begin(), cluster_distances.end());

                    std::unordered_map<uint32_t, size_t> cluster_rank;
                    for (size_t r = 0; r < cluster_distances.size(); ++r) {
                        cluster_rank[cluster_distances[r].second] = r;
                    }

                    // 将 groundtruth 的 top-k id 存入 set，便于判断
                    std::unordered_set<labeltype> gt_set;
                    std::priority_queue<std::pair<float, labeltype>> gt_copy = answers[i];
                    while (!gt_copy.empty()) {
                        gt_set.insert(gt_copy.top().second);
                        gt_copy.pop();
                    }

                    for (auto vid : pruned_vertices) {
                        // 若被剪枝点也出现在 GT 中
                        if (gt_set.find(vid) != gt_set.end()) {
                            uint32_t cluster_id = cluster_ids[vid];
                            size_t rank = cluster_rank[cluster_id];
                            if (rank < num_centroids / 10) {
                                adsampling::pruned_cluster_rank_count[rank]++;
                            }
                        }
                    }
                }
            }
        
        std::unordered_set<labeltype> gt_set;
        std::priority_queue<std::pair<float, labeltype>> gt_copy = answers[i];
        while (!gt_copy.empty()) {
            gt_set.insert(gt_copy.top().second);
            gt_copy.pop();
        }

        // Step 1: query 的最近 centroid
        uint32_t query_cluster_id = appr_alg.query_nearest_centroids_.empty() ? cluster_ids[appr_alg.entry_id_] : appr_alg.query_nearest_centroids_[0];
        const float* query_centroid = appr_alg.getCentroid(query_cluster_id);

        // Step 2: 排序所有 centroid 到 query 的距离
        std::vector<std::pair<float, uint32_t>> cluster_distances;
        for (size_t c = 0; c < appr_alg.getNumCentroids(); ++c) {
            float dist = appr_alg.fstdistfunc_(query_centroid, appr_alg.getCentroid(c), appr_alg.dist_func_param_);
            cluster_distances.emplace_back(dist, c);
        }
        std::sort(cluster_distances.begin(), cluster_distances.end());
        std::unordered_map<uint32_t, size_t> cluster_rank;
        for (size_t r = 0; r < cluster_distances.size(); ++r) {
            cluster_rank[cluster_distances[r].second] = r;
        }

        // Step 3: 遍历 pruned ∩ GT
        for (auto vid : appr_alg.pruned_vertices_) {
            if (gt_set.find(vid) != gt_set.end()) {
                const std::vector<uint32_t>& flags = appr_alg.cluster_flags_[vid];
                if (!flags.empty()) {
                    // 找出 flags 中距离 query 最近的 cluster
                    size_t min_rank = cluster_rank.size();
                    for (auto cid : flags) {
                        if (cluster_rank.find(cid) != cluster_rank.end()) {
                            min_rank = std::min(min_rank, cluster_rank[cid]);
                        }
                    }
                    if (min_rank < 20) {
                        adsampling::pruned_in_gt_flag_rank_count[min_rank]++;
                    }
                }
            }
        }

        */

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

    int subk=10000;

    while(iarg != -1){
        iarg = getopt_long(argc, argv, "d:i:q:g:r:t:n:k:e:p:f:c:l:", longopts, &ind);
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
        }
    }   

    Matrix<unsigned> G(groundtruth_path);
    Matrix<float> Q(query_path);
    Matrix<unsigned> L(cluster_ids_path);

    freopen(result_path,"a",stdout);
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
