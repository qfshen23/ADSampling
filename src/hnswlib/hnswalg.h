#pragma once

/*
This implementation is largely based on https://github.com/nmslib/hnswlib. 
We highlight the following functions which are closely related to our proposed algorithms.

Function 1: searchBaseLayerST
    - the original search algorithm HNSW, which applies FDScanning for DCOs wrt the N_ef th NN

Function 2: searchBaseLayerAD
    - the proposed search algorithm HNSW+, which applies ADSampling for DCOs wrt the N_ef th NN

Function 2: searchBaseLayerADstar
    - the proposed search algorithm HNSW++, which applies ADSampling for DCOs wrt the K th NN
    - It applies the approximate distance (i.e., the by-product of ADSampling) as a key to guide graph routing.

We have included detailed comments in these functions. 
*/

#include "visited_list_pool.h"
#include "hnswlib.h"
#include "adsampling.h"
#include "utils.h"
#include <atomic>
#include <random>
#include <stdlib.h>
#include <assert.h>
#include <unordered_set>
#include <list>
#include <queue>
#include <vector>
#include <algorithm>

using namespace std;

namespace hnswlib {
    typedef unsigned int tableint;
    typedef unsigned int linklistsizeint;

    template<typename dist_t>
    class HierarchicalNSW : public AlgorithmInterface<dist_t> {
    public:
        static const tableint max_update_element_locks = 65536;
        HierarchicalNSW(SpaceInterface<dist_t> *s) {
        }

        // HierarchicalNSW(SpaceInterface<dist_t> *s, const std::string &location, bool nmslib = false, size_t max_elements=0) {
        //     loadIndex(location, s, max_elements);
        // }
        
        HierarchicalNSW(SpaceInterface<dist_t> *s, const std::string &location, const std::string &centroid_path = "", bool nmslib = false, size_t max_elements=0) {
            loadIndex(location, s, max_elements);
            if (!centroid_path.empty()) {
                loadCentroids(centroid_path);
            }
        }

        HierarchicalNSW(SpaceInterface<dist_t> *s, size_t max_elements, size_t M = 16, size_t ef_construction = 200, size_t random_seed = 100) :
                link_list_locks_(max_elements), link_list_update_locks_(max_update_element_locks), element_levels_(max_elements) {
            max_elements_ = max_elements;

            num_deleted_ = 0;
            data_size_ = s->get_data_size();
            fstdistfunc_ = s->get_dist_func();
            dist_func_param_ = s->get_dist_func_param();
            M_ = M;
            maxM_ = M_;
            maxM0_ = M_ * 2;
            ef_construction_ = std::max(ef_construction,M_);
            ef_ = 10;

            level_generator_.seed(random_seed);
            update_probability_generator_.seed(random_seed + 1);

            size_links_level0_ = maxM0_ * sizeof(tableint) + sizeof(linklistsizeint);
            size_data_per_element_ = size_links_level0_ + data_size_ + sizeof(labeltype);
            offsetData_ = size_links_level0_;
            label_offset_ = size_links_level0_ + data_size_;
            offsetLevel0_ = 0;

            data_level0_memory_ = (char *) malloc(max_elements_ * size_data_per_element_);
            if (data_level0_memory_ == nullptr)
                throw std::runtime_error("Not enough memory");

            cur_element_count = 0;

            visited_list_pool_ = new VisitedListPool(1, max_elements);

            //initializations for special treatment of the first node
            enterpoint_node_ = -1;
            maxlevel_ = -1;

            linkLists_ = (char **) malloc(sizeof(void *) * max_elements_);
            if (linkLists_ == nullptr)
                throw std::runtime_error("Not enough memory: HierarchicalNSW failed to allocate linklists");
            size_links_per_element_ = maxM_ * sizeof(tableint) + sizeof(linklistsizeint);
            mult_ = 1 / log(1.0 * M_);
            revSize_ = 1.0 / mult_;
        }

        struct CompareByFirst {
            constexpr bool operator()(std::pair<dist_t, tableint> const &a,
                                      std::pair<dist_t, tableint> const &b) const noexcept {
                return a.first < b.first;
            }
        };

        ~HierarchicalNSW() {
            free(data_level0_memory_);
            for (tableint i = 0; i < cur_element_count; i++) {
                if (element_levels_[i] > 0)
                    free(linkLists_[i]);
            }
            free(linkLists_);
            delete visited_list_pool_;
            
            // Clear cluster flags (no need to free explicitly since using vectors)
            cluster_flags_.clear();
            
            // Free centroids
            if (centroids_ != nullptr) {
                for (size_t i = 0; i < num_centroids_; i++) {
                    if (centroids_[i] != nullptr) {
                        free(centroids_[i]);
                    }
                }
                free(centroids_);
                centroids_ = nullptr;
            }
        }

        size_t max_elements_;
        size_t cur_element_count;
        size_t size_data_per_element_;
        size_t size_links_per_element_;
        size_t num_deleted_;

        size_t M_;
        size_t maxM_;
        size_t maxM0_;
        size_t ef_construction_;

        double mult_, revSize_;
        int maxlevel_;


        VisitedListPool *visited_list_pool_;
        std::mutex cur_element_count_guard_;

        std::vector<std::mutex> link_list_locks_;

        // Locks to prevent race condition during update/insert of an element at same time.
        // Note: Locks for additions can also be used to prevent this race condition if the querying of KNN is not exposed along with update/inserts i.e multithread insert/update/query in parallel.
        std::vector<std::mutex> link_list_update_locks_;
        tableint enterpoint_node_;

        size_t size_links_level0_;
        size_t offsetData_, offsetLevel0_;

        char *data_level0_memory_;
        char **linkLists_;
        std::vector<int> element_levels_;

        size_t data_size_;

        size_t label_offset_;
        DISTFUNC<dist_t> fstdistfunc_;
        void *dist_func_param_;
        std::unordered_map<labeltype, tableint> label_lookup_;

        std::default_random_engine level_generator_;
        std::default_random_engine update_probability_generator_;

        inline labeltype getExternalLabel(tableint internal_id) const {
            labeltype return_label;
            memcpy(&return_label,(data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_), sizeof(labeltype));
            return return_label;
        }

        inline void setExternalLabel(tableint internal_id, labeltype label) const {
            memcpy((data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_), &label, sizeof(labeltype));
        }

        inline labeltype *getExternalLabeLp(tableint internal_id) const {
            return (labeltype *) (data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_);
        }

        inline char *getDataByInternalId(tableint internal_id) const {
            return (data_level0_memory_ + internal_id * size_data_per_element_ + offsetData_);
        }

        int getRandomLevel(double reverse_size) {
            std::uniform_real_distribution<double> distribution(0.0, 1.0);
            double r = -log(distribution(level_generator_)) * reverse_size;
            return (int) r;
        }


        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
        searchBaseLayer(tableint ep_id, const void *data_point, int layer) {
            VisitedList *vl = visited_list_pool_->getFreeVisitedList();
            vl_type *visited_array = vl->mass;
            vl_type visited_array_tag = vl->curV;
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>> candidateSet;

            //std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
            //std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidateSet;

            dist_t lowerBound;
            if (!isMarkedDeleted(ep_id)) {
                dist_t dist = fstdistfunc_(data_point, getDataByInternalId(ep_id), dist_func_param_);
                top_candidates.emplace(dist, ep_id);
                lowerBound = dist;
                candidateSet.emplace(-dist, ep_id);
            } else {
                lowerBound = std::numeric_limits<dist_t>::max();
                candidateSet.emplace(-lowerBound, ep_id);
            }
            visited_array[ep_id] = visited_array_tag;

            while (!candidateSet.empty()) {
                std::pair<dist_t, tableint> curr_el_pair = candidateSet.top();
                if ((-curr_el_pair.first) > lowerBound && top_candidates.size() == ef_construction_) {
                    break;
                }
                candidateSet.pop();

                tableint curNodeNum = curr_el_pair.second;

                std::unique_lock <std::mutex> lock(link_list_locks_[curNodeNum]);

                int *data;// = (int *)(linkList0_ + curNodeNum * size_links_per_element0_);
                if (layer == 0) {
                    data = (int*)get_linklist0(curNodeNum);
                } else {
                    data = (int*)get_linklist(curNodeNum, layer);
//                    data = (int *) (linkLists_[curNodeNum] + (layer - 1) * size_links_per_element_);
                }
                size_t size = getListCount((linklistsizeint*)data);
                tableint *datal = (tableint *) (data + 1);
#ifdef USE_SSE
                _mm_prefetch((char *) (visited_array + *(data + 1)), _MM_HINT_T0);
                _mm_prefetch((char *) (visited_array + *(data + 1) + 64), _MM_HINT_T0);
                _mm_prefetch(getDataByInternalId(*datal), _MM_HINT_T0);
                _mm_prefetch(getDataByInternalId(*(datal + 1)), _MM_HINT_T0);
#endif

                for (size_t j = 0; j < size; j++) {
                    tableint candidate_id = *(datal + j);
//                    if (candidate_id == 0) continue;
#ifdef USE_SSE
                    _mm_prefetch((char *) (visited_array + *(datal + j + 1)), _MM_HINT_T0);
                    _mm_prefetch(getDataByInternalId(*(datal + j + 1)), _MM_HINT_T0);
#endif
                    if (visited_array[candidate_id] == visited_array_tag) continue;
                    visited_array[candidate_id] = visited_array_tag;
                    char *currObj1 = (getDataByInternalId(candidate_id));

                    dist_t dist1 = fstdistfunc_(data_point, currObj1, dist_func_param_);
                    if (top_candidates.size() < ef_construction_ || lowerBound > dist1) {
                        candidateSet.emplace(-dist1, candidate_id);
#ifdef USE_SSE
                        _mm_prefetch(getDataByInternalId(candidateSet.top().second), _MM_HINT_T0);
#endif

                        if (!isMarkedDeleted(candidate_id))
                            top_candidates.emplace(dist1, candidate_id);

                        if (top_candidates.size() > ef_construction_)
                            top_candidates.pop();

                        if (!top_candidates.empty())
                            lowerBound = top_candidates.top().first;
                    }
                }
            }
            visited_list_pool_->releaseVisitedList(vl);

            return top_candidates;
        }

        mutable std::atomic<long> metric_distance_computations;
        mutable std::atomic<long> metric_hops;

        __attribute__((noinline))
        bool check_prune(std::vector<std::vector<uint32_t>>& cluster_flags, tableint candidate_id, const std::vector<uint32_t>& nearest_centroids) const {
            // Check if candidate_id's flags have any overlap with nearest centroids
            // Returns true if we can prune this candidate
            
            // Check if the candidate's clusters contain any of the query's nearest centroids
            const std::vector<uint32_t>& candidate_clusters = cluster_flags[candidate_id];
            
            // If the candidate has no clusters, it cannot be pruned
            if (candidate_clusters.empty()) {
                return false;
            }
            
            // Check for any intersection between candidate clusters and nearest centroids
            for (const auto& centroid_id : nearest_centroids) {
                if (std::find(candidate_clusters.begin(), candidate_clusters.end(), centroid_id) != candidate_clusters.end()) {
                    // Found a match, cannot be pruned
                    return false;
                }
            }
            
            // No overlap, can be pruned
            return true;
        }

        template <bool has_deletions, bool collect_metrics=false>
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>>
        searchBaseLayerST(tableint ep_id, const void *data_point, size_t ef) const {
            VisitedList *vl = visited_list_pool_->getFreeVisitedList();
            vl_type *visited_array = vl->mass;
            vl_type visited_array_tag = vl->curV;

            entry_id_ = ep_id;
            pruned_vertices_.clear();
            entry_ids_.push_back(ep_id); 

              

            // top_candidates - the result set R
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>> top_candidates;
            // candidate_set  - the search set S
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidate_set;

            dist_t lowerBound;
            // Insert the entry point to the result and search set with its exact distance as a key. 
            if (!has_deletions || !isMarkedDeleted(ep_id)) {
#ifdef COUNT_DIST_TIME
                StopW stopw = StopW();
#endif
                dist_t dist = fstdistfunc_(data_point, getDataByInternalId(ep_id), dist_func_param_);
#ifdef COUNT_DIST_TIME
                adsampling::distance_time += stopw.getElapsedTimeMicro();
#endif          
                adsampling::tot_dist_calculation++;
                adsampling::tot_full_dist++;
                lowerBound = dist;
                top_candidates.emplace(dist, ep_id);
                candidate_set.emplace(-dist, ep_id);
            } 
            else {
                lowerBound = std::numeric_limits<dist_t>::max();
                candidate_set.emplace(-lowerBound, ep_id);
            }

            visited_array[ep_id] = visited_array_tag;
            adsampling::visited_vector_ids.push_back(ep_id);
            int cnt_visit = 1;

            std::map<tableint, int> hops;
            hops[ep_id] = 0;

            
            bool flag = false;

            // Iteratively generate candidates and conduct DCOs to maintain the result set R.
            while (!candidate_set.empty()) {
                // std::vector<std::pair<dist_t, tableint>> candidates;
                // if(cnt_visit > 100) {  
                //     flag = true;
                // }
                std::pair<dist_t, tableint> current_node_pair = candidate_set.top();

                std::vector<int> out_edges;

                // When the smallest object in S has its distance larger than the largest in R, terminate the algorithm.
                if ((-current_node_pair.first) > lowerBound && (top_candidates.size() == ef || has_deletions == false)) {
                    break;
                }
                candidate_set.pop();

                int hop = hops[current_node_pair.second];

                

                /*
                // if (cnt_visit > 10000 && !topk_cluster_flags_.empty()) {
                //     tableint candidate_id = current_node_pair.second;
                //     int collision_count = 0;
                //     for (size_t w = 0; w < cluster_flag_width_; w++) {
                //         uint32_t overlap = query_bitmasks[w] & topk_cluster_flags_[candidate_id][w];
                //         collision_count += __builtin_popcount(overlap);
                //     }
                    
                //     if ((float)collision_count / topk_clusters_ < 0.5) {
                //         adsampling::pruned_by_flags++;
                //         pruned_vertices_.push_back(candidate_id);
                //         continue;
                //     }
                // }

                // if(hop > 1) {
                //     continue;
                // }
                */
                 
                tableint current_node_id = current_node_pair.second;
                int *data = (int *) get_linklist0(current_node_id);
                size_t size = getListCount((linklistsizeint*)data);

                // Enumerate all the neighbors of the object and view them as candidates of KNNs. 
                for (size_t j = 1; j <= size; j++) {
                    int candidate_id = *(data + j);
                    out_edges.push_back(candidate_id);
                    if (!(visited_array[candidate_id] == visited_array_tag)) {
                        // out_edges.push_back(candidate_id);
                        // adsampling::visited_vector_ids.push_back(candidate_id);
                        visited_array[candidate_id] = visited_array_tag;

                        hops[candidate_id] = hop + 1;
                        adsampling::visited_hop[candidate_id] = hop + 1;
                        // Apply pruning if we have nearest centroids and cluster flags
                        // bool prune_candidate = false;
                        // if (!nearest_centroids.empty() && !cluster_flags_.empty()) {
                        //     prune_candidate = check_prune(cluster_flags_, candidate_id, nearest_centroids);
                        //     if (prune_candidate) {
                        //         adsampling::pruned_by_flags++;
                        //         pruned_vertices_.push_back(candidate_id);
                        //         continue;
                        //     }
                        // }

                        // Check bit collision count between query and candidate's top-k cluster flags
                        // if (cnt_visit > 20000 && !topk_cluster_flags_.empty()) {
                        //     int collision_count = 0;
                        //     for (size_t w = 0; w < cluster_flag_width_; w++) {
                        //         uint32_t overlap = query_bitmasks[w] & topk_cluster_flags_[candidate_id][w];
                        //         collision_count += __builtin_popcount(overlap);
                        //     }
                            
                        //     if ((float)collision_count / topk_clusters_ < 0.4) {
                        //         adsampling::pruned_by_flags++;
                        //         pruned_vertices_.push_back(candidate_id);
                        //         continue;
                        //     }
                        // }

                        // Conduct DCO with FDScanning wrt the N_ef th NN: 
                        // (1) calculate its exact distance 
                        // (2) compare it with the N_ef th distance (i.e., lowerBound)
                        char *currObj1 = (getDataByInternalId(candidate_id));
#ifdef COUNT_DIST_TIME
                        StopW stopw = StopW();
#endif
                        dist_t dist = fstdistfunc_(data_point, currObj1, dist_func_param_);
#ifdef COUNT_DIST_TIME
                        adsampling::distance_time += stopw.getElapsedTimeMicro();
#endif                  

                        adsampling::tot_full_dist++;
                        if (top_candidates.size() < ef || lowerBound > dist) {                      
                            candidate_set.emplace(-dist, candidate_id);
                            
                            top_candidates.emplace(dist, candidate_id);

                            if (top_candidates.size() > ef)
                                top_candidates.pop();

                            if (!top_candidates.empty())
                                lowerBound = top_candidates.top().first;
                        }

                        //candidates.emplace_back(dist, candidate_id);
                    }
                }

                adsampling::out_edge_for_visited_vector.push_back(out_edges);
                                                                                                                        
                // std::sort(candidates.begin(), candidates.end());
                // int candidate_cnt = candidates.size();
                // if(flag) {
                //     for(int i = 0; i < candidate_cnt / 2; i++) {
                //         dist_t dist = candidates[i].first;
                //         tableint candidate_id = candidates[i].second;
                //         if (top_candidates.size() < ef || lowerBound > dist) {                      
                //             candidate_set.emplace(-dist, candidate_id);

                //             top_candidates.emplace(dist, candidate_id);

                //             if (top_candidates.size() > ef)
                //                 top_candidates.pop();

                //             if (!top_candidates.empty())
                //                 lowerBound = top_candidates.top().first;
                //         }
                //         visited_array[candidate_id] = visited_array_tag;
                //         adsampling::tot_full_dist++;
                //         cnt_visit++;
                //     }
                // } else {
                //     for(int i = candidate_cnt / 2;i < candidate_cnt; i++) {
                //         dist_t dist = candidates[i].first;
                //         tableint candidate_id = candidates[i].second;
                //         if (top_candidates.size() < ef || lowerBound > dist) {                      
                //             candidate_set.emplace(-dist, candidate_id);
                            
                //             top_candidates.emplace(dist, candidate_id);

                //             if (top_candidates.size() > ef)
                //                 top_candidates.pop();

                //             if (!top_candidates.empty())
                //                 lowerBound = top_candidates.top().first;
                //         }
                //         visited_array[candidate_id] = visited_array_tag;
                //         adsampling::tot_full_dist++;
                //         cnt_visit++;
                //     }
                // }
            
            
            }

            // int sum_hops = 0;
            // auto new_top_candidates = top_candidates;
            // while(!new_top_candidates.size() > 100) {
            //     new_top_candidates.top();
            // }

            // while (!new_top_candidates.empty()) {
            //     auto top_candidate = new_top_candidates.top();
            //     new_top_candidates.pop();
            //     sum_hops += counts[top_candidate.second];
            // }
            // adsampling::avg_hop += (float)sum_hops / top_candidates.size();

            adsampling::tot_dist_calculation += cnt_visit;
            visited_list_pool_->releaseVisitedList(vl);
            return top_candidates;
        }

        template <bool has_deletions, bool collect_metrics=false>
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>>
        searchBaseLayerAD(tableint ep_id, const void *data_point, size_t ef) const {
            VisitedList *vl = visited_list_pool_->getFreeVisitedList();
            vl_type *visited_array = vl->mass;
            vl_type visited_array_tag = vl->curV;

            // top_candidates - the result set R
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>> top_candidates;
            // candidate_set  - the search set S
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidate_set;

            dist_t lowerBound;
            // Insert the entry point to the result and search set with its exact distance as a key. 
            if (!has_deletions || !isMarkedDeleted(ep_id)) {
#ifdef COUNT_DIST_TIME
                StopW stopw = StopW();
#endif
                dist_t dist = fstdistfunc_(data_point, getDataByInternalId(ep_id), dist_func_param_);
#ifdef COUNT_DIST_TIME
                adsampling::distance_time += stopw.getElapsedTimeMicro();
#endif          
                adsampling::tot_dist_calculation++;
                adsampling::tot_full_dist ++;
                lowerBound = dist;
                top_candidates.emplace(dist, ep_id);
                candidate_set.emplace(-dist, ep_id);
            } 
            else {
                lowerBound = std::numeric_limits<dist_t>::max();
                candidate_set.emplace(-lowerBound, ep_id);
            }

            visited_array[ep_id] = visited_array_tag;
            int cnt_visit = 0;

            // Iteratively generate candidates and conduct DCOs to maintain the result set R.
            while (!candidate_set.empty()) {
                std::pair<dist_t, tableint> current_node_pair = candidate_set.top();
                
                // When the smallest object in S has its distance larger than the largest in R, terminate the algorithm.
                if ((-current_node_pair.first) > lowerBound && (top_candidates.size() == ef || has_deletions == false)) {
                    break;
                }
                candidate_set.pop();

                // Fetch the smallest object in S. 
                tableint current_node_id = current_node_pair.second;
                int *data = (int *) get_linklist0(current_node_id);
                size_t size = getListCount((linklistsizeint*)data);
                if(collect_metrics){
                    metric_hops++;
                    metric_distance_computations+=size;
                }

                // Enumerate all the neighbors of the object and view them as candidates of KNNs. 
                for (size_t j = 1; j <= size; j++) {
                    int candidate_id = *(data + j);
                    if (!(visited_array[candidate_id] == visited_array_tag)) {
                        cnt_visit ++;
                        visited_array[candidate_id] = visited_array_tag;

                        // If the result set is not full, then calculate the exact distance. 
                        // (i.e., assume the distance threshold to be infinity)
                        if (top_candidates.size() < ef){
                            char *currObj1 = (getDataByInternalId(candidate_id));
#ifdef COUNT_DIST_TIME
                            StopW stopw = StopW();
#endif
                            dist_t dist = fstdistfunc_(data_point, currObj1, dist_func_param_);    
#ifdef COUNT_DIST_TIME
                            adsampling::distance_time += stopw.getElapsedTimeMicro();
#endif                                         
                            adsampling::tot_full_dist ++;
                            if (!has_deletions || !isMarkedDeleted(candidate_id))
                                candidate_set.emplace(-dist, candidate_id);
                            if (!has_deletions || !isMarkedDeleted(candidate_id))
                                top_candidates.emplace(dist, candidate_id);
                            if (!top_candidates.empty())
                                lowerBound = top_candidates.top().first;
                        }
                        // Otherwise, conduct DCO with ADSampling wrt the N_ef th NN. 
                        else {
#ifdef COUNT_DIST_TIME
                            StopW stopw = StopW();
#endif                            
                            dist_t dist = adsampling::dist_comp(lowerBound, getDataByInternalId(candidate_id), data_point, 0, 0);
#ifdef COUNT_DIST_TIME
                            adsampling::distance_time += stopw.getElapsedTimeMicro();
#endif                               
                            if(dist >= 0){
                                candidate_set.emplace(-dist, candidate_id);
                                if (!has_deletions || !isMarkedDeleted(candidate_id))
                                    top_candidates.emplace(dist, candidate_id);
                                if (top_candidates.size() > ef)
                                    top_candidates.pop();
                                if (!top_candidates.empty())
                                    lowerBound = top_candidates.top().first;
                            }
                        }
                    }
                }
            }
            adsampling::tot_dist_calculation += cnt_visit;
            visited_list_pool_->releaseVisitedList(vl);
            return top_candidates;
        }

        template <bool has_deletions, bool collect_metrics=false>
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>>
        searchBaseLayerADstar(tableint ep_id, const void *data_point, size_t ef, size_t k) const {
            VisitedList *vl = visited_list_pool_->getFreeVisitedList();
            vl_type *visited_array = vl->mass;
            vl_type visited_array_tag = vl->curV;

            // answers        - the KNN set R1
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>> answers;
            // top_candidates - the result set R2
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>> top_candidates;
            // candidate_set  - the search set S
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidate_set;
            
            dist_t lowerBound;
            dist_t lowerBoundcan;
            // Insert the entry point to the result and search set with its exact distance as a key. 
            if (!has_deletions || !isMarkedDeleted(ep_id)) {
#ifdef COUNT_DIST_TIME
                StopW stopw = StopW();
#endif                   
                dist_t dist = fstdistfunc_(data_point, getDataByInternalId(ep_id), dist_func_param_);
#ifdef COUNT_DIST_TIME
                adsampling::distance_time += stopw.getElapsedTimeMicro();
#endif          
                adsampling::tot_dist_calculation++;          
                adsampling::tot_full_dist ++;
                lowerBound = dist;
                lowerBoundcan = dist;
                answers.emplace(dist, ep_id);
                top_candidates.emplace(dist, ep_id);
                candidate_set.emplace(-dist, ep_id);
            } 
            else {
                lowerBound = std::numeric_limits<dist_t>::max();
                lowerBoundcan = std::numeric_limits<dist_t>::max();
                candidate_set.emplace(-lowerBound, ep_id);
            }

            visited_array[ep_id] = visited_array_tag;
            int cnt_visit = 0;
            // Iteratively generate candidates.
            while (!candidate_set.empty()) {
                std::pair<dist_t, tableint> current_node_pair = candidate_set.top();

                // When the smallest object in S has its distance larger than the largest in R2, terminate the algorithm.
                if ((-current_node_pair.first) > top_candidates.top().first && (top_candidates.size() == ef || has_deletions == false)) {
                    break;
                }
                candidate_set.pop();

                // Fetch the smallest object in S. 
                tableint current_node_id = current_node_pair.second;
                int *data = (int *) get_linklist0(current_node_id);
                size_t size = getListCount((linklistsizeint*)data);
                if(collect_metrics){
                    metric_hops++;
                    metric_distance_computations+=size;
                }


                // Enumerate all the neighbors of the object and view them as candidates of KNNs. 
                for (size_t j = 1; j <= size; j++) {
                    int candidate_id = *(data + j);
                    if (!(visited_array[candidate_id] == visited_array_tag)) {
                        cnt_visit ++;
                        visited_array[candidate_id] = visited_array_tag;


                        // If the KNN set is not full, then calculate the exact distance. (i.e., assume the distance threshold to be infinity)
                        if (answers.size() < k){
                            char *currObj1 = (getDataByInternalId(candidate_id));
#ifdef COUNT_DIST_TIME
                            StopW stopw = StopW();
#endif                            
                            dist_t dist = fstdistfunc_(data_point, currObj1, dist_func_param_);    
#ifdef COUNT_DIST_TIME
                            adsampling::distance_time += stopw.getElapsedTimeMicro();
#endif                             
                            adsampling::tot_full_dist ++;
                            if (!has_deletions || !isMarkedDeleted(candidate_id)){
                                candidate_set.emplace(-dist, candidate_id);
                                top_candidates.emplace(dist, candidate_id);
                                answers.emplace(dist, candidate_id);
                            }
                            if (!answers.empty())
                                lowerBound = answers.top().first;
                            if (!top_candidates.empty())
                                lowerBoundcan = top_candidates.top().first;
                        }
                        // Otherwise, conduct DCO with ADSampling wrt the Kth NN. 
                        else {
                            char *currObj1 = (getDataByInternalId(candidate_id));
#ifdef COUNT_DIST_TIME
                            StopW stopw = StopW();
#endif                            
                            dist_t dist = adsampling::dist_comp(lowerBound, currObj1, data_point, 0, 0);
#ifdef COUNT_DIST_TIME
                            adsampling::distance_time += stopw.getElapsedTimeMicro();
#endif                              
                            // If it's a positive object, then include it in R1, R2, S. 
                            if(dist >= 0){
                                candidate_set.emplace(-dist, candidate_id);
                                if(!has_deletions || !isMarkedDeleted(candidate_id)){
                                    top_candidates.emplace(dist, candidate_id);
                                    answers.emplace(dist, candidate_id);
                                }
                                if(top_candidates.size() > ef)
                                    top_candidates.pop();
                                if(answers.size() > k)
                                    answers.pop();

                                if (!answers.empty())
                                    lowerBound = answers.top().first;
                                if (!top_candidates.empty())
                                    lowerBoundcan = top_candidates.top().first;
                            }
                            // If it's a negative object, then update R2, S with the approximate distance.
                            else{
                                if(top_candidates.size() < ef || lowerBoundcan > -dist){
                                    top_candidates.emplace(-dist, candidate_id);
                                    candidate_set.emplace(dist, candidate_id);
                                }
                                if(top_candidates.size() > ef){
                                    top_candidates.pop();
                                }
                                if (!top_candidates.empty())
                                    lowerBoundcan = top_candidates.top().first;
                            }
                        }
                    }
                }
            }
            adsampling::tot_dist_calculation += cnt_visit;
            visited_list_pool_->releaseVisitedList(vl);
            return answers;
        }
        
        void getNeighborsByHeuristic2(
                std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> &top_candidates,
        const size_t M) {
            if (top_candidates.size() < M) {
                return;
            }

            std::priority_queue<std::pair<dist_t, tableint>> queue_closest;
            std::vector<std::pair<dist_t, tableint>> return_list;
            while (top_candidates.size() > 0) {
                queue_closest.emplace(-top_candidates.top().first, top_candidates.top().second);
                top_candidates.pop();
            }

            while (queue_closest.size()) {
                if (return_list.size() >= M)
                    break;
                std::pair<dist_t, tableint> curent_pair = queue_closest.top();
                dist_t dist_to_query = -curent_pair.first;
                queue_closest.pop();
                bool good = true;

                for (std::pair<dist_t, tableint> second_pair : return_list) {
                    dist_t curdist =
                            fstdistfunc_(getDataByInternalId(second_pair.second),
                                         getDataByInternalId(curent_pair.second),
                                         dist_func_param_);;
                    if (curdist < dist_to_query) {
                        good = false;
                        break;
                    }
                }
                if (good) {
                    return_list.push_back(curent_pair);
                }
            }

            for (std::pair<dist_t, tableint> curent_pair : return_list) {
                top_candidates.emplace(-curent_pair.first, curent_pair.second);
            }
        }


        linklistsizeint *get_linklist0(tableint internal_id) const {
            return (linklistsizeint *) (data_level0_memory_ + internal_id * size_data_per_element_ + offsetLevel0_);
        };

        linklistsizeint *get_linklist0(tableint internal_id, char *data_level0_memory_) const {
            return (linklistsizeint *) (data_level0_memory_ + internal_id * size_data_per_element_ + offsetLevel0_);
        };

        linklistsizeint *get_linklist(tableint internal_id, int level) const {
            return (linklistsizeint *) (linkLists_[internal_id] + (level - 1) * size_links_per_element_);
        };

        linklistsizeint *get_linklist_at_level(tableint internal_id, int level) const {
            return level == 0 ? get_linklist0(internal_id) : get_linklist(internal_id, level);
        };

        tableint mutuallyConnectNewElement(const void *data_point, tableint cur_c,
                                       std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> &top_candidates,
        int level, bool isUpdate) {
            size_t Mcurmax = level ? maxM_ : maxM0_;
            getNeighborsByHeuristic2(top_candidates, M_);
            if (top_candidates.size() > M_)
                throw std::runtime_error("Should be not be more than M_ candidates returned by the heuristic");

            std::vector<tableint> selectedNeighbors;
            selectedNeighbors.reserve(M_);
            while (top_candidates.size() > 0) {
                selectedNeighbors.push_back(top_candidates.top().second);
                top_candidates.pop();
            }

            tableint next_closest_entry_point = selectedNeighbors.back();

            {
                linklistsizeint *ll_cur;
                if (level == 0)
                    ll_cur = get_linklist0(cur_c);
                else
                    ll_cur = get_linklist(cur_c, level);

                if (*ll_cur && !isUpdate) {
                    throw std::runtime_error("The newly inserted element should have blank link list");
                }
                setListCount(ll_cur,selectedNeighbors.size());
                tableint *data = (tableint *) (ll_cur + 1);
                for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
                    if (data[idx] && !isUpdate)
                        throw std::runtime_error("Possible memory corruption");
                    if (level > element_levels_[selectedNeighbors[idx]])
                        throw std::runtime_error("Trying to make a link on a non-existent level");

                    data[idx] = selectedNeighbors[idx];

                }
            }

            for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {

                std::unique_lock <std::mutex> lock(link_list_locks_[selectedNeighbors[idx]]);

                linklistsizeint *ll_other;
                if (level == 0)
                    ll_other = get_linklist0(selectedNeighbors[idx]);
                else
                    ll_other = get_linklist(selectedNeighbors[idx], level);

                size_t sz_link_list_other = getListCount(ll_other);

                if (sz_link_list_other > Mcurmax)
                    throw std::runtime_error("Bad value of sz_link_list_other");
                if (selectedNeighbors[idx] == cur_c)
                    throw std::runtime_error("Trying to connect an element to itself");
                if (level > element_levels_[selectedNeighbors[idx]])
                    throw std::runtime_error("Trying to make a link on a non-existent level");

                tableint *data = (tableint *) (ll_other + 1);

                bool is_cur_c_present = false;
                if (isUpdate) {
                    for (size_t j = 0; j < sz_link_list_other; j++) {
                        if (data[j] == cur_c) {
                            is_cur_c_present = true;
                            break;
                        }
                    }
                }

                // If cur_c is already present in the neighboring connections of `selectedNeighbors[idx]` then no need to modify any connections or run the heuristics.
                if (!is_cur_c_present) {
                    if (sz_link_list_other < Mcurmax) {
                        data[sz_link_list_other] = cur_c;
                        setListCount(ll_other, sz_link_list_other + 1);
                    } else {
                        // finding the "weakest" element to replace it with the new one
                        dist_t d_max = fstdistfunc_(getDataByInternalId(cur_c), getDataByInternalId(selectedNeighbors[idx]),
                                                    dist_func_param_);
                        // Heuristic:
                        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidates;
                        candidates.emplace(d_max, cur_c);

                        for (size_t j = 0; j < sz_link_list_other; j++) {
                            candidates.emplace(
                                    fstdistfunc_(getDataByInternalId(data[j]), getDataByInternalId(selectedNeighbors[idx]),
                                                 dist_func_param_), data[j]);
                        }

                        getNeighborsByHeuristic2(candidates, Mcurmax);

                        int indx = 0;
                        while (candidates.size() > 0) {
                            data[indx] = candidates.top().second;
                            candidates.pop();
                            indx++;
                        }

                        setListCount(ll_other, indx);
                        // Nearest K:
                        /*int indx = -1;
                        for (int j = 0; j < sz_link_list_other; j++) {
                            dist_t d = fstdistfunc_(getDataByInternalId(data[j]), getDataByInternalId(rez[idx]), dist_func_param_);
                            if (d > d_max) {
                                indx = j;
                                d_max = d;
                            }
                        }
                        if (indx >= 0) {
                            data[indx] = cur_c;
                        } */
                    }
                }
            }

            return next_closest_entry_point;
        }

        std::mutex global;
        size_t ef_;

        void setEf(size_t ef) {
            ef_ = ef;
        }


        std::priority_queue<std::pair<dist_t, tableint>> searchKnnInternal(void *query_data, int k) {
            std::priority_queue<std::pair<dist_t, tableint  >> top_candidates;
            if (cur_element_count == 0) return top_candidates;
            tableint currObj = enterpoint_node_;
            dist_t curdist = fstdistfunc_(query_data, getDataByInternalId(enterpoint_node_), dist_func_param_);

            for (size_t level = maxlevel_; level > 0; level--) {
                bool changed = true;
                while (changed) {
                    changed = false;
                    int *data;
                    data = (int *) get_linklist(currObj,level);
                    int size = getListCount(data);
                    tableint *datal = (tableint *) (data + 1);
                    for (int i = 0; i < size; i++) {
                        tableint cand = datal[i];
                        if (cand < 0 || cand > max_elements_)
                            throw std::runtime_error("cand error");
                        dist_t d = fstdistfunc_(query_data, getDataByInternalId(cand), dist_func_param_);

                        if (d < curdist) {
                            curdist = d;
                            currObj = cand;
                            changed = true;
                        }
                    }
                }
            }

            if (num_deleted_) {
                std::priority_queue<std::pair<dist_t, tableint  >> top_candidates1=searchBaseLayerST<true>(currObj, query_data,
                                                                                                           ef_);
                top_candidates.swap(top_candidates1);
            }
            else{
                std::priority_queue<std::pair<dist_t, tableint  >> top_candidates1=searchBaseLayerST<false>(currObj, query_data,
                                                                                                            ef_);
                top_candidates.swap(top_candidates1);
            }

            while (top_candidates.size() > k) {
                top_candidates.pop();
            }
            return top_candidates;
        };

        void resizeIndex(size_t new_max_elements){
            if (new_max_elements<cur_element_count)
                throw std::runtime_error("Cannot resize, max element is less than the current number of elements");


            delete visited_list_pool_;
            visited_list_pool_ = new VisitedListPool(1, new_max_elements);


            element_levels_.resize(new_max_elements);

            std::vector<std::mutex>(new_max_elements).swap(link_list_locks_);

            // Reallocate base layer
            char * data_level0_memory_new = (char *) realloc(data_level0_memory_, new_max_elements * size_data_per_element_);
            if (data_level0_memory_new == nullptr)
                throw std::runtime_error("Not enough memory: resizeIndex failed to allocate base layer");
            data_level0_memory_ = data_level0_memory_new;

            // Reallocate all other layers
            char ** linkLists_new = (char **) realloc(linkLists_, sizeof(void *) * new_max_elements);
            if (linkLists_new == nullptr)
                throw std::runtime_error("Not enough memory: resizeIndex failed to allocate other layers");
            linkLists_ = linkLists_new;

            max_elements_ = new_max_elements;
        }

        void saveIndex(const std::string &location) {
            std::ofstream output(location, std::ios::binary);
            std::streampos position;

            writeBinaryPOD(output, offsetLevel0_);
            writeBinaryPOD(output, max_elements_);
            writeBinaryPOD(output, cur_element_count);
            writeBinaryPOD(output, size_data_per_element_);
            writeBinaryPOD(output, label_offset_);
            writeBinaryPOD(output, offsetData_);
            writeBinaryPOD(output, maxlevel_);
            writeBinaryPOD(output, enterpoint_node_);
            writeBinaryPOD(output, maxM_);

            writeBinaryPOD(output, maxM0_);
            writeBinaryPOD(output, M_);
            writeBinaryPOD(output, mult_);
            writeBinaryPOD(output, ef_construction_);

            output.write(data_level0_memory_, cur_element_count * size_data_per_element_);

            for (size_t i = 0; i < cur_element_count; i++) {
                unsigned int linkListSize = element_levels_[i] > 0 ? size_links_per_element_ * element_levels_[i] : 0;
                writeBinaryPOD(output, linkListSize);
                if (linkListSize)
                    output.write(linkLists_[i], linkListSize);
            }
            output.close();
        }

        void loadIndex(const std::string &location, SpaceInterface<dist_t> *s, size_t max_elements_i=0) {
            std::ifstream input(location, std::ios::binary);

            if (!input.is_open())
                throw std::runtime_error("Cannot open file");

            // get file size:
            input.seekg(0,input.end);
            std::streampos total_filesize=input.tellg();
            input.seekg(0,input.beg);

            readBinaryPOD(input, offsetLevel0_);
            readBinaryPOD(input, max_elements_);
            readBinaryPOD(input, cur_element_count);

            size_t max_elements = max_elements_i;
            if(max_elements < cur_element_count)
                max_elements = max_elements_;
            max_elements_ = max_elements;
            readBinaryPOD(input, size_data_per_element_);
            readBinaryPOD(input, label_offset_);
            readBinaryPOD(input, offsetData_);
            readBinaryPOD(input, maxlevel_);
            readBinaryPOD(input, enterpoint_node_);

            readBinaryPOD(input, maxM_);
            readBinaryPOD(input, maxM0_);
            readBinaryPOD(input, M_);
            readBinaryPOD(input, mult_);
            readBinaryPOD(input, ef_construction_);


            data_size_ = s->get_data_size();
            fstdistfunc_ = s->get_dist_func();
            dist_func_param_ = s->get_dist_func_param();

            auto pos=input.tellg();


            /// Optional - check if index is ok:

            input.seekg(cur_element_count * size_data_per_element_,input.cur);
            for (size_t i = 0; i < cur_element_count; i++) {
                if(input.tellg() < 0 || input.tellg()>=total_filesize){
                    throw std::runtime_error("Index seems to be corrupted or unsupported");
                }

                unsigned int linkListSize;
                readBinaryPOD(input, linkListSize);
                if (linkListSize != 0) {
                    input.seekg(linkListSize,input.cur);
                }
            }

            // throw exception if it either corrupted or old index
            if(input.tellg()!=total_filesize)
                throw std::runtime_error("Index seems to be corrupted or unsupported");

            input.clear();

            /// Optional check end

            input.seekg(pos,input.beg);

            data_level0_memory_ = (char *) malloc(max_elements * size_data_per_element_);
            if (data_level0_memory_ == nullptr)
                throw std::runtime_error("Not enough memory: loadIndex failed to allocate level0");
            input.read(data_level0_memory_, cur_element_count * size_data_per_element_);

            size_links_per_element_ = maxM_ * sizeof(tableint) + sizeof(linklistsizeint);

            size_links_level0_ = maxM0_ * sizeof(tableint) + sizeof(linklistsizeint);
            std::vector<std::mutex>(max_elements).swap(link_list_locks_);
            std::vector<std::mutex>(max_update_element_locks).swap(link_list_update_locks_);

            visited_list_pool_ = new VisitedListPool(1, max_elements);

            linkLists_ = (char **) malloc(sizeof(void *) * max_elements);
            if (linkLists_ == nullptr)
                throw std::runtime_error("Not enough memory: loadIndex failed to allocate linklists");
            element_levels_ = std::vector<int>(max_elements);
            revSize_ = 1.0 / mult_;
            ef_ = 10;
            for (size_t i = 0; i < cur_element_count; i++) {
                label_lookup_[getExternalLabel(i)]=i;
                unsigned int linkListSize;
                readBinaryPOD(input, linkListSize);
                if (linkListSize == 0) {
                    element_levels_[i] = 0;

                    linkLists_[i] = nullptr;
                } else {
                    element_levels_[i] = linkListSize / size_links_per_element_;
                    linkLists_[i] = (char *) malloc(linkListSize);
                    if (linkLists_[i] == nullptr)
                        throw std::runtime_error("Not enough memory: loadIndex failed to allocate linklist");
                    input.read(linkLists_[i], linkListSize);
                }
            }

            for (size_t i = 0; i < cur_element_count; i++) {
                if(isMarkedDeleted(i))
                    num_deleted_ += 1;
            }

            input.close();

            return;
        }

        template<typename data_t>
        std::vector<data_t> getDataByLabel(labeltype label) const
        {
            tableint label_c;
            auto search = label_lookup_.find(label);
            if (search == label_lookup_.end() || isMarkedDeleted(search->second)) {
                throw std::runtime_error("Label not found");
            }
            label_c = search->second;

            char* data_ptrv = getDataByInternalId(label_c);
            size_t dim = *((size_t *) dist_func_param_);
            std::vector<data_t> data;
            data_t* data_ptr = (data_t*) data_ptrv;
            for (int i = 0; i < dim; i++) {
                data.push_back(*data_ptr);
                data_ptr += 1;
            }
            return data;
        }

        static const unsigned char DELETE_MARK = 0x01;
        // static const unsigned char REUSE_MARK = 0x10;
        /**
         * Marks an element with the given label deleted, does NOT really change the current graph.
         * @param label
         */
        void markDelete(labeltype label)
        {
            auto search = label_lookup_.find(label);
            if (search == label_lookup_.end()) {
                throw std::runtime_error("Label not found");
            }
            tableint internalId = search->second;
            markDeletedInternal(internalId);
        }

        /**
         * Uses the first 8 bits of the memory for the linked list to store the mark,
         * whereas maxM0_ has to be limited to the lower 24 bits, however, still large enough in almost all cases.
         * @param internalId
         */
        void markDeletedInternal(tableint internalId) {
            assert(internalId < cur_element_count);
            if (!isMarkedDeleted(internalId))
            {
                unsigned char *ll_cur = ((unsigned char *)get_linklist0(internalId))+2;
                *ll_cur |= DELETE_MARK;
                num_deleted_ += 1;
            }
            else
            {
                throw std::runtime_error("The requested to delete element is already deleted");
            }
        }

        /**
         * Remove the deleted mark of the node, does NOT really change the current graph.
         * @param label
         */
        void unmarkDelete(labeltype label)
        {
            auto search = label_lookup_.find(label);
            if (search == label_lookup_.end()) {
                throw std::runtime_error("Label not found");
            }
            tableint internalId = search->second;
            unmarkDeletedInternal(internalId);
        }

        /**
         * Remove the deleted mark of the node.
         * @param internalId
         */
        void unmarkDeletedInternal(tableint internalId) {
            assert(internalId < cur_element_count);
            if (isMarkedDeleted(internalId))
            {
                unsigned char *ll_cur = ((unsigned char *)get_linklist0(internalId))+2;
                *ll_cur &= ~DELETE_MARK;
                num_deleted_ -= 1;
            }
            else
            {
                throw std::runtime_error("The requested to undelete element is not deleted");
            }
        }

        /**
         * Checks the first 8 bits of the memory to see if the element is marked deleted.
         * @param internalId
         * @return
         */
        bool isMarkedDeleted(tableint internalId) const {
            unsigned char *ll_cur = ((unsigned char*)get_linklist0(internalId))+2;
            return *ll_cur & DELETE_MARK;
        }

        unsigned short int getListCount(linklistsizeint * ptr) const {
            return *((unsigned short int *)ptr);
        }

        void setListCount(linklistsizeint * ptr, unsigned short int size) const {
            *((unsigned short int*)(ptr))=*((unsigned short int *)&size);
        }

        void addPoint(const void *data_point, labeltype label) {
            addPoint(data_point, label,-1);
        }

        void updatePoint(const void *dataPoint, tableint internalId, float updateNeighborProbability) {
            // update the feature vector associated with existing point with new vector
            memcpy(getDataByInternalId(internalId), dataPoint, data_size_);

            int maxLevelCopy = maxlevel_;
            tableint entryPointCopy = enterpoint_node_;
            // If point to be updated is entry point and graph just contains single element then just return.
            if (entryPointCopy == internalId && cur_element_count == 1)
                return;

            int elemLevel = element_levels_[internalId];
            std::uniform_real_distribution<float> distribution(0.0, 1.0);
            for (int layer = 0; layer <= elemLevel; layer++) {
                std::unordered_set<tableint> sCand;
                std::unordered_set<tableint> sNeigh;
                std::vector<tableint> listOneHop = getConnectionsWithLock(internalId, layer);
                if (listOneHop.size() == 0)
                    continue;

                sCand.insert(internalId);

                for (auto&& elOneHop : listOneHop) {
                    sCand.insert(elOneHop);

                    if (distribution(update_probability_generator_) > updateNeighborProbability)
                        continue;

                    sNeigh.insert(elOneHop);

                    std::vector<tableint> listTwoHop = getConnectionsWithLock(elOneHop, layer);
                    for (auto&& elTwoHop : listTwoHop) {
                        sCand.insert(elTwoHop);
                    }
                }

                for (auto&& neigh : sNeigh) {
                    // if (neigh == internalId)
                    //     continue;

                    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidates;
                    size_t size = sCand.find(neigh) == sCand.end() ? sCand.size() : sCand.size() - 1; // sCand guaranteed to have size >= 1
                    size_t elementsToKeep = std::min(ef_construction_, size);
                    for (auto&& cand : sCand) {
                        if (cand == neigh)
                            continue;

                        dist_t distance = fstdistfunc_(getDataByInternalId(neigh), getDataByInternalId(cand), dist_func_param_);
                        if (candidates.size() < elementsToKeep) {
                            candidates.emplace(distance, cand);
                        } else {
                            if (distance < candidates.top().first) {
                                candidates.pop();
                                candidates.emplace(distance, cand);
                            }
                        }
                    }

                    // Retrieve neighbours using heuristic and set connections.
                    getNeighborsByHeuristic2(candidates, layer == 0 ? maxM0_ : maxM_);

                    {
                        std::unique_lock <std::mutex> lock(link_list_locks_[neigh]);
                        linklistsizeint *ll_cur;
                        ll_cur = get_linklist_at_level(neigh, layer);
                        size_t candSize = candidates.size();
                        setListCount(ll_cur, candSize);
                        tableint *data = (tableint *) (ll_cur + 1);
                        for (size_t idx = 0; idx < candSize; idx++) {
                            data[idx] = candidates.top().second;
                            candidates.pop();
                        }
                    }
                }
            }

            repairConnectionsForUpdate(dataPoint, entryPointCopy, internalId, elemLevel, maxLevelCopy);
        };

        void repairConnectionsForUpdate(const void *dataPoint, tableint entryPointInternalId, tableint dataPointInternalId, int dataPointLevel, int maxLevel) {
            tableint currObj = entryPointInternalId;
            if (dataPointLevel < maxLevel) {
                dist_t curdist = fstdistfunc_(dataPoint, getDataByInternalId(currObj), dist_func_param_);
                for (int level = maxLevel; level > dataPointLevel; level--) {
                    bool changed = true;
                    while (changed) {
                        changed = false;
                        unsigned int *data;
                        std::unique_lock <std::mutex> lock(link_list_locks_[currObj]);
                        data = get_linklist_at_level(currObj,level);
                        int size = getListCount(data);
                        tableint *datal = (tableint *) (data + 1);
#ifdef USE_SSE
                        _mm_prefetch(getDataByInternalId(*datal), _MM_HINT_T0);
#endif
                        for (int i = 0; i < size; i++) {
#ifdef USE_SSE
                            _mm_prefetch(getDataByInternalId(*(datal + i + 1)), _MM_HINT_T0);
#endif
                            tableint cand = datal[i];
                            dist_t d = fstdistfunc_(dataPoint, getDataByInternalId(cand), dist_func_param_);
                            if (d < curdist) {
                                curdist = d;
                                currObj = cand;
                                changed = true;
                            }
                        }
                    }
                }
            }

            if (dataPointLevel > maxLevel)
                throw std::runtime_error("Level of item to be updated cannot be bigger than max level");

            for (int level = dataPointLevel; level >= 0; level--) {
                std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> topCandidates = searchBaseLayer(
                        currObj, dataPoint, level);

                std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> filteredTopCandidates;
                while (topCandidates.size() > 0) {
                    if (topCandidates.top().second != dataPointInternalId)
                        filteredTopCandidates.push(topCandidates.top());

                    topCandidates.pop();
                }

                // Since element_levels_ is being used to get `dataPointLevel`, there could be cases where `topCandidates` could just contains entry point itself.
                // To prevent self loops, the `topCandidates` is filtered and thus can be empty.
                if (filteredTopCandidates.size() > 0) {
                    bool epDeleted = isMarkedDeleted(entryPointInternalId);
                    if (epDeleted) {
                        filteredTopCandidates.emplace(fstdistfunc_(dataPoint, getDataByInternalId(entryPointInternalId), dist_func_param_), entryPointInternalId);
                        if (filteredTopCandidates.size() > ef_construction_)
                            filteredTopCandidates.pop();
                    }

                    currObj = mutuallyConnectNewElement(dataPoint, dataPointInternalId, filteredTopCandidates, level, true);
                }
            }
        }

        std::vector<tableint> getConnectionsWithLock(tableint internalId, int level) {
            std::unique_lock <std::mutex> lock(link_list_locks_[internalId]);
            unsigned int *data = get_linklist_at_level(internalId, level);
            int size = getListCount(data);
            std::vector<tableint> result(size);
            tableint *ll = (tableint *) (data + 1);
            memcpy(result.data(), ll,size * sizeof(tableint));
            return result;
        };

        tableint addPoint(const void *data_point, labeltype label, int level) {

            tableint cur_c = 0;
            {
                // Checking if the element with the same label already exists
                // if so, updating it *instead* of creating a new element.
                std::unique_lock <std::mutex> templock_curr(cur_element_count_guard_);
                auto search = label_lookup_.find(label);
                if (search != label_lookup_.end()) {
                    tableint existingInternalId = search->second;
                    templock_curr.unlock();

                    std::unique_lock <std::mutex> lock_el_update(link_list_update_locks_[(existingInternalId & (max_update_element_locks - 1))]);

                    if (isMarkedDeleted(existingInternalId)) {
                        unmarkDeletedInternal(existingInternalId);
                    }
                    updatePoint(data_point, existingInternalId, 1.0);
                    
                    return existingInternalId;
                }

                if (cur_element_count >= max_elements_) {
                    throw std::runtime_error("The number of elements exceeds the specified limit");
                };

                cur_c = cur_element_count;
                cur_element_count++;
                label_lookup_[label] = cur_c;
            }

            // Take update lock to prevent race conditions on an element with insertion/update at the same time.
            std::unique_lock <std::mutex> lock_el_update(link_list_update_locks_[(cur_c & (max_update_element_locks - 1))]);
            std::unique_lock <std::mutex> lock_el(link_list_locks_[cur_c]);
            int curlevel = getRandomLevel(mult_);
            if (level > 0)
                curlevel = level;

            element_levels_[cur_c] = curlevel;


            std::unique_lock <std::mutex> templock(global);
            int maxlevelcopy = maxlevel_;
            if (curlevel <= maxlevelcopy)
                templock.unlock();
            tableint currObj = enterpoint_node_;
            tableint enterpoint_copy = enterpoint_node_;


            memset(data_level0_memory_ + cur_c * size_data_per_element_ + offsetLevel0_, 0, size_data_per_element_);

            // Initialisation of the dat and labela
            memcpy(getExternalLabeLp(cur_c), &label, sizeof(labeltype));
            memcpy(getDataByInternalId(cur_c), data_point, data_size_);


            if (curlevel) {
                linkLists_[cur_c] = (char *) malloc(size_links_per_element_ * curlevel + 1);
                if (linkLists_[cur_c] == nullptr)
                    throw std::runtime_error("Not enough memory: addPoint failed to allocate linklist");
                memset(linkLists_[cur_c], 0, size_links_per_element_ * curlevel + 1);
            }

            if ((signed)currObj != -1) {

                if (curlevel < maxlevelcopy) {

                    dist_t curdist = fstdistfunc_(data_point, getDataByInternalId(currObj), dist_func_param_);
                    for (int level = maxlevelcopy; level > curlevel; level--) {


                        bool changed = true;
                        while (changed) {
                            changed = false;
                            unsigned int *data;
                            std::unique_lock <std::mutex> lock(link_list_locks_[currObj]);
                            data = get_linklist(currObj,level);
                            int size = getListCount(data);

                            tableint *datal = (tableint *) (data + 1);
                            for (int i = 0; i < size; i++) {
                                tableint cand = datal[i];
                                if (cand < 0 || cand > max_elements_)
                                    throw std::runtime_error("cand error");
                                dist_t d = fstdistfunc_(data_point, getDataByInternalId(cand), dist_func_param_);
                                if (d < curdist) {
                                    curdist = d;
                                    currObj = cand;
                                    changed = true;
                                }
                            }
                        }
                    }
                }

                bool epDeleted = isMarkedDeleted(enterpoint_copy);
                for (int level = std::min(curlevel, maxlevelcopy); level >= 0; level--) {
                    if (level > maxlevelcopy || level < 0)  // possible?
                        throw std::runtime_error("Level error");

                    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates = searchBaseLayer(
                            currObj, data_point, level);
                    if (epDeleted) {
                        top_candidates.emplace(fstdistfunc_(data_point, getDataByInternalId(enterpoint_copy), dist_func_param_), enterpoint_copy);
                        if (top_candidates.size() > ef_construction_)
                            top_candidates.pop();
                    }
                    currObj = mutuallyConnectNewElement(data_point, cur_c, top_candidates, level, false);
                }


            } else {
                // Do nothing for the first element
                enterpoint_node_ = 0;
                maxlevel_ = curlevel;

            }

            //Releasing lock for the maximum level
            if (curlevel > maxlevelcopy) {
                enterpoint_node_ = cur_c;
                maxlevel_ = curlevel;
            }
            return cur_c;
        };

        // max heap
        std::priority_queue<std::pair<dist_t, labeltype >>
        searchKnn(void *query_data, size_t k, int adaptive=0) const {
            
            std::priority_queue<std::pair<dist_t, labeltype >> result;
            if (cur_element_count == 0) return result;

            tableint currObj = enterpoint_node_;
#ifdef COUNT_DIST_TIME
            StopW stopw = StopW();
#endif
            dist_t curdist = fstdistfunc_(query_data, getDataByInternalId(enterpoint_node_), dist_func_param_);
#ifdef COUNT_DIST_TIME
            adsampling::distance_time += stopw.getElapsedTimeMicro();
#endif
            adsampling::tot_dist_calculation ++;

            // StopW stopw = StopW();

            for (int level = maxlevel_; level > 0; level--) {
                
                bool changed = true;
                while (changed) {
                    
                    changed = false;
                    unsigned int *data;

                    data = (unsigned int *) get_linklist(currObj, level);
                    int size = getListCount(data);
                    metric_hops++;
                    metric_distance_computations+=size;

                    tableint *datal = (tableint *) (data + 1);
                    for (int i = 0; i < size; i++) {
                        tableint cand = datal[i];
                        if (cand < 0 || cand > max_elements_)
                            throw std::runtime_error("cand error");
                        adsampling::tot_dist_calculation ++;
                        if(adaptive){
#ifdef COUNT_DIST_TIME
                            StopW stopw = StopW();
#endif
                            dist_t d = adsampling::dist_comp(curdist, getDataByInternalId(cand), query_data, 0, 0);
#ifdef COUNT_DIST_TIME
                            adsampling::distance_time += stopw.getElapsedTimeMicro();
#endif
                            if(d > 0){
                                curdist = d;
                                currObj = cand;
                                changed = true;
                            }
                        }
                        else {
#ifdef COUNT_DIST_TIME
                            StopW stopw = StopW();
#endif
                            dist_t d = fstdistfunc_(query_data, getDataByInternalId(cand), dist_func_param_);
#ifdef COUNT_DIST_TIME
                            adsampling::distance_time += stopw.getElapsedTimeMicro();
#endif
                            adsampling::tot_full_dist ++;
                            if (d < curdist) {
                                curdist = d;
                                currObj = cand;
                                changed = true;
                            }
                        }
                    }
                }
            }
            
            // adsampling::time1 += stopw.getElapsedTimeMicro();
            
            // max heap
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>> top_candidates;
            
            if (num_deleted_) {
                if(adaptive == 1) top_candidates=searchBaseLayerADstar<true,true>(currObj, query_data, std::max(ef_, k), k);
                else if(adaptive == 2) top_candidates=searchBaseLayerAD<true,true>(currObj, query_data, std::max(ef_, k));
                else top_candidates=searchBaseLayerST<true,true>(currObj, query_data, std::max(ef_, k));
            }
            else{
                if(adaptive == 1) top_candidates=searchBaseLayerADstar<true,true>(currObj, query_data, std::max(ef_, k), k);
                else if(adaptive == 2) top_candidates=searchBaseLayerAD<true,true>(currObj, query_data, std::max(ef_, k));
                else {
                    // stopw = StopW();
                    top_candidates=searchBaseLayerST<false,true>(currObj, query_data, std::max(ef_, k));
                    // adsampling::time2 += stopw.getElapsedTimeMicro();
                }
            }

            while (top_candidates.size() > k) {
                top_candidates.pop();
            }
            while (top_candidates.size() > 0) {
                std::pair<dist_t, tableint> rez = top_candidates.top();
                result.push(std::pair<dist_t, labeltype>(rez.first, getExternalLabel(rez.second)));
                top_candidates.pop();
            }
            return result;
        };

        void checkIntegrity(){
            int connections_checked=0;
            std::vector <int > inbound_connections_num(cur_element_count,0);
            for(int i = 0;i < cur_element_count; i++){
                for(int l = 0;l <= element_levels_[i]; l++){
                    linklistsizeint *ll_cur = get_linklist_at_level(i,l);
                    int size = getListCount(ll_cur);
                    tableint *data = (tableint *) (ll_cur + 1);
                    std::unordered_set<tableint> s;
                    for (int j=0; j<size; j++){
                        assert(data[j] > 0);
                        assert(data[j] < cur_element_count);
                        assert (data[j] != i);
                        inbound_connections_num[data[j]]++;
                        s.insert(data[j]);
                        connections_checked++;

                    }
                    assert(s.size() == size);
                }
            }
            if(cur_element_count > 1){
                int min1=inbound_connections_num[0], max1=inbound_connections_num[0];
                for(int i=0; i < cur_element_count; i++){
                    assert(inbound_connections_num[i] > 0);
                    min1=std::min(inbound_connections_num[i],min1);
                    max1=std::max(inbound_connections_num[i],max1);
                }
                std::cout << "Min inbound: " << min1 << ", Max inbound:" << max1 << "\n";
            }
            std::cout << "integrity ok, checked " << connections_checked << " connections\n";
        }

        // Compute arc flags for each node based on cluster IDs
        void computeClusterFlags(const std::vector<uint32_t>& cluster_ids, size_t depth) {
            if (cluster_ids.size() < cur_element_count) {
                throw std::runtime_error("Number of cluster IDs is less than the number of elements in the index");
            }

            // Find the maximum cluster ID to determine the number of clusters
            uint32_t max_cluster_id = 0;
            for (size_t i = 0; i < cur_element_count; i++) {
                if (cluster_ids[i] > max_cluster_id) {
                    max_cluster_id = cluster_ids[i];
                }
            }
            size_t num_clusters = max_cluster_id + 1;
            
            std::cout << "Found " << num_clusters << " clusters" << std::endl;
            
            // Initialize cluster_flags_ with empty vectors for each node
            cluster_flags_.clear();
            cluster_flags_.resize(cur_element_count);
            
            std::cout << "Computing cluster flags with OpenMP..." << std::endl;
            
            // Parallelize the BFS computation with OpenMP
            #pragma omp parallel for schedule(dynamic)
            for (size_t i = 0; i < cur_element_count; i++) {
                if (i % 10000 == 0) {
                    #pragma omp critical
                    {
                        std::cout << "Processing node " << i << "/" << cur_element_count << std::endl;
                    }
                }
                
                // Skip deleted nodes
                if (isMarkedDeleted(i)) {
                    continue;
                }
                
                // Mark the node's own cluster as reachable
                uint32_t node_cluster = cluster_ids[i];
                cluster_flags_[i].push_back(node_cluster);
                
                // BFS to find reachable clusters within the specified depth
                std::queue<std::pair<tableint, size_t>> q; // (node, current_depth)
                std::unordered_set<tableint> visited;
                std::unordered_set<uint32_t> reachable_clusters;
                
                // Add node's own cluster
                reachable_clusters.insert(node_cluster);
                
                q.push({i, 0});
                visited.insert(i);
                
                while (!q.empty()) {
                    auto [current, current_depth] = q.front();
                    q.pop();
                    
                    // If we've reached the maximum depth, stop exploring further
                    if (current_depth >= depth) {
                        continue;
                    }
                    
                    // Get neighbors at level 0 (the most detailed level)
                    linklistsizeint *ll_cur = get_linklist0(current);
                    int size = getListCount(ll_cur);
                    tableint *neighbors = (tableint *) (ll_cur + 1);
                    
                    // Process each neighbor
                    for (int j = 0; j < size; j++) {
                        tableint neighbor = neighbors[j];
                        
                        // Skip deleted nodes
                        if (isMarkedDeleted(neighbor)) {
                            continue;
                        }
                        
                        // Mark the neighbor's cluster as reachable from the starting node
                        uint32_t neighbor_cluster = cluster_ids[neighbor];
                        
                        // Add to reachable clusters if not already added
                        if (reachable_clusters.find(neighbor_cluster) == reachable_clusters.end()) {
                            reachable_clusters.insert(neighbor_cluster);
                        }
                        
                        // If we haven't visited this neighbor yet, add it to the queue
                        if (visited.find(neighbor) == visited.end()) {
                            visited.insert(neighbor);
                            q.push({neighbor, current_depth + 1});
                        }
                    }
                }
                
                // Convert set to vector and store in cluster_flags_
                cluster_flags_[i].reserve(reachable_clusters.size());
                for (const auto& cluster : reachable_clusters) {
                    if (std::find(cluster_flags_[i].begin(), cluster_flags_[i].end(), cluster) == cluster_flags_[i].end()) {
                        cluster_flags_[i].push_back(cluster);
                    }
                }
                
                // Sort for faster lookups
                std::sort(cluster_flags_[i].begin(), cluster_flags_[i].end());
            }
            
            std::cout << "Finished computing cluster flags" << std::endl;
        }
        
        // Save the computed cluster flags to a file
        void saveFlags(const std::string &filename) {
            if (cluster_flags_.empty()) {
                throw std::runtime_error("Cluster flags have not been computed yet");
            }
            
            std::ofstream output(filename, std::ios::binary);
            if (!output.is_open()) {
                throw std::runtime_error("Cannot open file for writing: " + filename);
            }
            
            // Write header information
            size_t num_nodes = cluster_flags_.size();
            output.write(reinterpret_cast<char*>(&num_nodes), sizeof(size_t));
            
            // Write cluster flags for each node
            for (size_t i = 0; i < num_nodes; i++) {
                // Write the number of clusters for this node
                size_t num_clusters = cluster_flags_[i].size();
                output.write(reinterpret_cast<char*>(&num_clusters), sizeof(size_t));
                
                // Write the actual cluster IDs
                if (num_clusters > 0) {
                    output.write(reinterpret_cast<const char*>(cluster_flags_[i].data()), num_clusters * sizeof(uint32_t));
                }
            }
            
            output.close();
            std::cout << "Saved cluster flags to " << filename << std::endl;
        }
        
        // Load cluster flags from a file
        void loadFlags(const std::string &filename) {
            std::ifstream input(filename, std::ios::binary);
            if (!input.is_open()) {
                throw std::runtime_error("Cannot open file for reading: " + filename);
            }
            
            // Read header information
            size_t num_nodes;
            input.read(reinterpret_cast<char*>(&num_nodes), sizeof(size_t));
            
            // Verify that the number of nodes matches our index
            if (num_nodes != cur_element_count) {
                std::cout << "Warning: Number of nodes in flags file (" << num_nodes 
                          << ") does not match the current index (" << cur_element_count 
                          << "). Using the smaller of the two." << std::endl;
                num_nodes = std::min(num_nodes, cur_element_count);
            }
            
            // Clear and resize the cluster_flags_ vector
            cluster_flags_.clear();
            cluster_flags_.resize(cur_element_count);
            
            // Read cluster flags for each node
            for (size_t i = 0; i < num_nodes; i++) {
                // Read the number of clusters for this node
                size_t num_clusters;
                input.read(reinterpret_cast<char*>(&num_clusters), sizeof(size_t));
                
                // Read the actual cluster IDs
                if (num_clusters > 0) {
                    cluster_flags_[i].resize(num_clusters);
                    input.read(reinterpret_cast<char*>(cluster_flags_[i].data()), num_clusters * sizeof(uint32_t));
                }
            }
            
            // Check if we've read the entire file
            if (!input.eof() && input.fail()) {
                std::cout << "Warning: Error occurred while reading flags file." << std::endl;
            }
            
            input.close();
        }

        // Load centroids from a .fvecs file
        void loadCentroids(const std::string &filename) {
            std::ifstream input(filename, std::ios::binary);
            if (!input.is_open()) {
                throw std::runtime_error("Cannot open centroid file for reading: " + filename);
            }
            
            // Read dimension from first 4 bytes
            int dim;
            input.read(reinterpret_cast<char*>(&dim), sizeof(int));
            
            // Get file size to calculate number of vectors
            input.seekg(0, std::ios::end);
            size_t fileSize = input.tellg();
            input.seekg(0, std::ios::beg);
            
            // Each vector has dim + 1 floats (4 bytes each)
            size_t bytesPerVector = (dim + 1) * sizeof(float);
            size_t num_vectors = fileSize / bytesPerVector;
            
            // Clean up existing centroids if any
            if (centroids_ != nullptr) {
                for (size_t i = 0; i < num_centroids_; i++) {
                    if (centroids_[i] != nullptr) {
                        free(centroids_[i]);
                    }
                }
                free(centroids_);
            }
            
            // Allocate memory for centroids
            centroids_ = (float**)malloc(num_vectors * sizeof(float*));
            if (centroids_ == nullptr) {
                throw std::runtime_error("Failed to allocate memory for centroids");
            }
            
            // Initialize all pointers to nullptr
            for (size_t i = 0; i < num_vectors; i++) {
                centroids_[i] = nullptr;
            }
            
            num_centroids_ = num_vectors;
            centroid_dim_ = dim;
            
            // Temporary buffer for reading each centroid
            std::vector<float> buf(dim + 1);
            
            // Read each centroid
            for (size_t i = 0; i < num_vectors; i++) {
                input.read(reinterpret_cast<char*>(buf.data()), (dim + 1) * sizeof(float));
                
                // Allocate memory for this centroid
                centroids_[i] = (float*)malloc(dim * sizeof(float));
                if (centroids_[i] == nullptr) {
                    throw std::runtime_error("Failed to allocate memory for centroid");
                }
                
                // Copy data (skip the first dimension value)
                memcpy(centroids_[i], buf.data() + 1, dim * sizeof(float));
            }
            
            has_centroids_ = true;
            input.close();
            std::cout << "Loaded " << num_vectors << " centroids with dimension " << dim << std::endl;
        }
        
        // Get centroid for a specific cluster ID
        const float* getCentroid(size_t cluster_id) const {
            if (!has_centroids_) {
                throw std::runtime_error("No centroids loaded");
            }
            
            if (cluster_id >= num_centroids_) {
                throw std::runtime_error("Cluster ID out of range");
            }
            
            return centroids_[cluster_id];
        }
        
        // Get centroid dimension
        size_t getCentroidDim() const {
            return centroid_dim_;
        }
        
        // Get number of centroids
        size_t getNumCentroids() const {
            return num_centroids_;
        }
        
        // Check if centroids are loaded
        bool hasCentroids() const {
            return has_centroids_;
        }

        void collectLayer1Vertices() {
            layer1_vertices_.clear();
            for (size_t i = 0; i < cur_element_count; i++) {
                if (element_levels_[i] >= 1) {
                    layer1_vertices_.push_back(i);
                }
            }   
        }

        auto analyzePrunedCandidates(HierarchicalNSW<float> &appr_alg,
                             int k_hop,
                             const std::vector<tableint>& result,
                             const std::vector<tableint>& gt) {
            std::unordered_set<tableint> found(result.begin(), result.end());
            // 1. Collect missed labels
            std::unordered_set<tableint> missed;
            for (tableint id : gt) {
                if (found.find(id) == found.end()) {
                    missed.insert(id);
                }
            }

            std::unordered_map<tableint, int> label_to_min_hop;
            std::unordered_set<tableint> visited;
            std::queue<std::pair<tableint, int>> q;

            // 2. Initialize multi-source BFS from all pruned vertices
            for (tableint start : appr_alg.pruned_vertices_) {
                q.push({start, 0});
                visited.insert(start);
            }

            while (!q.empty()) {
                auto [cur, hop] = q.front();
                q.pop();

                if (hop > k_hop) continue;

                if (missed.find(cur) != missed.end()) {
                    if (!label_to_min_hop.count(cur) || hop < label_to_min_hop[cur]) {
                        label_to_min_hop[cur] = hop;
                    }
                }

                if (hop < k_hop) {
                    int* data = (int*)appr_alg.get_linklist0(cur);
                    size_t size = appr_alg.getListCount((linklistsizeint*)data);
                    tableint* neighbors = (tableint*)(data + 1);
                    for (size_t j = 0; j < size; ++j) {
                        tableint nb = neighbors[j];
                        if (visited.insert(nb).second) {
                            q.push({nb, hop + 1});
                        }
                    }
                }
            }

            // 3. 统计 hop 分布
            std::unordered_map<int, int> hop_hit_count;
            int unreachable = 0;

            for (tableint miss_id : missed) {
                if (label_to_min_hop.count(miss_id)) {
                    hop_hit_count[label_to_min_hop[miss_id]]++;
                } else {
                    unreachable++;
                }
            }

            for (int h = 0; h < 8; h++) {
                adsampling::hit_by_pruned[h] += hop_hit_count[h];
            }
        }

        void loadTopkClusters(const std::string &filename, int topk) {
            topk_clusters_ = (size_t)topk;

            std::ifstream input(filename, std::ios::binary);
            if (!input.is_open()) {
                throw std::runtime_error("Cannot open topk clusters file for reading: " + filename);
            }

            int num_clusters = num_centroids_;
            size_t vec_index = 0;

            cluster_flag_width_ = (num_clusters + 31) / 32;
            topk_cluster_flags_.resize(cur_element_count, std::vector<uint32_t>(cluster_flag_width_, 0));

            while (input.read(reinterpret_cast<char*>(&num_clusters), sizeof(int))) {
                if (num_clusters <= 0) {
                    throw std::runtime_error("Invalid number of clusters in topk cluster file");
                }

                if (vec_index >= cur_element_count) {
                    throw std::runtime_error("Top-k cluster file has more entries than base vectors");
                }

                std::vector<int> cluster_ids(num_clusters);
                input.read(reinterpret_cast<char*>(cluster_ids.data()), sizeof(int) * num_clusters);

                // Only take the first topk clusters
                size_t clusters_to_process = std::min((size_t)topk, (size_t)num_clusters);
                
                // Map only the nearest topk cluster_ids to bitmask
                for (size_t i = 0; i < clusters_to_process; i++) {
                    int cluster_id = cluster_ids[i];
                    if (cluster_id < 0 || cluster_id >= num_clusters) {
                        throw std::runtime_error("Cluster ID out of range");
                    }
                    size_t word = cluster_id / 32;
                    size_t bit = cluster_id % 32;
                    topk_cluster_flags_[vec_index][word] |= (1U << bit);
                }

                vec_index++;
            }

            if (vec_index != cur_element_count) {
                std::cerr << "Warning: topk cluster file has " << vec_index 
                        << " entries, but index has " << cur_element_count << " base vectors." << std::endl;
            }

            input.close();
        }


        mutable std::vector<std::vector<uint32_t>> topk_cluster_flags_;
        mutable size_t cluster_flag_width_ = 0;
        mutable size_t topk_clusters_ = 0;

        mutable tableint entry_id_ = 0;
        mutable std::vector<uint32_t> query_nearest_centroids_;

        mutable std::vector<tableint> pruned_vertices_;
        mutable std::vector<tableint> entry_ids_;

        mutable std::vector<tableint> layer1_vertices_;

        // Storage for cluster flags - for each node, stores which clusters are reachable
        // Now using a vector of vectors instead of a bitmap
        mutable std::vector<std::vector<uint32_t>> cluster_flags_;
        
        // Storage for centroids data
        float** centroids_ = nullptr;
        size_t num_centroids_ = 0;
        size_t centroid_dim_ = 0;
        bool has_centroids_ = false;
    };
}
