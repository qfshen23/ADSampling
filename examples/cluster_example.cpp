#include "../src/hnswlib/hnswlib.h"
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>

// Simple k-means clustering for demonstration
std::vector<uint8_t> k_means_clustering(std::vector<std::vector<float>>& data, int k) {
    int n = data.size();
    int dim = data[0].size();
    
    // Initialize centroids randomly
    std::vector<std::vector<float>> centroids(k);
    for (int i = 0; i < k; i++) {
        int random_idx = rand() % n;
        centroids[i] = data[random_idx];
    }
    
    // Assign cluster IDs
    std::vector<uint8_t> cluster_ids(n, 0);
    
    for (int iter = 0; iter < 10; iter++) {  // 10 iterations of k-means
        // Assign points to clusters
        for (int i = 0; i < n; i++) {
            float min_dist = std::numeric_limits<float>::max();
            int best_cluster = 0;
            
            for (int j = 0; j < k; j++) {
                float dist = 0;
                for (int d = 0; d < dim; d++) {
                    float diff = data[i][d] - centroids[j][d];
                    dist += diff * diff;
                }
                
                if (dist < min_dist) {
                    min_dist = dist;
                    best_cluster = j;
                }
            }
            
            cluster_ids[i] = best_cluster;
        }
        
        // Update centroids
        std::vector<int> counts(k, 0);
        std::vector<std::vector<float>> new_centroids(k, std::vector<float>(dim, 0));
        
        for (int i = 0; i < n; i++) {
            int cluster_id = cluster_ids[i];
            counts[cluster_id]++;
            for (int d = 0; d < dim; d++) {
                new_centroids[cluster_id][d] += data[i][d];
            }
        }
        
        for (int j = 0; j < k; j++) {
            if (counts[j] > 0) {
                for (int d = 0; d < dim; d++) {
                    centroids[j][d] = new_centroids[j][d] / counts[j];
                }
            }
        }
    }
    
    return cluster_ids;
}

int main() {
    // Set random seed
    srand(time(nullptr));
    
    // Parameters
    const int dim = 10;          // Dimension of vectors
    const int num_elements = 1000; // Number of elements
    const int num_clusters = 16;  // Number of clusters (should be <= 64)
    const int max_bfs_depth = 2;  // Maximum depth for BFS traversal
    
    // Generate random data
    std::vector<std::vector<float>> data(num_elements, std::vector<float>(dim));
    std::vector<float> flat_data(num_elements * dim);
    
    for (int i = 0; i < num_elements; i++) {
        for (int j = 0; j < dim; j++) {
            float val = static_cast<float>(rand()) / RAND_MAX;
            data[i][j] = val;
            flat_data[i * dim + j] = val;
        }
    }
    
    // Create an HNSW index
    hnswlib::L2Space space(dim);
    hnswlib::HierarchicalNSW<float> alg(&space, num_elements);
    
    // Add points to the index
    for (int i = 0; i < num_elements; i++) {
        alg.addPoint(&flat_data[i * dim], i);
    }
    
    std::cout << "Added " << num_elements << " elements to the index." << std::endl;
    
    // Run k-means clustering
    std::vector<uint8_t> cluster_ids = k_means_clustering(data, num_clusters);
    
    std::cout << "Assigned cluster IDs using k-means with " << num_clusters << " clusters." << std::endl;
    
    // Assign cluster IDs to vertices in the index
    alg.setClusterIds(cluster_ids);
    
    // Compute cluster flags using BFS
    alg.computeClusterFlags(max_bfs_depth);
    
    std::cout << "Computed cluster flags using BFS with max depth = " << max_bfs_depth << std::endl;
    
    // Save the index with cluster information
    alg.saveIndex("index_with_clusters.bin");
    
    std::cout << "Saved index with cluster information to 'index_with_clusters.bin'" << std::endl;
    
    // Load the index back from disk
    hnswlib::HierarchicalNSW<float> loaded_alg(&space, "index_with_clusters.bin");
    
    std::cout << "Loaded index from 'index_with_clusters.bin'" << std::endl;
    
    // Verify that the cluster information was saved correctly
    bool verification_passed = true;
    for (int i = 0; i < num_elements; i++) {
        uint8_t original_cluster_id = cluster_ids[i];
        uint8_t loaded_cluster_id = loaded_alg.getVertexClusterId(i);
        
        if (original_cluster_id != loaded_cluster_id) {
            std::cout << "Error: Cluster ID mismatch for vertex " << i << ": " 
                     << static_cast<int>(original_cluster_id) << " vs " 
                     << static_cast<int>(loaded_cluster_id) << std::endl;
            verification_passed = false;
            break;
        }
        
        // Print information for a few example vertices
        if (i < 5) {
            uint64_t flag = loaded_alg.getVertexClusterFlag(i);
            std::vector<uint8_t> present_clusters = hnswlib::HierarchicalNSW<float>::flagToClusterIds(flag);
            
            std::cout << "Vertex " << i << ": Cluster ID = " << static_cast<int>(loaded_cluster_id) 
                     << ", Flag = " << flag << ", Clusters in neighborhood: ";
            
            for (uint8_t cluster : present_clusters) {
                std::cout << static_cast<int>(cluster) << " ";
            }
            std::cout << std::endl;
        }
    }
    
    if (verification_passed) {
        std::cout << "Verification passed! Cluster information was saved and loaded correctly." << std::endl;
    } else {
        std::cout << "Verification failed!" << std::endl;
    }
    
    return 0;
} 