import numpy as np
import struct
import os
from tqdm import tqdm

def read_fvecs(filename, c_contiguous=True):
    fv = np.fromfile(filename, dtype=np.float32)
    if fv.size == 0:
        return np.zeros((0, 0))
    dim = fv.view(np.int32)[0]
    assert dim > 0
    fv = fv.reshape(-1, 1 + dim)
    if not all(fv.view(np.int32)[:, 0] == dim):
        raise IOError("Non-uniform vector sizes in " + filename)
    fv = fv[:, 1:]
    if c_contiguous:
        fv = fv.copy()
    return fv

def read_ivecs(filename):
    fv = np.fromfile(filename, dtype=np.int32)
    if fv.size == 0:
        return np.zeros((0, 0))
    dim = fv[0]
    assert dim > 0
    fv = fv.reshape(-1, 1 + dim)
    if not all(fv[:, 0] == dim):
        raise IOError("Non-uniform vector sizes in " + filename)
    return fv[:, 1:]

def compute_multi_overlap_ratio(query_top_clusters, vector_top_clusters, k_list):
    """
    Compute overlap ratios for multiple k values.
    Returns a list of ratios for each k in k_list.
    """
    ratios = []
    for k in k_list:
        set1 = set(query_top_clusters[:k])
        set2 = set(vector_top_clusters[:k])
        ratios.append(len(set1 & set2) / k)
    return ratios

def main():
    datasets = ['sift']  # List of datasets to process
    K = 1024  # Number of clusters
    nprobe = 120  # Number of nearest clusters to probe
    k_list = [4, 8, 16, 32]  # Multi-k overlap
    top_x_values = [20000, 40000, 60000]  # For recall computation
    gt_neighbors = 10000  # Number of ground truth neighbors to consider

    for dataset in datasets:
        print(f"\n=== Processing dataset: {dataset} ===")
        
        # File paths
        base_path = f'/data/vector_datasets/{dataset}'
        query_path = f'{base_path}/{dataset}_query.fvecs'
        gt_path = f'{base_path}/{dataset}_groundtruth_10000.ivecs'
        centroids_path = f'{base_path}/{dataset}_centroid_{K}.fvecs'
        top_clusters_path = f'{base_path}/{dataset}_top_clusters_{K}.ivecs'
        cluster_ids_path = f'{base_path}/{dataset}_cluster_id_{K}.ivecs'
        
        # Skip if files don't exist
        missing_files = [p for p in [query_path, gt_path, centroids_path, top_clusters_path, cluster_ids_path] if not os.path.exists(p)]
        if missing_files:
            print(f"Skipping {dataset} - {missing_files} missing")
            continue
        
        print("Loading data...")
        queries = read_fvecs(query_path)
        groundtruth = read_ivecs(gt_path)[:, :gt_neighbors]
        centroids = read_fvecs(centroids_path)
        top_clusters = read_ivecs(top_clusters_path)  # Each row: sorted cluster IDs by distance
        cluster_ids = read_ivecs(cluster_ids_path)    # Each row: assigned cluster ID
        
        print(f"Queries shape: {queries.shape}")
        print(f"Groundtruth shape: {groundtruth.shape}")
        print(f"Centroids shape: {centroids.shape}")
        print(f"Top clusters shape: {top_clusters.shape}")
        print(f"Cluster IDs shape: {cluster_ids.shape}")
        
        num_queries = queries.shape[0]
        recall_results = {x: [] for x in top_x_values}
        
        print("\nComputing multi-k overlap scores and recall...")
        for query_idx in tqdm(range(min(1000, num_queries))):  # Process up to 1000 queries
            query = queries[query_idx:query_idx+1]
            
            # Compute distances from query to all centroids
            distances = np.sum((query - centroids) ** 2, axis=1)
            nearest_clusters = np.argsort(distances)[:nprobe]
            
            # Find all vectors in these nearest clusters
            probe_vector_ids = []
            for cluster_id in nearest_clusters:
                vectors_in_cluster = np.where(cluster_ids.flatten() == cluster_id)[0]
                probe_vector_ids.extend(vectors_in_cluster)
            
            # Get query's cluster ranking
            query_top_clusters = np.argsort(distances)
            
            # Compute score (sum of overlaps for k_list) for each probe vector
            scores = []
            for vector_id in probe_vector_ids:
                vector_top_clusters = top_clusters[vector_id]
                ratios = compute_multi_overlap_ratio(query_top_clusters, vector_top_clusters, k_list)
                scores.append(np.sum(ratios))  # or np.mean(ratios)
            
            gt_vector_ids = groundtruth[query_idx]
            
            if scores and len(probe_vector_ids) > 0:
                print(f"Query {query_idx} - Recall by score (sum of overlaps):")
                sorted_indices = np.argsort(scores)[::-1]
                sorted_probe_ids = [probe_vector_ids[i] for i in sorted_indices]

                for top_x in top_x_values:
                    if top_x <= len(sorted_probe_ids):
                        top_x_set = set(sorted_probe_ids[:top_x])
                        gt_set = set(gt_vector_ids)
                        recall = len(top_x_set & gt_set) / len(gt_vector_ids)
                        recall_results[top_x].append(recall)
                        print(f"  Top-{top_x}: {recall:.4f}")
                    else:
                        print(f"  Top-{top_x}: Not enough probe vectors (only {len(sorted_probe_ids)})")
                print()
        
        # Print average recall results for this dataset
        print(f"\nAverage recall results for {dataset}:")
        for top_x in top_x_values:
            if recall_results[top_x]:
                avg_recall = np.mean(recall_results[top_x])
                print(f"Average recall for top-{top_x}: {avg_recall:.4f}")

if __name__ == '__main__':
    main()
