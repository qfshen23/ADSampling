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

def compute_overlap_ratio(clusters1, clusters2, k):
    """Compute overlap ratio between two top-k cluster assignments"""
    set1 = set(clusters1[:k])
    set2 = set(clusters2[:k])
    intersection = len(set1.intersection(set2))
    return intersection / k

def compute_statistics(overlap_ratios):
    """Compute min, 25%, mean, 75%, max statistics"""
    ratios = np.array(overlap_ratios)
    return {
        'min': np.min(ratios),
        'p25': np.percentile(ratios, 25),
        'mean': np.mean(ratios),
        'p75': np.percentile(ratios, 75),
        'max': np.max(ratios)
    }

def compute_recall_by_overlap(probe_vector_ids, overlap_ratios, gt_vector_ids, top_x):
    """Compute recall among top-x vectors sorted by overlap ratio"""
    # Sort probe vectors by overlap ratio (descending)
    sorted_indices = np.argsort(overlap_ratios)[::-1]
    sorted_probe_ids = [probe_vector_ids[i] for i in sorted_indices]
    
    # Take top-x vectors
    top_x_vectors = set(sorted_probe_ids[:top_x])
    
    # Count how many GT vectors are in top-x
    gt_set = set(gt_vector_ids)
    overlap_count = len(top_x_vectors.intersection(gt_set))
    
    # Recall = overlap_count / total_gt_neighbors
    recall = overlap_count / len(gt_vector_ids)
    
    return recall

def main():
    datasets = ['sift']  # List of datasets to process
    K = 1024 * 16  # Number of clusters
    k_overlap = 64  # Top-k clusters for overlap computation
    nprobe = 80  # Number of nearest clusters to probe
    top_x_values = [500, 1000, 2000, 3000, 4000]  # Different x values for recall computation
    gt_neighbors = 10  # Number of ground truth neighbors to consider

    for dataset in datasets:
        print(f"\n=== Processing dataset: {dataset} ===")
        
        # File paths
        base_path = f'/data/vector_datasets/{dataset}'
        query_path = f'{base_path}/{dataset}_query.fvecs'
        gt_path = f'{base_path}/{dataset}_groundtruth_10000.ivecs'
        centroids_path = f'{base_path}/{dataset}_centroid_{K}.fvecs'
        top_clusters_path = f'{base_path}/{dataset}_top_clusters_1024_of_{K}.ivecs'
        cluster_ids_path = f'{base_path}/{dataset}_cluster_id_{K}.ivecs'
        
        # Skip if files don't exist
        missing_files = [p for p in [query_path, gt_path, centroids_path, top_clusters_path, cluster_ids_path] if not os.path.exists(p)]
        if missing_files:
            print(f"Skipping {dataset} - {missing_files} missing")
            continue
        
        print("Loading data...")
        queries = read_fvecs(query_path)
        groundtruth = read_ivecs(gt_path)[:,:gt_neighbors]  # Only take top-K ground truth neighbors
        centroids = read_fvecs(centroids_path)
        top_clusters = read_ivecs(top_clusters_path)  # Each row: sorted cluster IDs by distance
        cluster_ids = read_ivecs(cluster_ids_path)    # Each row: assigned cluster ID
        
        print(f"Queries shape: {queries.shape}")
        print(f"Groundtruth shape: {groundtruth.shape}")
        print(f"Centroids shape: {centroids.shape}")
        print(f"Top clusters shape: {top_clusters.shape}")
        print(f"Cluster IDs shape: {cluster_ids.shape}")
        
        num_queries = queries.shape[0]
        
        # Store recall results for averaging
        recall_results = {x: [] for x in top_x_values}
        
        print("\nComputing query-probe vector overlaps...")
        for query_idx in tqdm(range(min(1000, num_queries))):  # Process first 100 queries
            query = queries[query_idx:query_idx+1]
            
            # Compute distances from query to all centroids
            distances = np.sum((query - centroids) ** 2, axis=1)
            nearest_clusters = np.argsort(distances)[:nprobe] 
            
            # Find all vectors in these nearest clusters
            probe_vector_ids = []
            for cluster_id in nearest_clusters:
                vectors_in_cluster = np.where(cluster_ids.flatten() == cluster_id)[0]
                probe_vector_ids.extend(vectors_in_cluster)
            
            # Get query's top-k cluster assignment
            query_distances_to_centroids = distances
            query_top_clusters = np.argsort(query_distances_to_centroids)
            
            # Compute overlap ratios for all probe vectors
            overlap_ratios = []
            for vector_id in probe_vector_ids:
                vector_top_clusters = top_clusters[vector_id]
                overlap_ratio = compute_overlap_ratio(query_top_clusters, vector_top_clusters, k_overlap)
                overlap_ratios.append(overlap_ratio)
            
            if overlap_ratios:
                stats = compute_statistics(overlap_ratios)
                print(f"Query {query_idx} - Probe vectors overlap: "
                      f"min={stats['min']:.4f}, p25={stats['p25']:.4f}, "
                      f"mean={stats['mean']:.4f}, p75={stats['p75']:.4f}, max={stats['max']:.4f}")
            
            # Compute overlap ratios for ground truth vectors
            gt_overlap_ratios = []
            gt_vector_ids = groundtruth[query_idx]
            
            for gt_id in gt_vector_ids:
                if gt_id < len(top_clusters):  # Ensure valid index
                    gt_top_clusters = top_clusters[gt_id]
                    overlap_ratio = compute_overlap_ratio(query_top_clusters, gt_top_clusters, k_overlap)
                    gt_overlap_ratios.append(overlap_ratio)
            
            # if gt_overlap_ratios:
            #     gt_stats = compute_statistics(gt_overlap_ratios)
            #     print(f"Query {query_idx} - GT vectors overlap: "
            #           f"min={gt_stats['min']:.4f}, p25={gt_stats['p25']:.4f}, "
            #           f"mean={gt_stats['mean']:.4f}, p75={gt_stats['p75']:.4f}, max={gt_stats['max']:.4f}")
            
            # Compute recall by overlap ratio for different top-x values
            if overlap_ratios and len(probe_vector_ids) > 0:
                print(f"Query {query_idx} - Recall by overlap (top-x most similar by overlap):")
                for top_x in top_x_values:
                    if top_x <= len(probe_vector_ids):
                        recall = compute_recall_by_overlap(probe_vector_ids, overlap_ratios, gt_vector_ids, top_x)
                        recall_results[top_x].append(recall)
                        print(f"  Top-{top_x}: {recall:.4f}")
                    else:
                        print(f"  Top-{top_x}: Not enough probe vectors (only {len(probe_vector_ids)})")
            
            print()
        
        # Print average recall results for this dataset
        print(f"\nAverage recall results for {dataset}:")
        for top_x in top_x_values:
            if recall_results[top_x]:
                avg_recall = np.mean(recall_results[top_x])
                print(f"Average recall for top-{top_x}: {avg_recall:.4f}")

if __name__ == '__main__':
    main()
