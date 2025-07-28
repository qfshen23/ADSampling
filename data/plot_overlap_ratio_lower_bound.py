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

def main():
    datasets = ['sift']
    K = 1024  # Number of clusters
    k_overlap = 64  # Top-k clusters for overlap computation
    nprobe = 50  # Number of nearest clusters to probe
    gt_neighbors = 100  # Number of ground truth neighbors to consider
    
    for dataset in datasets:
        print(f"\n=== Processing dataset: {dataset} ===")
        
        # File paths
        base_path = f'/data/vector_datasets/{dataset}'
        query_path = f'{base_path}/{dataset}_query.fvecs'
        gt_path = f'{base_path}/{dataset}_groundtruth_10000.ivecs'
        centroids_path = f'{base_path}/{dataset}_centroid_{K}.fvecs'
        top_clusters_path = f'{base_path}/{dataset}_top_clusters_1024.ivecs'
        cluster_ids_path = f'{base_path}/{dataset}_cluster_id_{K}.ivecs'
        
        # Skip if files don't exist
        missing_files = [p for p in [query_path, gt_path, centroids_path, top_clusters_path, cluster_ids_path] if not os.path.exists(p)]
        if missing_files:
            print(f"Skipping {dataset} - {missing_files} missing")
            continue
        
        print("Loading data...")
        queries = read_fvecs(query_path)
        groundtruth = read_ivecs(gt_path)[:,:gt_neighbors]
        centroids = read_fvecs(centroids_path)
        top_clusters = read_ivecs(top_clusters_path)
        cluster_ids = read_ivecs(cluster_ids_path)
        
        print(f"Queries shape: {queries.shape}")
        print(f"Groundtruth shape: {groundtruth.shape}")
        print(f"Centroids shape: {centroids.shape}")
        print(f"Top clusters shape: {top_clusters.shape}")
        print(f"Cluster IDs shape: {cluster_ids.shape}")
        
        num_queries = queries.shape[0]
        
        # Statistics for overlap ratio thresholds
        p50_above_counts = []
        p80_above_counts = []
        p100_above_counts = []
        total_probe_vectors = []
        
        print("\nComputing overlap ratio statistics...")
        for query_idx in tqdm(range(min(1000, num_queries))):
            query = queries[query_idx:query_idx+1]
            
            # Compute distances from query to all centroids
            distances = np.sum((query - centroids) ** 2, axis=1)
            nearest_clusters = np.argsort(distances)[:nprobe]
            query_top_clusters = np.argsort(distances)
            
            # Get ground truth vectors for this query
            gt_vector_ids = groundtruth[query_idx]
            
            # Compute overlap ratios for ground truth vectors
            gt_overlap_ratios = []
            for gt_id in gt_vector_ids:
                if gt_id < len(top_clusters):
                    gt_top_clusters = top_clusters[gt_id]
                    overlap_ratio = compute_overlap_ratio(query_top_clusters, gt_top_clusters, k_overlap)
                    gt_overlap_ratios.append(overlap_ratio)
            
            if not gt_overlap_ratios:
                continue
                
            # Sort GT overlap ratios and get percentiles
            gt_overlap_ratios = np.array(gt_overlap_ratios)
            gt_overlap_ratios_sorted = np.sort(gt_overlap_ratios)
            
            # Get 50th, 80th and 100th (max) percentile thresholds
            p50_threshold = np.percentile(gt_overlap_ratios_sorted, 50)
            p80_threshold = np.percentile(gt_overlap_ratios_sorted, 80)
            p100_threshold = np.max(gt_overlap_ratios_sorted)
            
            # Find all vectors in probed clusters
            probe_vector_ids = []
            for cluster_id in nearest_clusters:
                vectors_in_cluster = np.where(cluster_ids.flatten() == cluster_id)[0]
                probe_vector_ids.extend(vectors_in_cluster)
            
            if not probe_vector_ids:
                continue
                
            total_probe_vectors.append(len(probe_vector_ids))
            
            # Compute overlap ratios for all probe vectors
            p50_count = 0
            p80_count = 0
            p100_count = 0
            
            for vector_id in probe_vector_ids:
                if vector_id < len(top_clusters):
                    vector_top_clusters = top_clusters[vector_id]
                    overlap_ratio = compute_overlap_ratio(query_top_clusters, vector_top_clusters, k_overlap)
                    
                    if overlap_ratio >= p50_threshold:
                        p50_count += 1
                    if overlap_ratio >= p80_threshold:
                        p80_count += 1
                    if overlap_ratio >= p100_threshold:
                        p100_count += 1
            
            p50_above_counts.append(p50_count)
            p80_above_counts.append(p80_count)
            p100_above_counts.append(p100_count)
            
            if query_idx % 100 == 0:
                print(f"Query {query_idx}: p50_threshold={p50_threshold:.4f}, p80_threshold={p80_threshold:.4f}, p100_threshold={p100_threshold:.4f}")
                print(f"  Probe vectors >= p50: {p50_count}/{len(probe_vector_ids)} ({p50_count/len(probe_vector_ids):.4f})")
                print(f"  Probe vectors >= p80: {p80_count}/{len(probe_vector_ids)} ({p80_count/len(probe_vector_ids):.4f})")
                print(f"  Probe vectors >= p100: {p100_count}/{len(probe_vector_ids)} ({p100_count/len(probe_vector_ids):.4f})")
        
        # Compute final statistics
        if p50_above_counts and p80_above_counts and p100_above_counts and total_probe_vectors:
            p50_ratios = np.array(p50_above_counts) / np.array(total_probe_vectors)
            p80_ratios = np.array(p80_above_counts) / np.array(total_probe_vectors)
            p100_ratios = np.array(p100_above_counts) / np.array(total_probe_vectors)
            
            print(f"\n=== Final Statistics for {dataset} (nprobe={nprobe}) ===")
            print(f"Average ratio of probe vectors >= 50th percentile GT overlap: {np.mean(p50_ratios):.4f}")
            print(f"Average ratio of probe vectors >= 80th percentile GT overlap: {np.mean(p80_ratios):.4f}")
            print(f"Average ratio of probe vectors >= 100th percentile GT overlap: {np.mean(p100_ratios):.4f}")
            print(f"Std ratio of probe vectors >= 50th percentile GT overlap: {np.std(p50_ratios):.4f}")
            print(f"Std ratio of probe vectors >= 80th percentile GT overlap: {np.std(p80_ratios):.4f}")
            print(f"Std ratio of probe vectors >= 100th percentile GT overlap: {np.std(p100_ratios):.4f}")
            print(f"Average total probe vectors: {np.mean(total_probe_vectors):.1f}")

if __name__ == '__main__':
    main()
