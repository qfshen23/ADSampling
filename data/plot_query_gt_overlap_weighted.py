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

def compute_weighted_overlap_score(query_clusters, base_clusters, k, distance):
    """
    Weighted overlap score = overlap_ratio / distance
    """
    overlap = compute_overlap_ratio(query_clusters, base_clusters, k)
    return overlap / (distance + 1e-6)  # Add small epsilon to avoid division by zero

def compute_statistics(scores):
    scores = np.array(scores)
    return {
        'min': np.min(scores),
        'p25': np.percentile(scores, 25),
        'mean': np.mean(scores),
        'p75': np.percentile(scores, 75),
        'max': np.max(scores)
    }

def compute_recall_by_overlap(probe_vector_ids, overlap_scores, gt_vector_ids, top_x):
    sorted_indices = np.argsort(overlap_scores)[::-1]
    sorted_probe_ids = [probe_vector_ids[i] for i in sorted_indices]
    top_x_vectors = set(sorted_probe_ids[:top_x])
    gt_set = set(gt_vector_ids)
    overlap_count = len(top_x_vectors.intersection(gt_set))
    recall = overlap_count / len(gt_vector_ids)
    return recall

def main():
    datasets = ['sift']
    K = 1024
    k_overlap = 64
    nprobe = 120
    top_x_values = [20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000]
    gt_neighbors = 10000

    for dataset in datasets:
        print(f"\n=== Processing dataset: {dataset} ===")
        
        base_path = f'/data/vector_datasets/{dataset}'
        query_path = f'{base_path}/{dataset}_query.fvecs'
        gt_path = f'{base_path}/{dataset}_groundtruth_10000.ivecs'
        centroids_path = f'{base_path}/{dataset}_centroid_{K}.fvecs'
        top_clusters_path = f'{base_path}/{dataset}_top_clusters_{K}.ivecs'
        cluster_ids_path = f'{base_path}/{dataset}_cluster_id_{K}.ivecs'
        
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
        recall_results = {x: [] for x in top_x_values}
        
        print("\nComputing weighted overlaps...")
        for query_idx in tqdm(range(min(10000, num_queries))):
            query = queries[query_idx:query_idx+1]
            distances = np.sum((query - centroids) ** 2, axis=1)
            nearest_clusters = np.argsort(distances)[:nprobe] 
            
            probe_vector_ids = []
            cluster_distances = []  # Store distances to centroids
            for cluster_id in nearest_clusters:
                vectors_in_cluster = np.where(cluster_ids.flatten() == cluster_id)[0]
                probe_vector_ids.extend(vectors_in_cluster)
                cluster_distances.extend([distances[cluster_id]] * len(vectors_in_cluster))
            
            query_top_clusters = np.argsort(distances)
            
            overlap_scores = []
            for vector_id, distance in zip(probe_vector_ids, cluster_distances):
                vector_top_clusters = top_clusters[vector_id]
                score = compute_weighted_overlap_score(query_top_clusters, vector_top_clusters, k_overlap, distance)
                overlap_scores.append(score)
            
            if overlap_scores:
                stats = compute_statistics(overlap_scores)
                print(f"Query {query_idx} - Weighted score stats: "
                      f"min={stats['min']:.4f}, p25={stats['p25']:.4f}, "
                      f"mean={stats['mean']:.4f}, p75={stats['p75']:.4f}, max={stats['max']:.4f}")
            
            gt_overlap_scores = []
            gt_vector_ids = groundtruth[query_idx]
            for gt_id in gt_vector_ids:
                if gt_id < len(top_clusters):
                    gt_top_clusters = top_clusters[gt_id]
                    # For GT vectors, use minimum distance as we don't know actual centroid distance
                    min_distance = np.min(distances)
                    score = compute_weighted_overlap_score(query_top_clusters, gt_top_clusters, k_overlap, min_distance)
                    gt_overlap_scores.append(score)
            
            if overlap_scores and len(probe_vector_ids) > 0:
                print(f"Query {query_idx} - Recall by weighted overlap score:")
                for top_x in top_x_values:
                    if top_x <= len(probe_vector_ids):
                        recall = compute_recall_by_overlap(probe_vector_ids, overlap_scores, gt_vector_ids, top_x)
                        recall_results[top_x].append(recall)
                        print(f"  Top-{top_x}: {recall:.4f}")
                    else:
                        print(f"  Top-{top_x}: Not enough probe vectors (only {len(probe_vector_ids)})")
            print()
        
        print(f"\nAverage recall results for {dataset}:")
        for top_x in top_x_values:
            if recall_results[top_x]:
                avg_recall = np.mean(recall_results[top_x])
                print(f"Average recall for top-{top_x}: {avg_recall:.4f}")

if __name__ == '__main__':
    main()
