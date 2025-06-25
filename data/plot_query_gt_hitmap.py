import numpy as np
import struct
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

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
    set1 = set(clusters1[:k])
    set2 = set(clusters2[:k])
    intersection = len(set1.intersection(set2))
    return intersection / k

def get_hit_positions(query_top_clusters, base_top_clusters, k_overlap):
    """
    对于一个 base 向量，统计它的 top-k cluster（base_top_clusters[:k_overlap]）里，
    哪些 cluster 在 query_top_clusters[:k_overlap] 中出现，并返回这些位置的下标集合
    """
    query_top_k = query_top_clusters[:k_overlap]
    base_top_k = base_top_clusters[:k_overlap]
    query_cluster_to_rank = {cid: idx for idx, cid in enumerate(query_top_k)}
    hit_positions = []
    for cid in base_top_k:
        if cid in query_cluster_to_rank:
            hit_positions.append(query_cluster_to_rank[cid])
    return hit_positions

def main():
    datasets = ['sift']
    K = 1024
    k_overlaps = [4, 8, 16, 32, 64, 128, 256, 512]  # Different overlap values
    nprobe = 120
    top_x_value = 40000
    gt_neighbors = 10000

    # Create figures directory if it doesn't exist
    figures_dir = 'figures'
    os.makedirs(figures_dir, exist_ok=True)

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
        groundtruth = read_ivecs(gt_path)[:, :gt_neighbors]
        centroids = read_fvecs(centroids_path)
        top_clusters = read_ivecs(top_clusters_path)
        cluster_ids = read_ivecs(cluster_ids_path)
        
        num_queries = queries.shape[0]

        for k_overlap in k_overlaps:
            print(f"\nCollecting hit statistics for k_overlap={k_overlap}...")

            hit_counts_A = np.zeros(k_overlap, dtype=np.int64)
            hit_counts_B = np.zeros(k_overlap, dtype=np.int64)
            total_A = 0
            total_B = 0

            for query_idx in tqdm(range(min(1000, num_queries))):
                query = queries[query_idx:query_idx+1]
                distances = np.sum((query - centroids) ** 2, axis=1)
                nearest_clusters = np.argsort(distances)[:nprobe]
                probe_vector_ids = []
                for cluster_id in nearest_clusters:
                    vectors_in_cluster = np.where(cluster_ids.flatten() == cluster_id)[0]
                    probe_vector_ids.extend(vectors_in_cluster)
                query_top_clusters = np.argsort(distances)
                overlap_ratios = []
                for vector_id in probe_vector_ids:
                    vector_top_clusters = top_clusters[vector_id]
                    overlap_ratio = compute_overlap_ratio(query_top_clusters, vector_top_clusters, k_overlap)
                    overlap_ratios.append(overlap_ratio)
                if len(overlap_ratios) < top_x_value:
                    continue
                sorted_indices = np.argsort(overlap_ratios)[::-1]
                selected_ids = [probe_vector_ids[i] for i in sorted_indices[:top_x_value]]
                gt_set = set(groundtruth[query_idx])
                A_ids = [vid for vid in selected_ids if vid in gt_set]
                B_ids = [vid for vid in selected_ids if vid not in gt_set]
                for vid in A_ids:
                    hit_pos = get_hit_positions(query_top_clusters, top_clusters[vid], k_overlap)
                    for pos in hit_pos:
                        hit_counts_A[pos] += 1
                    total_A += 1
                for vid in B_ids:
                    hit_pos = get_hit_positions(query_top_clusters, top_clusters[vid], k_overlap)
                    for pos in hit_pos:
                        hit_counts_B[pos] += 1
                    total_B += 1

            # Normalize to get hit probability per position
            if total_A > 0:
                hit_prob_A = hit_counts_A / total_A
            else:
                hit_prob_A = np.zeros_like(hit_counts_A)
            if total_B > 0:
                hit_prob_B = hit_counts_B / total_B
            else:
                hit_prob_B = np.zeros_like(hit_counts_B)

            # Plot
            plt.figure(figsize=(10, 5))
            plt.bar(np.arange(k_overlap), hit_prob_A, width=0.4, label='A: in GT (10000-NN)')
            plt.bar(np.arange(k_overlap)+0.4, hit_prob_B, width=0.4, label='B: not in GT', alpha=0.7)
            plt.xlabel('Query top-k cluster position')
            plt.ylabel('Hit ratio')
            plt.title(f'Hit map for top {top_x_value} (k_overlap={k_overlap})')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(figures_dir, f'{dataset}_query_gt_hitmap_k{k_overlap}.png'), dpi=600)
            plt.close()

if __name__ == '__main__':
    main()
