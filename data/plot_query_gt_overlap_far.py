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
    set1 = set(clusters1[:k])
    set2 = set(clusters2[:k])
    return len(set1 & set2) / k

def compute_statistics(overlap_ratios):
    ratios = np.array(overlap_ratios)
    return {
        'min': np.min(ratios),
        'p25': np.percentile(ratios, 25),
        'mean': np.mean(ratios),
        'p75': np.percentile(ratios, 75),
        'max': np.max(ratios)
    }

def main():
    datasets = ['sift']
    K = 1024
    k_overlap_far = 8
    nprobe = 120
    top_x_values = [110000]
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
        groundtruth = read_ivecs(gt_path)[:, :gt_neighbors]
        centroids = read_fvecs(centroids_path)
        top_clusters = read_ivecs(top_clusters_path)
        cluster_ids = read_ivecs(cluster_ids_path)

        num_queries = queries.shape[0]
        recall_results = {x: [] for x in top_x_values}

        print("\nPruning with far-overlap ratio only (stage one)...")
        for query_idx in tqdm(range(min(1000, num_queries))):
            query = queries[query_idx:query_idx+1]
            distances = np.sum((query - centroids) ** 2, axis=1)
            nearest_clusters = np.argsort(distances)[:nprobe]

            probe_vector_ids = []
            for cluster_id in nearest_clusters:
                probe_vector_ids.extend(np.where(cluster_ids.flatten() == cluster_id)[0])

            query_far_clusters = np.argsort(distances)[-k_overlap_far:]

            # Compute far overlap for all probe vectors
            far_overlap_ratios = []
            for vid in probe_vector_ids:
                vector_top_clusters = top_clusters[vid]
                ratio = compute_overlap_ratio(query_far_clusters, vector_top_clusters, k_overlap_far)
                far_overlap_ratios.append(ratio)

            gt_vector_ids = set(groundtruth[query_idx])

            for top_x in top_x_values:
                if top_x > len(probe_vector_ids):
                    print(f"Query {query_idx}: Not enough probe vectors for top-{top_x}")
                    continue

                selected_indices = np.argsort(far_overlap_ratios)[:top_x]
                selected_ids = [probe_vector_ids[i] for i in selected_indices]
                hit = len(set(selected_ids) & gt_vector_ids)
                recall = hit / len(gt_vector_ids)
                recall_results[top_x].append(recall)

                stats = compute_statistics([far_overlap_ratios[i] for i in selected_indices])
                print(f"Query {query_idx} Top-{top_x}: recall={recall:.4f}, far_overlap mean={stats['mean']:.4f}")

        print(f"\n=== Average Recall with Far-Overlap Pruning Only ===")
        for x in top_x_values:
            if recall_results[x]:
                print(f"Top-{x}: Avg Recall = {np.mean(recall_results[x]):.4f}")

if __name__ == '__main__':
    main()
