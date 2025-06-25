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
    k_overlap_near = 64
    k_overlap_far = 16
    nprobe = 120
    top_x_value = 80000  # 从 far-overlap 最小中保留这么多
    top_y_values = [20000, 30000, 40000, 50000, 60000]  # 从 top_x_value 中选出这些用于 recall
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
        recall_results = {y: [] for y in top_y_values}

        print("\nFiltering by far-overlap, then ranking by near-overlap...")
        for query_idx in tqdm(range(min(1000, num_queries))):
            query = queries[query_idx:query_idx+1]
            distances = np.sum((query - centroids) ** 2, axis=1)
            nearest_clusters = np.argsort(distances)[:nprobe]

            # Collect probe vectors
            probe_vector_ids = []
            for cluster_id in nearest_clusters:
                probe_vector_ids.extend(np.where(cluster_ids.flatten() == cluster_id)[0])

            query_top_clusters = np.argsort(distances)
            query_far_clusters = query_top_clusters[-k_overlap_far:]

            # Compute far-overlap
            far_overlap_ratios = []
            for vid in probe_vector_ids:
                vector_top_clusters = top_clusters[vid]
                ratio = compute_overlap_ratio(query_far_clusters, vector_top_clusters, k_overlap_far)
                far_overlap_ratios.append(ratio)

            # Stage 1: Select vectors with smallest far-overlap
            if top_x_value > len(probe_vector_ids):
                print(f"Query {query_idx}: not enough vectors for top_x={top_x_value}")
                continue

            indices_stage1 = np.argsort(far_overlap_ratios)[:top_x_value]
            stage1_ids = [probe_vector_ids[i] for i in indices_stage1]

            # Stage 2: On top-x set, compute near-overlap
            near_overlap_ratios = []
            for vid in stage1_ids:
                vector_top_clusters = top_clusters[vid]
                ratio = compute_overlap_ratio(query_top_clusters, vector_top_clusters, k_overlap_near)
                near_overlap_ratios.append(ratio)

            # Groundtruth
            gt_vector_ids = set(groundtruth[query_idx])

            for top_y in top_y_values:
                if top_y > len(stage1_ids):
                    print(f"  Top-{top_y}: not enough stage1 vectors ({len(stage1_ids)})")
                    continue

                indices_stage2 = np.argsort(near_overlap_ratios)[::-1][:top_y]
                selected_ids = [stage1_ids[i] for i in indices_stage2]

                recall = len(set(selected_ids) & gt_vector_ids) / len(gt_vector_ids)
                recall_results[top_y].append(recall)

                # Optional: print stats
                selected_ratios = [near_overlap_ratios[i] for i in indices_stage2]
                stats = compute_statistics(selected_ratios)
                print(f"Query {query_idx} Top-{top_y}: recall={recall:.4f}, mean_overlap={stats['mean']:.4f}")

        print(f"\n=== Average recall results for dataset: {dataset} ===")
        for y in top_y_values:
            if recall_results[y]:
                print(f"  Top-{y}: Avg Recall = {np.mean(recall_results[y]):.4f}")

if __name__ == '__main__':
    main()
