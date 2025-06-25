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

def main():
    datasets = ['sift']
    K = 1024
    k_overlap = 64
    nprobe = 120
    top_x_values = [20000, 30000, 40000]
    gt_neighbors = 10000

    bucket_size = 10
    num_buckets = nprobe // bucket_size

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

        # 初始化：所有query的累计bucket统计
        all_query_bucket_counts_dict = {top_x: np.zeros(num_buckets, dtype=int) for top_x in top_x_values}

        for query_idx in tqdm(range(min(1000, num_queries))):  # 可自行调整query数
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

            gt_vector_ids = groundtruth[query_idx]
            gt_set = set(gt_vector_ids)
            cluster_to_rank = {c: r for r, c in enumerate(np.argsort(distances))}

            for top_x in top_x_values:
                if overlap_ratios and len(probe_vector_ids) > 0 and top_x <= len(probe_vector_ids):
                    sorted_indices = np.argsort(overlap_ratios)[::-1]
                    sorted_probe_ids = [probe_vector_ids[i] for i in sorted_indices]
                    top_x_ids = sorted_probe_ids[:top_x]

                    non_gt_probe_ids = [vid for vid in top_x_ids if vid not in gt_set]

                    cluster_bucket_counts = np.zeros(num_buckets, dtype=int)
                    for vid in non_gt_probe_ids:
                        cluster_id = cluster_ids[vid][0]
                        if cluster_id in cluster_to_rank:
                            rank = cluster_to_rank[cluster_id]
                            if rank < nprobe:
                                bucket_idx = rank // bucket_size
                                cluster_bucket_counts[bucket_idx] += 1
                    # 累加到总统计
                    all_query_bucket_counts_dict[top_x] += cluster_bucket_counts

        # 所有 query 处理完，绘图
        bucket_labels = [f"{i*bucket_size+1}-{(i+1)*bucket_size}" for i in range(num_buckets)]
        for top_x in top_x_values:
            plt.figure(figsize=(8, 4))
            plt.bar(bucket_labels, all_query_bucket_counts_dict[top_x])
            plt.title(f'All Queries, Top-{top_x}, high-overlap non-GT probe vectors')
            plt.xlabel('Query top-cluster rank bucket')
            plt.ylabel('Total Count')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f'{dataset}_all_queries_top_{top_x}_distribution.png', dpi=600)
            plt.close()
if __name__ == '__main__':
    main()
