import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import struct
import os

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
    gt_neighbors = 10000

    for dataset in datasets:
        print(f"\n=== Processing dataset: {dataset} ===")
        base_path = f'/data/vector_datasets/{dataset}'
        base_vectors_path = f'{base_path}/{dataset}_base.fvecs'
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
        base_vectors = read_fvecs(base_vectors_path)
        queries = read_fvecs(query_path)
        groundtruth = read_ivecs(gt_path)[:, :gt_neighbors]
        centroids = read_fvecs(centroids_path)
        top_clusters = read_ivecs(top_clusters_path)
        cluster_ids = read_ivecs(cluster_ids_path)


        query_idx = 2
        query = queries[query_idx:query_idx+1]
        distances = np.sum((query - centroids) ** 2, axis=1)
        nearest_clusters = np.argsort(distances)[:nprobe]

        # 找到 nprobe 内所有 vector，并记录其距离和 overlap ratio
        base_vector_ids = []
        base_vector_distances = []
        base_vector_overlap_ratios = []

        query_top_clusters = np.argsort(distances)

        for cluster_id in nearest_clusters:
            vectors_in_cluster = np.where(cluster_ids.flatten() == cluster_id)[0]
            for vector_id in vectors_in_cluster:
                # 计算该base vector到query的距离
                base_vector_ids.append(vector_id)
                # 若有原始base数据，可替换为 base_vectors[vector_id]
                base_dist = np.sum((query.flatten() - base_vectors[vector_id]) ** 2)
                base_vector_distances.append(base_dist)
                # 计算overlap ratio
                vector_top_clusters = top_clusters[vector_id]
                overlap_ratio = compute_overlap_ratio(query_top_clusters, vector_top_clusters, k_overlap)
                base_vector_overlap_ratios.append(overlap_ratio)

        # 按距离升序排序
        base_vector_distances = np.array(base_vector_distances)
        base_vector_overlap_ratios = np.array(base_vector_overlap_ratios)
        sort_idx = np.argsort(base_vector_distances)
        overlap_ratios_sorted = base_vector_overlap_ratios[sort_idx]

        # 创建bucket
        bucket_size = 1000
        num_buckets = len(overlap_ratios_sorted) // bucket_size
        bucket_data = []
        
        for i in range(num_buckets):
            start_idx = i * bucket_size
            end_idx = start_idx + bucket_size
            bucket_data.append(overlap_ratios_sorted[start_idx:end_idx])

        # 可视化
        plt.figure(figsize=(12, 4))
        plt.boxplot(bucket_data, positions=range(len(bucket_data)))
        plt.xlabel(f'Bucket index (sorted by distance, bucket size={bucket_size})')
        plt.ylabel(f'Overlap ratio (top-{k_overlap})')
        plt.title(f'Box Plot of Overlap Ratio (bucket size={bucket_size}), Query {query_idx}')
        
        # Set x-axis ticks at larger intervals
        plt.xticks(range(0, len(bucket_data), 5), range(0, len(bucket_data), 5))  # Show every 5th tick with actual vector indices
        
        plt.tight_layout()
        os.makedirs(f'windows_{dataset}', exist_ok=True)
        plt.savefig(f'windows_{dataset}/{dataset}_query_{query_idx}_overlap_ratio_boxplot.png', dpi=600)
        plt.close()

if __name__ == '__main__':
    main()
