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
    intersection = len(set1.intersection(set2))
    return intersection / k

def compute_recall(candidate_ids, gt_vector_ids):
    candidate_set = set(candidate_ids)
    gt_set = set(gt_vector_ids)
    hit_count = len(candidate_set & gt_set)
    recall = hit_count / len(gt_set)
    return recall

def main():
    datasets = ['sift']
    K = 1024
    k_overlap = 64
    nprobe = 120
    nprobe_strict = 20  # 前10个cluster全部保留
    top_x_value = 40000
    gt_neighbors = 10000

    for dataset in datasets:
        print(f"\n=== Processing dataset: {dataset} top_x_value={top_x_value} ===")
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
        recall_results = []

        print("\nRunning hybrid overlap+dco strategy...")
        for query_idx in tqdm(range(min(1000, num_queries))):
            query = queries[query_idx:query_idx+1]
            distances = np.sum((query - centroids) ** 2, axis=1)
            nearest_clusters = np.argsort(distances)[:nprobe]
            # 集合A：前16个cluster所有vector
            strict_clusters = nearest_clusters[:nprobe_strict]
            setA_vector_ids = []
            for cluster_id in strict_clusters:
                vectors_in_cluster = np.where(cluster_ids.flatten() == cluster_id)[0]
                setA_vector_ids.extend(vectors_in_cluster)
            a = len(setA_vector_ids)
            # 集合B候选范围
            relaxed_clusters = nearest_clusters[nprobe_strict:]
            relaxed_vector_ids = []
            for cluster_id in relaxed_clusters:
                vectors_in_cluster = np.where(cluster_ids.flatten() == cluster_id)[0]
                relaxed_vector_ids.extend(vectors_in_cluster)
            # 针对集合B，计算overlap ratio，并选top (top_x_value - a)
            query_top_clusters = np.argsort(distances)
            relaxed_overlap_ratios = []
            for vector_id in relaxed_vector_ids:
                vector_top_clusters = top_clusters[vector_id]
                overlap_ratio = compute_overlap_ratio(query_top_clusters, vector_top_clusters, k_overlap)
                relaxed_overlap_ratios.append(overlap_ratio)
            if a >= top_x_value:
                candidate_ids = setA_vector_ids[:top_x_value]
            else:
                # 按overlap ratio排序
                sorted_indices = np.argsort(relaxed_overlap_ratios)[::-1]
                n_b = top_x_value - a
                selected_B = [relaxed_vector_ids[i] for i in sorted_indices[:n_b]]
                candidate_ids = setA_vector_ids + selected_B
            # 计算recall
            gt_vector_ids = groundtruth[query_idx]
            recall = compute_recall(candidate_ids, gt_vector_ids)
            recall_results.append(recall)
            print(f"Query {query_idx} - |A|={a}, |B|={top_x_value - a}, recall={recall:.4f}")

        print(f"\nAverage recall over {len(recall_results)} queries for {dataset}, top_x_value={top_x_value}: {np.mean(recall_results):.4f}")

if __name__ == '__main__':
    main()
