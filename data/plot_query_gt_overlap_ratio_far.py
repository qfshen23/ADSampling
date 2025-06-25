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
        return np.zeros((0, 0), dtype=np.int32)
    dim = fv[0]
    assert dim > 0
    fv = fv.reshape(-1, 1 + dim)
    if not all(fv[:, 0] == dim):
        raise IOError("Non-uniform vector sizes in " + filename)
    return fv[:, 1:]

def compute_overlap_ratio(clusters1, clusters2, k):
    set1 = set(clusters1[:k])
    set2 = set(clusters2[:k])
    return len(set1.intersection(set2)) / k

def main():
    dataset = 'sift'
    K = 1024
    topk_values = [4, 8, 16, 32, 64, 128]
    num_gt = 10000
    max_query = 10000   # 只取前1000个 query
    base_path = f'/data/vector_datasets/{dataset}'
    query_path = f'{base_path}/{dataset}_query.fvecs'
    gt_path = f'{base_path}/{dataset}_groundtruth_10000.ivecs'
    centroids_path = f'{base_path}/{dataset}_centroid_{K}.fvecs'
    top_clusters_path = f'{base_path}/{dataset}_top_clusters_{K}.ivecs'

    queries = read_fvecs(query_path)
    groundtruth = read_ivecs(gt_path)[:, :num_gt]
    centroids = read_fvecs(centroids_path)
    top_clusters = read_ivecs(top_clusters_path)

    num_queries = min(max_query, queries.shape[0])
    overlap_curves = {k: np.zeros(num_gt) for k in topk_values}        # 最近的top-k overlap
    overlap_curves_far = {k: np.zeros(num_gt) for k in topk_values}   # 最远的top-k overlap

    for query_idx in tqdm(range(num_queries)):
        query = queries[query_idx:query_idx + 1]
        dists = np.sum((query - centroids) ** 2, axis=1)
        query_top_clusters = np.argsort(dists)
        query_far_clusters = query_top_clusters[::-1]  # 逆序，最远在前

        gt_ids = groundtruth[query_idx]
        for rank, gt_id in enumerate(gt_ids):
            if gt_id >= len(top_clusters):
                continue
            gt_top_clusters = top_clusters[gt_id]
            gt_far_clusters = gt_top_clusters[::-1]
            for k in topk_values:
                # 最近top-k
                overlap = compute_overlap_ratio(query_top_clusters, gt_top_clusters, k)
                overlap_curves[k][rank] += overlap
                # 最远top-k
                overlap_far = compute_overlap_ratio(query_far_clusters, gt_far_clusters, k)
                overlap_curves_far[k][rank] += overlap_far

    # Normalize
    for k in topk_values:
        overlap_curves[k] /= num_queries
        overlap_curves_far[k] /= num_queries

    x = np.arange(num_gt)
    plt.figure(figsize=(12, 8))
    for k in topk_values:
        plt.plot(x, overlap_curves[k], label=f'Nearest Top-{k}')
    plt.xlabel("Groundtruth Vector Rank")
    plt.ylabel("Average Overlap Ratio")
    plt.title("Average GT Cluster Overlap vs. Query Nearest Top-K Clusters")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{dataset}_query_gt_overlap_ratio_nearest.png', dpi=600)    
    plt.close()

    plt.figure(figsize=(12, 8))
    for k in topk_values:
        plt.plot(x, overlap_curves_far[k], label=f'Farthest Top-{k}')
    plt.xlabel("Groundtruth Vector Rank")
    plt.ylabel("Average Overlap Ratio")
    plt.title("Average GT Cluster Overlap vs. Query Farthest Top-K Clusters")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{dataset}_query_gt_overlap_ratio_far.png', dpi=600)
    plt.close()

if __name__ == '__main__':
    main()
