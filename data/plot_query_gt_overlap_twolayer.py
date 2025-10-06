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
        return np.zeros((0, 0), dtype=np.int32)
    dim = fv[0]
    assert dim > 0
    fv = fv.reshape(-1, 1 + dim)
    if not all(fv[:, 0] == dim):
        raise IOError("Non-uniform vector sizes in " + filename)
    return fv[:, 1:]

def compute_overlap_ratio(query_top_clusters, vector_top_clusters, query_k, base_k=64):
    set1 = set(query_top_clusters[:query_k])
    set2 = set(vector_top_clusters[:base_k])
    intersection = len(set1.intersection(set2))
    return intersection / query_k

def compute_recall(selected_vector_ids, gt_vector_ids):
    hit = len(set(selected_vector_ids).intersection(set(gt_vector_ids)))
    return hit / len(gt_vector_ids)

def main():
    # Parameters
    datasets = ['gist']
    K = 1024
    nprobe = 120
    gt_nn = 10  # 设置要使用的gt邻居数量
    
    # 配置：每层(query_topk, keep_topn)
    layer_configs = [
        (128, 80000),
        (64, 20000),
    ]

    for dataset in datasets:
        print(f"\n=== Processing dataset: {dataset} ===")

        base_path = f'/data/vector_datasets/{dataset}'
        query_path = f'{base_path}/{dataset}_query.fvecs'
        gt_path = f'{base_path}/{dataset}_groundtruth_10000.ivecs'
        centroids_path = f'{base_path}/{dataset}_centroid_{K}.fvecs'
        top_clusters_path = f'{base_path}/{dataset}_top_clusters_{K}.ivecs'
        cluster_ids_path = f'{base_path}/{dataset}_cluster_id_{K}.ivecs'

        if not all(os.path.exists(p) for p in [query_path, gt_path, centroids_path, top_clusters_path, cluster_ids_path]):
            print(f"Skipping {dataset} - some files missing")
            continue

        print("Loading data...")
        queries = read_fvecs(query_path)
        groundtruth = read_ivecs(gt_path)[:, :gt_nn]  # 只取前gt_nn个邻居
        centroids = read_fvecs(centroids_path)
        top_clusters = read_ivecs(top_clusters_path)
        cluster_ids = read_ivecs(cluster_ids_path)

        print(f"Queries shape: {queries.shape}")
        print(f"Groundtruth shape: {groundtruth.shape}")
        print(f"Centroids shape: {centroids.shape}")
        print(f"Top clusters shape: {top_clusters.shape}")
        print(f"Cluster IDs shape: {cluster_ids.shape}")

        num_queries = queries.shape[0]
        recall_results = []

        print("\nComputing multi-layer overlap and recall...")
        for query_idx in tqdm(range(min(100, num_queries))):
            query = queries[query_idx:query_idx+1]
            distances = np.sum((query - centroids) ** 2, axis=1)
            query_top_clusters = np.argsort(distances)  # 排序得到所有cluster的rank

            # 选出nprobe个最近的cluster
            nearest_clusters = np.argsort(distances)[:nprobe]

            # 聚合probe中所有base vector的id
            probe_vector_ids = []
            for cluster_id in nearest_clusters:
                probe_vector_ids.extend(np.where(cluster_ids.flatten() == cluster_id)[0])

            current_vector_ids = probe_vector_ids

            # 多层过滤
            for layer_idx, (query_topk, keep_topn) in enumerate(layer_configs):
                if len(current_vector_ids) == 0:
                    print(f"  Query {query_idx} - Layer {layer_idx} - No candidates left")
                    break
                overlap_ratios = []
                for vector_id in current_vector_ids:
                    vector_top_clusters = top_clusters[vector_id]
                    ratio = compute_overlap_ratio(query_top_clusters, vector_top_clusters, query_topk, base_k=64)
                    overlap_ratios.append(ratio)

                # 保留overlap最大的top-n
                if len(overlap_ratios) <= keep_topn:
                    top_indices = np.arange(len(overlap_ratios))
                else:
                    top_indices = np.argsort(overlap_ratios)[::-1][:keep_topn]
                current_vector_ids = [current_vector_ids[i] for i in top_indices]

                # 如果到最后一层，算recall
                if layer_idx == len(layer_configs) - 1:
                    gt_vector_ids = groundtruth[query_idx]
                    recall = compute_recall(current_vector_ids, gt_vector_ids)
                    recall_results.append(recall)
                    print(f"Query {query_idx} - Final Recall@{keep_topn}: {recall:.4f}")

        # 汇总
        print("\n=== Final Results ===")
        if recall_results:
            avg_recall = np.mean(recall_results)
            print(f"Average Recall@{layer_configs[-1][1]} over {len(recall_results)} queries: {avg_recall:.4f}")

if __name__ == '__main__':
    main()
