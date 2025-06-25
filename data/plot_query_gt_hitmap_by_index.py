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

def main():
    datasets = ['sift']  # List of datasets to process
    K = 1024  # Number of clusters
    top_k = 16    # 你想分析的 top-k
    nprobe = 120  # Number of nearest clusters to probe
    num_queries = 1000  # 只分析前 1K query

    for dataset in datasets:
        print(f"\n=== Processing dataset: {dataset} ===")
        
        base_path = f'/data/vector_datasets/{dataset}'
        query_path = f'{base_path}/{dataset}_query.fvecs'
        centroids_path = f'{base_path}/{dataset}_centroid_{K}.fvecs'
        top_clusters_path = f'{base_path}/{dataset}_top_clusters_{K}.ivecs'
        cluster_ids_path = f'{base_path}/{dataset}_cluster_id_{K}.ivecs'
        
        # Check file existence
        missing_files = [p for p in [query_path, centroids_path, top_clusters_path, cluster_ids_path] if not os.path.exists(p)]
        if missing_files:
            print(f"Skipping {dataset} - {missing_files} missing")
            continue
        
        print("Loading data...")
        queries = read_fvecs(query_path)
        centroids = read_fvecs(centroids_path)
        top_clusters = read_ivecs(top_clusters_path)  # Each row: sorted cluster IDs by distance (base vectors)
        cluster_ids = read_ivecs(cluster_ids_path)    # Each row: assigned cluster ID for each base vector
        
        print(f"Queries shape: {queries.shape}")
        print(f"Centroids shape: {centroids.shape}")
        print(f"Top clusters shape: {top_clusters.shape}")
        print(f"Cluster IDs shape: {cluster_ids.shape}")
        
        query_num = min(num_queries, queries.shape[0])
        base_num = cluster_ids.shape[0]

        # 统计 hitmap
        hitmap = np.zeros(top_k, dtype=int)

        print("\nComputing cluster rank hitmap...")
        for query_idx in tqdm(range(query_num)):  
            query = queries[query_idx:query_idx+1]

            # 1. 查询到所有 centroids 的距离
            distances = np.sum((query - centroids) ** 2, axis=1)
            nearest_clusters = np.argsort(distances)[:nprobe]   # 前 nprobe 个 cluster

            # 2. Query 的 top-k cluster 列表（实际你可以取全量，这里只看前top_k）
            query_topk_clusters = np.argsort(distances)[:top_k]    # (top_k, )

            # 3. 找出这些 cluster 里所有的 base vectors
            probe_vector_ids = []
            for cluster_id in nearest_clusters:
                vectors_in_cluster = np.where(cluster_ids.flatten() == cluster_id)[0]
                probe_vector_ids.extend(vectors_in_cluster)
            
            # 4. 对每个 base vector 统计
            for bvid in probe_vector_ids:
                bv_topk_clusters = top_clusters[bvid][:top_k]  # (top_k, )

                # 遍历 query 的每个 top-k cluster
                for i in range(top_k):
                    if query_topk_clusters[i] in bv_topk_clusters:
                        hitmap[i] += 1
                        # 这里如果你希望一个 base vector 命中多个 i，可以不 break
                        # 如果你只想每个base vector命中第一个 i，就 break
                        # break
        print("Hitmap:", hitmap)
        
        # 归一化（可选：每个 query 归一化，每个 i 再平均，或直接总和）
        plt.figure(figsize=(10,4))
        plt.bar(np.arange(1, top_k+1), hitmap)
        plt.xlabel("Query Top-k Cluster Rank")
        plt.ylabel("Hit Count (over all nprobe base vectors)")
        plt.title(f"Hitmap of Base Vector's Top-{top_k} Cluster Covering Query's Cluster Rank")
        plt.savefig(f'{dataset}_query_gt_hitmap_by_index_k{top_k}.png', dpi=600)
        plt.close()

if __name__ == '__main__':
    main()
