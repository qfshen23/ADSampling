import numpy as np
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

def overlap_ratio(list1, list2):
    """ 计算两个列表的 overlap ratio """
    return len(set(list1) & set(list2)) / len(list1)

def simulate_ivf(
    dataset='sift',
    nprobe=128,
    k=8,          # 每组cluster数
    b=500,         # block大小
    dco_budget=40000,
    topK=10000,
):
    # File paths
    base_path = f'/data/vector_datasets/{dataset}'
    base_fvecs_path = f'{base_path}/{dataset}_base.fvecs'
    query_fvecs_path = f'{base_path}/{dataset}_query.fvecs'
    gt_ivecs_path = f'{base_path}/{dataset}_groundtruth_10000.ivecs'
    K = 1024  # Number of clusters
    centroids_fvecs_path = f'{base_path}/{dataset}_centroid_{K}.fvecs'
    base_cluster_rank_path = f'{base_path}/{dataset}_top_clusters_{K}.ivecs'
    base_assign_path = f'{base_path}/{dataset}_cluster_id_{K}.ivecs'

    # 读取数据
    base = read_fvecs(base_fvecs_path)
    query = read_fvecs(query_fvecs_path)
    centroids = read_fvecs(centroids_fvecs_path)
    gt = read_ivecs(gt_ivecs_path)
    base_cluster_rank = read_ivecs(base_cluster_rank_path)  # [nb, nlist]
    base_assign = read_ivecs(base_assign_path)  # [nb]
    assert base.shape[0] == base_cluster_rank.shape[0] == base_assign.shape[0]

    nq = query.shape[0]
    nlist = centroids.shape[0]
    assert nprobe % k == 0

    recalls = []
    
    for qid in tqdm(range(nq)):
        q = query[qid]
        # 1. 计算 query 到每个 cluster 的距离
        dists_to_centroids = np.linalg.norm(centroids - q, axis=1)  # [nlist]
        cluster_rank = np.argsort(dists_to_centroids)  # [nlist]
        probe_clusters = cluster_rank[:nprobe]  # [nprobe]
        # 2. 分组，每组k个cluster
        groups = [probe_clusters[i:i+k] for i in range(0, nprobe, k)]

        all_blocks = []  # (block_dco, [base_ids in block])
        for group_idx, clusters_in_group in enumerate(groups):
            # 获取 group 覆盖的所有 base vector 下标
            in_group_mask = np.isin(base_assign, clusters_in_group)
            base_ids_in_group = np.where(in_group_mask)[0]
            if len(base_ids_in_group) == 0:
                continue

            # 3. 对每个 base 计算 overlap ratio
            # k_for_overlap = (group_idx + 1) * k
            k_for_overlap = 64
            q_topk_clusters = cluster_rank[:k_for_overlap]
            group_with_overlap = []
            for base_id in base_ids_in_group:
                base_topk_clusters = base_cluster_rank[base_id, :k_for_overlap]
                overlap = overlap_ratio(q_topk_clusters, base_topk_clusters)
                group_with_overlap.append((overlap, base_id))
            # 按 overlap ratio 降序
            group_with_overlap.sort(reverse=True)

            # 4. 按 b 分块
            for i in range(0, len(group_with_overlap), b):
                block = group_with_overlap[i:i+b]
                block_base_ids = [item[1] for item in block]
                # block 第一个 vector 用于算 block dco
                dco = np.linalg.norm(base[block_base_ids[0]] - q)
                all_blocks.append((dco, block_base_ids))

        # 5. 按 dco 升序拼块，直到总 vector ≥ dco_budget
        all_blocks.sort()
        selected = []
        for dco, block_base_ids in all_blocks:
            selected.extend(block_base_ids)
            if len(selected) >= dco_budget:
                selected = selected[:dco_budget]  # 精确截断到 budget
                break

        selected = set(selected)
        hit = len(selected & set(gt[qid, :topK]))
        recalls.append(hit / topK)

        # Output mean recall every 100 queries
        if (qid + 1) % 100 == 0:
            current_mean = np.mean(recalls)
            print(f"Query {qid + 1}/{nq}, Current Mean Recall@{topK}: {current_mean:.4f}")

    mean_recall = np.mean(recalls)
    print(f"\nFinal Recall@{topK}: {mean_recall:.4f}")

simulate_ivf('sift', nprobe=128, k=64, b=50, dco_budget=40000, topK=10000)