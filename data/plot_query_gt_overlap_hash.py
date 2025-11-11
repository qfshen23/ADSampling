import numpy as np
import struct
import os
from tqdm import tqdm
from typing import Dict, List

# =========================
# I/O helpers (keep as-is)
# =========================
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

# =========================
# Recall utilities
# =========================
def compute_recall_by_scores(probe_vector_ids, scores, gt_vector_ids, top_x):
    """Recall when ranking by generic scores (e.g., MinHash similarity)."""
    sorted_indices = np.argsort(scores)[::-1]
    sorted_probe_ids = [probe_vector_ids[i] for i in sorted_indices]
    top_x_vectors = set(sorted_probe_ids[:top_x])
    gt_set = set(map(int, gt_vector_ids))
    overlap_count = len(top_x_vectors.intersection(gt_set))
    recall = overlap_count / max(1, len(gt_vector_ids))
    return recall

def compute_statistics(arr):
    arr = np.array(arr) if not isinstance(arr, np.ndarray) else arr
    if arr.size == 0:
        return {'min':0,'p25':0,'mean':0,'p75':0,'max':0}
    return {
        'min': float(np.min(arr)),
        'p25': float(np.percentile(arr, 25)),
        'mean': float(np.mean(arr)),
        'p75': float(np.percentile(arr, 75)),
        'max': float(np.max(arr)),
    }

# =========================
# MinHash (Jaccard) utils
# =========================
class MinHasher:
    """
    Standard MinHash with num_perm hash functions:
    h_i(x) = (a_i * x + b_i) mod P, signature[i] = min_x h_i(x).
    MinHash 相等率 ~= Jaccard(A,B).
    """
    def __init__(self, num_perm: int = 8, seed: int = 42, prime: int = 2147483647):
        """
        num_perm: #哈希函数；签名比特数 = 32 * num_perm。
                  4->128bit, 8->256bit, 16->512bit
        seed: RNG seed
        prime: 大素数，需 > 最大 cluster_id；默认 2^31-1
        """
        assert num_perm > 0
        self.num_perm = int(num_perm)
        self.prime = int(prime)
        rng = np.random.default_rng(seed)
        self.a = rng.integers(1, self.prime - 1, size=self.num_perm, dtype=np.int64)
        self.b = rng.integers(0, self.prime - 1, size=self.num_perm, dtype=np.int64)

    def sign(self, ids: np.ndarray) -> np.ndarray:
        """Compute MinHash signature (uint32[num_perm]) for a set of int ids."""
        if ids.size == 0:
            return np.full(self.num_perm, np.uint32(self.prime), dtype=np.uint32)
        x = ids.astype(np.int64, copy=False)
        hx = (self.a[:, None] * x[None, :] + self.b[:, None]) % self.prime
        sig = hx.min(axis=1).astype(np.uint32, copy=False)
        return sig

    @staticmethod
    def similarity(sig1: np.ndarray, sig2: np.ndarray) -> float:
        """Fraction of equal components between two signatures (≈ Jaccard)."""
        return float(np.mean(sig1 == sig2))

def est_intersection_from_minhash(sim, k: int):
    """
    sim: float 或 ndarray，≈ Jaccard(A,B)
    等大小集合（|A|=|B|=k）：m = 2kJ/(1+J)
    返回与 sim 同形状的交集估计 m（float/ndarray）
    """
    sim = np.asarray(sim, dtype=np.float64)
    sim = np.clip(sim, 0.0, 1.0)            # 稳健：限制在 [0,1]
    denom = 1.0 + sim
    # 避免除零：当 sim=-1 不会发生；这里还是加个保护
    denom = np.where(denom == 0.0, 1.0, denom)
    return (2.0 * k * sim) / denom

def est_overlap_ratio_from_minhash(sim, k: int):
    """
    sim: float 或 ndarray，≈ Jaccard(A,B)
    等大小集合：overlap_ratio = m/k = 2J/(1+J)
    返回与 sim 同形状的 overlap_ratio（float/ndarray）
    """
    sim = np.asarray(sim, dtype=np.float64)
    sim = np.clip(sim, 0.0, 1.0)
    denom = 1.0 + sim
    denom = np.where(denom == 0.0, 1.0, denom)
    return (2.0 * sim) / denom


# =========================
# Main: MinHash-only DCO selection
# =========================
def main():
    # -------- Parameters --------
    datasets = ['sift']           # 数据集
    K = 1024                      # 簇总数
    k_overlap = 64                # 每个集合大小（top-k clusters）
    nprobe = 20                   # 查询探测的最近簇数量
    top_x_values = [3000]  # 用于评测 recall 的不同 cutoff
    gt_neighbors = 10             # ground-truth 前多少作为正例
    max_queries = 1000            # 最多处理多少个 query

    # MinHash 参数（直接决定签名长度=32*num_perm 比特）
    num_perm = 8                  # 4=128bit, 8=256bit, 16=512bit
    mh_seed = 12345

    # DCO 预算：每个 query 选择多少向量去做 DCO（按 MinHash 分数 top-M）
    # 这不影响评测（评测用 top_x_values），只是给你一个“实际 DCO 列表”的输出示意
    dco_budget = 50000            # 可设为 None/0 表示不额外输出 DCO 列表

    # ----------------------------
    for dataset in datasets:
        print(f"\n=== Processing dataset: {dataset} ===")
        base_path = f'/data/vector_datasets/{dataset}'
        query_path = f'{base_path}/{dataset}_query.fvecs'
        gt_path = f'{base_path}/{dataset}_groundtruth_10000.ivecs'
        centroids_path = f'{base_path}/{dataset}_centroid_{K}.fvecs'
        top_clusters_path = f'{base_path}/{dataset}_top_clusters_1024.ivecs'
        cluster_ids_path = f'{base_path}/{dataset}_cluster_id_{K}.ivecs'

        missing_files = [p for p in [query_path, gt_path, centroids_path, top_clusters_path, cluster_ids_path]
                         if not os.path.exists(p)]
        if missing_files:
            print(f"Skipping {dataset} - {missing_files} missing")
            continue

        print("Loading data...")
        queries = read_fvecs(query_path)
        groundtruth = read_ivecs(gt_path)[:, :gt_neighbors]
        centroids = read_fvecs(centroids_path)
        top_clusters = read_ivecs(top_clusters_path)  # [N, >=k_overlap], each row is cluster IDs sorted by distance
        cluster_ids = read_ivecs(cluster_ids_path)    # [N, 1] cluster assignment for each base vector

        print(f"Queries shape: {queries.shape}")
        print(f"Groundtruth shape: {groundtruth.shape}")
        print(f"Centroids shape: {centroids.shape}")
        print(f"Top clusters shape: {top_clusters.shape}")
        print(f"Cluster IDs shape: {cluster_ids.shape}")

        num_queries = min(max_queries, queries.shape[0])

        # 结果容器（评测使用）
        recall_minhash = {x: [] for x in top_x_values}

        # MinHasher 实例（共享）
        minhasher = MinHasher(num_perm=num_perm, seed=mh_seed)

        # 懒缓存：只对触及到的 base 向量计算签名
        sig_cache: Dict[int, np.ndarray] = {}

        print("\nMinHash-only ranking → choose vectors for DCO...")
        for qidx in tqdm(range(num_queries)):
            qv = queries[qidx:qidx+1]

            # 1) 计算 query 到所有 centroids 的距离，取 nprobe 个最近簇
            distances = np.sum((qv - centroids) ** 2, axis=1)
            nearest_clusters = np.argsort(distances)[:nprobe]

            # 2) 收集这些簇中的 base 向量作为 probe 集合
            probe_vector_ids: List[int] = []
            cluster_lbl = cluster_ids.flatten()
            for cid in nearest_clusters:
                vids = np.where(cluster_lbl == cid)[0]
                if vids.size:
                    probe_vector_ids.extend(vids.tolist())
            if not probe_vector_ids:
                continue

            # 3) 取 query 的 top-k 簇集合并做 MinHash 签名
            #    注意：这里用“按距离全排序”的索引作为 cluster id（与 base 的 top_clusters 空间一致）
            q_order = np.argsort(distances)
            q_set = q_order[:k_overlap].astype(np.int32)
            q_sig = minhasher.sign(q_set)

            # 4) 对所有 probe 向量计算 MinHash 相似度（≈Jaccard），直接作为打分
            mh_scores = np.empty(len(probe_vector_ids), dtype=np.float32)
            for i, vid in enumerate(probe_vector_ids):
                if vid not in sig_cache:
                    vec_top = top_clusters[vid, :k_overlap].astype(np.int32, copy=False)
                    sig_cache[vid] = minhasher.sign(vec_top)
                mh_scores[i] = MinHasher.similarity(q_sig, sig_cache[vid])

            # 5) （可选）输出本 query 的“要做 DCO 的候选列表”（按 MinHash 分数 Top-M）
            if dco_budget and dco_budget > 0:
                m = min(dco_budget, len(probe_vector_ids))
                topm_idx = np.argpartition(mh_scores, -m)[-m:]
                # 真正的 DCO 向量：
                dco_ids = [probe_vector_ids[i] for i in topm_idx]
                # 如果你需要顺序严格从高到低：
                ord_in_m = np.argsort(mh_scores[topm_idx])[::-1]
                dco_ids = [dco_ids[i] for i in ord_in_m]
                # 你可以在这里把 dco_ids 交给你的 DCO 模块

            # 6) 评测：直接用 MinHash 分数排名来估计“被挑中的向量是否覆盖到 GT”
            gt_ids = groundtruth[qidx]
            mh_stats = compute_statistics(mh_scores)
            print(f"Query {qidx} - MinHash score stats: "
                  f"min={mh_stats['min']:.4f}, p25={mh_stats['p25']:.4f}, "
                  f"mean={mh_stats['mean']:.4f}, p75={mh_stats['p75']:.4f}, max={mh_stats['max']:.4f}")

            # 也可把 MinHash 分数映射为“估计 overlap ratio”，看看分布（诊断用）
            est_overlap_ratio = est_overlap_ratio_from_minhash(np.array(mh_scores), k_overlap)
            est_stats = compute_statistics(est_overlap_ratio)
            print(f"Query {qidx} - Estimated overlap ratio (from MinHash) stats: "
                  f"min={est_stats['min']:.4f}, p25={est_stats['p25']:.4f}, "
                  f"mean={est_stats['mean']:.4f}, p75={est_stats['p75']:.4f}, max={est_stats['max']:.4f}")

            for top_x in top_x_values:
                if top_x <= len(probe_vector_ids):
                    r = compute_recall_by_scores(probe_vector_ids, mh_scores, gt_ids, top_x)
                    recall_minhash[top_x].append(r)
                else:
                    print(f"  [Warn] top-{top_x} > #probes ({len(probe_vector_ids)})")

        # -------- 汇总 --------
        print(f"\n=== Average recall (MinHash-only ranking) for {dataset} ===")
        for x in top_x_values:
            vals = recall_minhash[x]
            avg = float(np.mean(vals)) if len(vals) else 0.0
            print(f"  top-{x}: {avg:.4f}  (num_perm={num_perm}, signature_bits={32*num_perm})")

if __name__ == '__main__':
    main()
