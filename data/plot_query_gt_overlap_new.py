import numpy as np
import os
from tqdm import tqdm
from typing import Dict, List, Tuple

# =========================
# I/O helpers（与你原版一致）
# =========================
def read_fvecs(filename, c_contiguous=True):
    fv = np.fromfile(filename, dtype=np.float32)
    if fv.size == 0:
        return np.zeros((0, 0), dtype=np.float32)
    dim = fv.view(np.int32)[0]
    assert dim > 0
    fv = fv.reshape(-1, 1 + dim)
    if not np.all(fv.view(np.int32)[:, 0] == dim):
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
    if not np.all(fv[:, 0] == dim):
        raise IOError("Non-uniform vector sizes in " + filename)
    return fv[:, 1:]

# =========================
# POPCNT（Python int.bit_count 很快）
# =========================
def popcnt_u64(w: np.uint64) -> int:
    return int(w).bit_count()

# =========================
# 评测工具
# =========================
def compute_recall_by_scores(probe_vector_ids, scores, gt_vector_ids, top_x):
    """按打分（降序）选前 top_x，统计与 GT 的交集比例。"""
    if len(probe_vector_ids) == 0:
        return 0.0
    scores = np.asarray(scores)
    m = min(top_x, len(probe_vector_ids))
    idx = np.argpartition(scores, -m)[-m:]
    order = np.argsort(scores[idx])[::-1]
    chosen = {probe_vector_ids[i] for i in idx[order]}
    gt_set = set(int(x) for x in gt_vector_ids)
    inter = len(chosen.intersection(gt_set))
    return inter / max(1, len(gt_vector_ids))

# =========================
# 简单双精度 k-means（不依赖 sklearn）
# =========================
def kmeans_np(X: np.ndarray, k: int, iters: int = 25, seed: int = 42) -> np.ndarray:
    """
    X: [N, D]
    返回: assignments [N] ∈ {0..k-1}
    """
    rng = np.random.default_rng(seed)
    N, D = X.shape
    # k-means++ 初始化（简化版）
    centroids = np.empty((k, D), dtype=X.dtype)
    idx0 = rng.integers(0, N)
    centroids[0] = X[idx0]
    dist2 = np.full(N, np.inf, dtype=X.dtype)
    for i in range(1, k):
        d2 = np.sum((X - centroids[i-1])**2, axis=1)
        dist2 = np.minimum(dist2, d2)
        probs = dist2 / (dist2.sum() + 1e-12)
        centroids[i] = X[rng.choice(N, p=probs)]
    # 迭代
    for _ in range(iters):
        # assign
        d2 = np.sum(X**2, axis=1, keepdims=True) + np.sum(centroids**2, axis=1)[None, :] - 2 * (X @ centroids.T)
        labels = np.argmin(d2, axis=1)
        # update
        for j in range(k):
            idx = (labels == j)
            if np.any(idx):
                centroids[j] = X[idx].mean(axis=0)
            else:
                # 空簇重采样
                centroids[j] = X[rng.integers(0, N)]
    return labels

# =========================
# HMB：层级元簇掩码
# - 将 C 个簇聚成 M 个“元簇”
# - 签名：集合中任一簇落入的元簇位置置 1（64-bit）
# - 打分：AND + popcnt
# =========================
class HMBSignature:
    def __init__(self, M_bits: int = 64, mode: str = "cooccur", seed: int = 1234,
                 cooccur_feat_dim: int = 64, cooccur_sample_size: int = 20000):
        """
        mode: "cooccur" | "centroid"
          - cooccur: 从 top_clusters 抽样构建簇共现嵌入（推荐）
          - centroid: 直接用簇中心坐标做 kmeans 成 64 类
        """
        assert M_bits == 64, "目前实现固定 64 位（如需 128 可以拓展）"
        self.M = int(M_bits)
        self.mode = mode
        self.seed = int(seed)
        self.cooccur_feat_dim = int(cooccur_feat_dim)
        self.cooccur_sample_size = int(cooccur_sample_size)
        self.cluster2meta = None  # [C] → {0..63}

    @staticmethod
    def _hash_bucket(ids: np.ndarray, F: int, seed: int) -> np.ndarray:
        """将 cluster id 映射到 0..F-1 的桶（用于低维共现特征）。"""
        x = ids.astype(np.uint64)
        z = (x + np.uint64(seed)) & np.uint64(0xFFFFFFFFFFFFFFFF)
        z = (z + np.uint64(0x9E3779B97F4A7C15)) & np.uint64(0xFFFFFFFFFFFFFFFF)
        z = (z ^ (z >> 30)) * np.uint64(0xBF58476D1CE4E5B9) & np.uint64(0xFFFFFFFFFFFFFFFF)
        z = (z ^ (z >> 27)) * np.uint64(0x94D049BB133111EB) & np.uint64(0xFFFFFFFFFFFFFFFF)
        z = z ^ (z >> 31)
        return (z % np.uint64(F)).astype(np.int64)

    def _build_cooccur_embeddings(self, top_clusters: np.ndarray, C: int, k_overlap: int) -> np.ndarray:
        """
        用抽样的 base 向量 top-m 列表构建每个簇的低维共现特征 E ∈ R^{C×F}。
        方法：对每条样本 T，先把 T 的簇 id 映射到 F 个桶的直方图 h，
             然后对 T 中每个簇 c：E[c] += h 。
        这样每条样本的复杂度 O(m)，总体 O(sample_size * m)。
        """
        N = top_clusters.shape[0]
        sample = min(self.cooccur_sample_size, N)
        rng = np.random.default_rng(self.seed)
        idxs = rng.choice(N, size=sample, replace=False)

        F = self.cooccur_feat_dim
        E = np.zeros((C, F), dtype=np.float32)
        seed_b = int(rng.integers(1, (1<<63)-1))
        for i in tqdm(range(sample), desc=f"HMB | build co-occur features (F={F})"):
            T = top_clusters[idxs[i], :k_overlap]
            buckets = self._hash_bucket(T, F, seed_b)               # [m]
            h = np.bincount(buckets, minlength=F).astype(np.float32) # [F]
            for c in T:
                E[int(c)] += h
        # 归一化（单位范数）
        norms = np.linalg.norm(E, axis=1, keepdims=True) + 1e-9
        E = E / norms
        return E  # [C, F]

    def fit(self, centroids: np.ndarray, top_clusters: np.ndarray, k_overlap: int):
        """
        拟合 64 个元簇：cluster2meta 映射表
        """
        C = centroids.shape[0]
        if self.mode == "cooccur":
            feats = self._build_cooccur_embeddings(top_clusters, C, k_overlap)  # [C, F]
        elif self.mode == "centroid":
            feats = centroids.astype(np.float32)                                 # [C, d]
            # 归一化避免量纲影响
            norms = np.linalg.norm(feats, axis=1, keepdims=True) + 1e-9
            feats = feats / norms
        else:
            raise ValueError(f"Unknown mode={self.mode}")

        labels = kmeans_np(feats, k=self.M, iters=25, seed=self.seed)  # [C]
        self.cluster2meta = labels.astype(np.int32)

    def sign_set(self, cluster_ids: np.ndarray) -> np.uint64:
        """
        对 top-m 簇 id 集合生成 64-bit 掩码：命中的元簇位置置 1。
        """
        sig = np.uint64(0)
        ids = np.asarray(cluster_ids, dtype=np.int64)
        for cid in ids:
            mc = int(self.cluster2meta[int(cid)]) & 63
            sig |= (np.uint64(1) << np.uint64(mc))
        return sig

    @staticmethod
    def similarity(sig_q: np.uint64, sig_x: np.uint64) -> float:
        return float(popcnt_u64(np.uint64(sig_q & sig_x)))  # AND+POPCNT

# =========================
# 主流程：HMB 训练 + 基于 HMB 的近似排名评测
# =========================
def main():
    # -------- 参数区（按需修改） --------
    datasets = ['sift']              # 数据集名称（路径与原脚本一致）
    K = 1024                         # 簇总数（需与数据一致）
    k_overlap = 64                   # 每个集合大小（top-m clusters）
    nprobe = 10                      # 查询探测簇数量
    top_x_values = [1000]            # 评测 cutoff（取前多少打分向量算 recall）
    gt_neighbors = 10                # GT 前 k，通常与 recall@k 对应
    max_queries = 1000               # 最多处理多少个 query

    # HMB 参数
    M_bits = 64                      # 元簇数量（位数）
    hmb_mode = "cooccur"             # "cooccur" 或 "centroid"
    cooccur_feat_dim = 64            # 共现特征维度（hash 桶数）
    cooccur_sample_size = 20000      # 抽样多少条 base 向量来统计共现
    seed = 1234

    # -----------------------------------
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

        # 加载数据
        print("Loading data...")
        queries = read_fvecs(query_path)                    # [Q, d]
        groundtruth = read_ivecs(gt_path)[:, :gt_neighbors] # [Q, gt_neighbors]
        centroids = read_fvecs(centroids_path)              # [K, d]
        top_clusters = read_ivecs(top_clusters_path)        # [N, >=k_overlap]
        cluster_ids = read_ivecs(cluster_ids_path).flatten()# [N]
        num_queries = min(max_queries, queries.shape[0])

        # 训练 HMB（构建 cluster→meta-cluster 映射）
        hmb = HMBSignature(M_bits=M_bits, mode=hmb_mode, seed=seed,
                           cooccur_feat_dim=cooccur_feat_dim,
                           cooccur_sample_size=cooccur_sample_size)
        print(f"Fitting HMB mapping (mode={hmb_mode}, M={M_bits}) ...")
        hmb.fit(centroids, top_clusters, k_overlap)

        # 评测容器
        recall_scores = {x: [] for x in top_x_values}

        # 查询循环（仅显示进度条）
        for qidx in tqdm(range(num_queries), desc=f"HMB(M={M_bits},{hmb_mode}) → ranking"):
            qv = queries[qidx:qidx+1]  # [1,d]

            # 1) q→centroid 距离并取 nprobe 个最近簇
            distances = np.sum((qv - centroids) ** 2, axis=1)  # [K]
            nearest_clusters = np.argsort(distances)[:nprobe]

            # 2) 收集这些簇中的 base 向量作为 probe 集合
            probe_vector_ids: List[int] = []
            for cid in nearest_clusters:
                vids = np.where(cluster_ids == cid)[0]
                if vids.size:
                    probe_vector_ids.extend(vids.tolist())
            if not probe_vector_ids:
                continue

            # 3) 生成 query 的 HMB 签名：q 的 top-k 簇 → OR 到元簇位
            q_order = np.argsort(distances)
            q_top = q_order[:k_overlap].astype(np.int32)
            q_sig = hmb.sign_set(q_top)  # uint64

            # 4) 对 probe 向量生成签名与打分
            scores = np.empty(len(probe_vector_ids), dtype=np.float32)
            for i, vid in enumerate(probe_vector_ids):
                vec_top = top_clusters[vid, :k_overlap].astype(np.int32, copy=False)
                x_sig = hmb.sign_set(vec_top)  # uint64
                scores[i] = HMBSignature.similarity(q_sig, x_sig)  # AND+POPCNT

            # 5) 统计 recall
            gt_ids = groundtruth[qidx]
            for top_x in top_x_values:
                if top_x <= len(probe_vector_ids):
                    r = compute_recall_by_scores(probe_vector_ids, scores, gt_ids, top_x)
                    recall_scores[top_x].append(r)

        # -------- 汇总 --------
        print(f"\n=== Average recall for {dataset} | METHOD=HMB(M={M_bits}, mode={hmb_mode}) ===")
        for x in top_x_values:
            vals = recall_scores[x]
            avg = float(np.mean(vals)) if len(vals) else 0.0
            print(f"  top-{x}: {avg:.4f}")

if __name__ == '__main__':
    main()
