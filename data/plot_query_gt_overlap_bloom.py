import numpy as np
import os
from tqdm import tqdm
from typing import Dict, List

# =========================
# I/O helpers (与你原版一致)
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
# 评测/统计工具
# =========================
def compute_recall_by_scores(probe_vector_ids, scores, gt_vector_ids, top_x):
    """按打分（降序）选前 top_x，统计与 GT 的交集比例。"""
    if len(probe_vector_ids) == 0:
        return 0.0
    scores = np.asarray(scores)
    # 取前 top_x 的索引（不完全排序以加速）
    m = min(top_x, len(probe_vector_ids))
    idx = np.argpartition(scores, -m)[-m:]
    # 严格从大到小排序（可选）
    order = np.argsort(scores[idx])[::-1]
    chosen = {probe_vector_ids[i] for i in idx[order]}
    gt_set = set(int(x) for x in gt_vector_ids)
    inter = len(chosen.intersection(gt_set))
    return inter / max(1, len(gt_vector_ids))

def compute_statistics(arr):
    arr = np.asarray(arr)
    if arr.size == 0:
        return {'min':0.0,'p25':0.0,'mean':0.0,'p75':0.0,'max':0.0}
    return {
        'min': float(np.min(arr)),
        'p25': float(np.percentile(arr, 25)),
        'mean': float(np.mean(arr)),
        'p75': float(np.percentile(arr, 75)),
        'max': float(np.max(arr)),
    }

# =========================
# Hashed Overlap Sketch（每个 cluster 固定 mask）
# =========================
class HashedOverlapSketch:
    """
    每个 cluster 分配一个固定的稀疏 bitmask（常重码），向量签名是这些 mask 的 OR：
    - m_bits: 总位数（64 的倍数，例 256/512/1024）
    - t_per_cluster: 每个 cluster 置位 bit 数
    sign(cluster_ids) -> np.uint64[m_bits//64]
    similarity(sig1, sig2, mode):
        - "and_popcnt":  popcnt(AND)     （越大 overlap 越大）
        - "bit_jaccard": popcnt(AND)/popcnt(OR) ∈ [0,1]
    """
    def __init__(self, num_clusters: int,
                 m_bits: int = 256,
                 t_per_cluster: int = 3,
                 seed: int = 42):
        assert m_bits % 64 == 0 and m_bits > 0
        assert t_per_cluster >= 1
        self.num_clusters = int(num_clusters)
        self.m_bits = int(m_bits)
        self.words = self.m_bits // 64
        self.t_per_cluster = int(t_per_cluster)
        self.seed = int(seed)

        rng = np.random.default_rng(seed)
        # 为每个 cluster 生成一个固定的 mask
        # mask_table[cid] 是一个 shape = [words] 的 uint64 向量
        self.mask_table = np.zeros((self.num_clusters, self.words),
                                   dtype=np.uint64)
        for cid in range(self.num_clusters):
            # 为该 cluster 选 t_per_cluster 个不同 bit 位置
            pos = rng.choice(self.m_bits,
                             size=self.t_per_cluster,
                             replace=False)
            for p in pos:
                w = int(p) >> 6           # // 64
                b = int(p) & 63          # % 64
                self.mask_table[cid, w] |= (np.uint64(1) << np.uint64(b))

    def sign(self, cluster_ids: np.ndarray) -> np.ndarray:
        """把一个 int cluster 集合编码成定长位图签名。"""
        sig = np.zeros(self.words, dtype=np.uint64)
        if cluster_ids is None:
            return sig
        ids = np.asarray(cluster_ids, dtype=np.int64)
        if ids.size == 0:
            return sig
        # 保证落在合法 [0, num_clusters)
        ids = ids[(ids >= 0) & (ids < self.num_clusters)]
        if ids.size == 0:
            return sig
        # OR 所有对应的 cluster mask（vectorized）
        sig = np.bitwise_or.reduce(self.mask_table[ids], axis=0)
        return sig

    @staticmethod
    def _popcnt_words(words: np.ndarray) -> int:
        total = 0
        for w in words:
            total += int(w).bit_count()
        return total

    def similarity(self, sig1: np.ndarray, sig2: np.ndarray,
                   mode: str = "and_popcnt") -> float:
        andw = (sig1 & sig2).astype(np.uint64)
        if mode == "and_popcnt":
            return float(self._popcnt_words(andw))
        elif mode == "bit_jaccard":
            orw = (sig1 | sig2).astype(np.uint64)
            a = self._popcnt_words(andw)
            o = self._popcnt_words(orw)
            return float(a) / float(o) if o > 0 else 0.0
        else:
            raise ValueError(f"Unknown score mode: {mode}")

# =========================
# 主流程：Hashed Overlap Sketch 近似排名 → 选 DCO
# =========================
def main():
    # -------- 参数区（按需修改） --------
    datasets = ['sift']              # 数据集列表
    K_cfg = 1024                     # 预期簇总数（用于选文件名），实际用 centroids.shape[0]
    k_overlap = 64                   # 每个集合大小（top-k clusters）
    nprobe = 10                      # 查询探测的最近簇数量
    top_x_values = [2000]            # 评测 cutoff
    gt_neighbors = 10                # 取前多少 GT 为正例
    max_queries = 1000               # 最多处理多少个 query

    # Sketch 参数
    m_bits = 256                     # 256/512/1024... 位；越大越稳
    t_per_cluster = 3                # 每个 cluster 置位个数
    sketch_seed = 12345
    score_mode = "and_popcnt"        # 或 "bit_jaccard"

    # DCO 预算：每个 query 选多少向量去做 DCO（按分数 Top-M）
    dco_budget = 50000               # 设为 0/None 可跳过输出 DCO 列表

    # -----------------------------------
    for dataset in datasets:
        print(f"\n=== Processing dataset: {dataset} ===")
        base_path = f'/data/vector_datasets/{dataset}'
        query_path = f'{base_path}/{dataset}_query.fvecs'
        gt_path = f'{base_path}/{dataset}_groundtruth_10000.ivecs'
        centroids_path = f'{base_path}/{dataset}_centroid_{K_cfg}.fvecs'
        top_clusters_path = f'{base_path}/{dataset}_top_clusters_1024.ivecs'
        cluster_ids_path = f'{base_path}/{dataset}_cluster_id_{K_cfg}.ivecs'

        missing_files = [p for p in [query_path, gt_path, centroids_path,
                                     top_clusters_path, cluster_ids_path]
                         if not os.path.exists(p)]
        if missing_files:
            print(f"Skipping {dataset} - {missing_files} missing")
            continue

        print("Loading data...")
        queries = read_fvecs(query_path)
        groundtruth = read_ivecs(gt_path)[:, :gt_neighbors]
        centroids = read_fvecs(centroids_path)
        top_clusters = read_ivecs(top_clusters_path)      # [N, >=k_overlap]
        cluster_ids = read_ivecs(cluster_ids_path)        # [N, 1]

        print(f"Queries shape: {queries.shape}")
        print(f"Groundtruth shape: {groundtruth.shape}")
        print(f"Centroids shape: {centroids.shape}")
        print(f"Top clusters shape: {top_clusters.shape}")
        print(f"Cluster IDs shape: {cluster_ids.shape}")

        num_queries = min(max_queries, queries.shape[0])
        K_real = centroids.shape[0]
        print(f"Detected num_clusters (K_real) = {K_real}")

        # 评测容器
        recall_scores = {x: [] for x in top_x_values}

        # HashedOverlapSketch 实例（共享）
        sketch = HashedOverlapSketch(
            num_clusters=K_real,
            m_bits=m_bits,
            t_per_cluster=t_per_cluster,
            seed=sketch_seed
        )

        # 懒缓存：只对触及到的 base 向量计算签名
        sig_cache: Dict[int, np.ndarray] = {}

        print("\nHashed-overlap-only ranking → choose vectors for DCO...")
        cluster_lbl = cluster_ids.flatten()

        for qidx in tqdm(range(num_queries)):
            qv = queries[qidx:qidx+1]

            # 1) 计算 query→centroid 的 L2 距离，取 nprobe 个最近簇
            distances = np.sum((qv - centroids) ** 2, axis=1)  # shape: [K_real]
            nearest_clusters = np.argsort(distances)[:nprobe]

            # 2) 收集这些簇中的 base 向量作为 probe 集合
            probe_vector_ids: List[int] = []
            for cid in nearest_clusters:
                vids = np.where(cluster_lbl == cid)[0]
                if vids.size:
                    probe_vector_ids.extend(vids.tolist())
            if not probe_vector_ids:
                continue

            # 3) 取 query 的 top-k_overlap 簇集合并生成签名
            q_order = np.argsort(distances)            # 0..K_real-1
            q_set = q_order[:k_overlap].astype(np.int32)
            q_sig = sketch.sign(q_set)

            # 4) 对 probe 向量计算 overlap 分数（直接用于 DCO 排名）
            scores = np.empty(len(probe_vector_ids), dtype=np.float32)
            for i, vid in enumerate(probe_vector_ids):
                if vid not in sig_cache:
                    vec_top = top_clusters[vid, :k_overlap].astype(
                        np.int32, copy=False)
                    sig_cache[vid] = sketch.sign(vec_top)
                scores[i] = sketch.similarity(q_sig, sig_cache[vid],
                                              mode=score_mode)

            # 5) （可选）输出本 query 的“要做 DCO 的候选列表”（按分数 Top-M）
            if dco_budget and dco_budget > 0:
                m = min(dco_budget, len(probe_vector_ids))
                idx_topm = np.argpartition(scores, -m)[-m:]
                ord_topm = np.argsort(scores[idx_topm])[::-1]
                dco_ids = [probe_vector_ids[j] for j in idx_topm[ord_topm]]
                # TODO: 在此把 dco_ids 交给你的 DCO 模块
                # 例如：run_dco_for_query(qidx, dco_ids)

            # 6) 打印分数分布（诊断用）与 recall 评测
            stats = compute_statistics(scores)
            print(f"Query {qidx} - overlap sketch score stats ({score_mode}): "
                  f"min={stats['min']:.4f}, p25={stats['p25']:.4f}, "
                  f"mean={stats['mean']:.4f}, p75={stats['p75']:.4f}, "
                  f"max={stats['max']:.4f}")

            gt_ids = groundtruth[qidx]
            for top_x in top_x_values:
                if top_x <= len(probe_vector_ids):
                    r = compute_recall_by_scores(
                        probe_vector_ids, scores, gt_ids, top_x
                    )
                    recall_scores[top_x].append(r)
                else:
                    print(f"  [Warn] top-{top_x} > #probes ({len(probe_vector_ids)})")

        # -------- 汇总 --------
        print(f"\n=== Average recall (Hashed-overlap-only ranking) for {dataset} ===")
        for x in top_x_values:
            vals = recall_scores[x]
            avg = float(np.mean(vals)) if len(vals) else 0.0
            print(f"  top-{x}: {avg:.4f}  "
                  f"(m_bits={m_bits}, t_per_cluster={t_per_cluster}, "
                  f"mode={score_mode})")

if __name__ == '__main__':
    main()
