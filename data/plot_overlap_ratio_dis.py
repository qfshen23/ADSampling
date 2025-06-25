import numpy as np
import matplotlib.pyplot as plt
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

def compute_block_overlaps(
    dataset='sift',
    cluster_ranges=[[0,16], [16,32], [32,48], [48,64], [64,80], [80,96], [96,112], [112,128]] ):
    # File paths
    base_path = f'/data/vector_datasets/{dataset}'
    base_fvecs_path = f'{base_path}/{dataset}_base.fvecs'
    query_fvecs_path = f'{base_path}/{dataset}_query.fvecs'
    gt_ivecs_path = f'{base_path}/{dataset}_groundtruth_10000.ivecs'
    K = 1024  # Number of clusters
    centroids_fvecs_path = f'{base_path}/{dataset}_centroid_{K}.fvecs'
    base_cluster_rank_path = f'{base_path}/{dataset}_top_clusters_{K}.ivecs'

    query = read_fvecs(query_fvecs_path)[:1000]  # [nq, dim]
    base = read_fvecs(base_fvecs_path)
    centroids = read_fvecs(centroids_fvecs_path)  # [nlist, dim]
    gt = read_ivecs(gt_ivecs_path)[:1000]       # [nq, gt_len]
    base_cluster_rank = read_ivecs(base_cluster_rank_path)  # [nb, nlist]

    nq = query.shape[0]
    gt_len = gt.shape[1]
    nlist = centroids.shape[0]

    # 预先算好所有 query 的 cluster_rank
    query_cluster_rank = []
    for qid in tqdm(range(nq)):
        q = query[qid]
        dists = np.linalg.norm(centroids - q, axis=1)
        cluster_rank = np.argsort(dists)
        query_cluster_rank.append(cluster_rank)
    query_cluster_rank = np.stack(query_cluster_rank)  # [nq, nlist]

    results = { (r0, r1): np.zeros((nq, gt_len)) for (r0, r1) in cluster_ranges }

    for qid in range(nq):
        q_clusters = query_cluster_rank[qid]
        for gid in range(gt_len):
            base_id = gt[qid, gid]
            base_clusters = base_cluster_rank[base_id]
            for (r0, r1) in cluster_ranges:
                # 注意：query/base 都取同样区间
                q_blk = q_clusters[r0:r1]
                b_blk = base_clusters[r0:r1]
                overlap = len(set(q_blk) & set(b_blk)) / (r1 - r0)
                results[(r0, r1)][qid, gid] = overlap

    # 画图，每个区间一张
    x = np.arange(1, gt_len + 1)  # gt idx from 1
    for (r0, r1), mat in results.items():
        # mat: [nq, gt_len]，画每个gt idx的均值曲线
        mean_overlap = mat.mean(axis=0)
        plt.figure()
        plt.plot(x, mean_overlap, marker='.')
        plt.xlabel("gt idx (1~%d)" % gt_len)
        plt.ylabel("Overlap Ratio (clusters %d-%d)" % (r0+1, r1))
        plt.title("Overlap Ratio for Clusters %d-%d" % (r0+1, r1))
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{dataset}_overlap_{r0+1}_{r1}.png', dpi=400)

# 用法示例
compute_block_overlaps(
    dataset='sift',
    cluster_ranges=[[0,16], [16,32], [32,48], [48,64], [64,80], [80,96], [96,112], [112,128]]
)
