# search_ivfpq.py
import os
import time
import struct
import numpy as np
import faiss

# ======= 配置 =======
source = '/data/vector_datasets/'
datasets = ['sift']
'''
5
10
15
20
25
30
35
40
45
50
'''
# 搜索参数
NPROBE_LIST = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]  # IVF 搜索探测数列表
TOPK = 10           # 评估 Recall@TOPK
WARMUP_QUERIES = 10 # 预热（避免冷启动影响计时）

# 索引路径
K = 1024
M = 16
nbits = 8
def idx_path_of(dataset):
    return os.path.join(source, dataset, f'{dataset}_ivfpq_IVF{K}_PQ{M}x{nbits}.faiss')

# ======= IO 工具 =======
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
    iv = np.fromfile(filename, dtype=np.int32)
    if iv.size == 0:
        return np.zeros((0, 0), dtype=np.int32)
    dim = iv.view(np.int32)[0]
    assert dim > 0
    iv = iv.reshape(-1, 1 + dim)
    if not np.all(iv[:, 0] == dim):
        raise IOError("Non-uniform vector sizes in " + filename)
    return iv[:, 1:]

# ======= 评估 =======
def recall_at_k(I_pred, I_gt, k):
    """
    I_pred: (nq, topk) 预测ID
    I_gt:   (nq, >=k)  ground truth 前k列或更多列
    """
    nq, tk = I_pred.shape
    k = min(k, I_gt.shape[1], tk)
    # 逐行统计交集
    correct = 0
    for i in range(nq):
        gt_set = set(I_gt[i, :k])
        pred = I_pred[i, :k]  # 常用@k
        correct += np.sum(np.isin(pred, list(gt_set)))
    return correct / (nq * k)

# ======= 主流程 =======
if __name__ == '__main__':
    for dataset in datasets:
        path = os.path.join(source, dataset)
        index_path = idx_path_of(dataset)
        query_path = os.path.join(path, f'{dataset}_query.fvecs')
        gt_path = os.path.join(path, f'{dataset}_groundtruth.ivecs')

        print(f'=== Search IVFPQ for {dataset} ===')
        print(f'Index:  {index_path}')
        print(f'Query:  {query_path}')
        print(f'GT:     {gt_path}')

        if not os.path.exists(index_path):
            raise FileNotFoundError(f'Index not found: {index_path}')
        if not os.path.exists(query_path):
            raise FileNotFoundError(query_path)
        if not os.path.exists(gt_path):
            raise FileNotFoundError(gt_path)

        # 加载索引
        index = faiss.read_index(index_path)
        if not isinstance(index, faiss.IndexIVF):
            print('Warning: index is not an IVF type; nprobe ignored.')
            continue

        # 读查询与GT
        Q = read_fvecs(query_path)
        I_gt = read_ivecs(gt_path)
        nq, d = Q.shape
        print(f'Loaded queries: nq={nq}, d={d}, topk={TOPK}')
        print()

        # 存储结果
        results = []

        # 遍历每个 nprobe 值
        for nprobe in NPROBE_LIST:
            index.nprobe = nprobe
            
            # 预热
            if nq > 0:
                qwarm = Q[:min(WARMUP_QUERIES, nq)]
                index.search(qwarm, TOPK)

            # 实测耗时
            t0 = time.perf_counter()
            D, I = index.search(Q, TOPK)
            t1 = time.perf_counter()

            # 统计
            elapsed = t1 - t0
            qps = nq / elapsed if elapsed > 0 else float('inf')
            rec = recall_at_k(I, I_gt, TOPK) * 100.0

            results.append((nprobe, rec, qps))
            print(f'nprobe={nprobe:4d}  Recall@{TOPK}={rec:6.2f}%  QPS={qps:10.2f}')

        # 输出最终结果表格
        print()
        print('--- Final Results (recall \\t qps) ---')
        for _, rec, qps in results:
            print(f'{rec:.2f}\t{qps:.2f}')
        print()
