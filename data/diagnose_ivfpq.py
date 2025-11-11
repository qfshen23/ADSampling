#!/usr/bin/env python3
# diagnose_ivfpq.py - 诊断 IVFPQ 索引问题
import os
import numpy as np
import faiss

# ======= 配置 =======
source = '/data/vector_datasets/'
dataset = 'gist'
K = 1024
M = 16
nbits = 8

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

# ======= 主流程 =======
if __name__ == '__main__':
    path = os.path.join(source, dataset)
    index_path = os.path.join(path, f'{dataset}_ivfpq_IVF{K}_PQ{M}x{nbits}.faiss')
    base_path = os.path.join(path, f'{dataset}_base.fvecs')
    query_path = os.path.join(path, f'{dataset}_query.fvecs')
    gt_path = os.path.join(path, f'{dataset}_groundtruth.ivecs')

    print(f'=== 诊断 IVFPQ 索引: {dataset} ===\n')

    # 1. 检查索引基本信息
    print('1. 检查索引基本信息')
    index = faiss.read_index(index_path)
    print(f'   索引类型: {type(index)}')
    print(f'   索引维度: {index.d}')
    print(f'   索引中向量数: {index.ntotal}')
    print(f'   是否已训练: {index.is_trained}')
    
    if hasattr(index, 'nlist'):
        print(f'   聚类数 (nlist): {index.nlist}')
    if hasattr(index, 'nprobe'):
        print(f'   当前 nprobe: {index.nprobe}')
    print()

    # 2. 检查 base 向量
    print('2. 检查 base 向量')
    X = read_fvecs(base_path)
    print(f'   Base 向量数: {X.shape[0]}')
    print(f'   Base 向量维度: {X.shape[1]}')
    print(f'   Base 向量范围: [{X.min():.4f}, {X.max():.4f}]')
    print()

    # 3. 检查查询向量
    print('3. 检查查询向量')
    Q = read_fvecs(query_path)
    print(f'   查询向量数: {Q.shape[0]}')
    print(f'   查询向量维度: {Q.shape[1]}')
    print(f'   查询向量范围: [{Q.min():.4f}, {Q.max():.4f}]')
    print()

    # 4. 检查 Ground Truth
    print('4. 检查 Ground Truth')
    I_gt = read_ivecs(gt_path)
    print(f'   GT 形状: {I_gt.shape}')
    print(f'   GT ID 范围: [{I_gt.min()}, {I_gt.max()}]')
    print(f'   GT 前 3 个查询的前 5 个结果:')
    for i in range(min(3, I_gt.shape[0])):
        print(f'      查询 {i}: {I_gt[i, :5]}')
    print()

    # 5. 暴力搜索验证 GT
    print('5. 用暴力搜索验证 GT (只测试前 10 个查询)')
    index_flat = faiss.IndexFlatL2(X.shape[1])
    index_flat.add(X)
    D_bf, I_bf = index_flat.search(Q[:10], 10)
    
    # 比较前 10 个查询的结果
    match_count = 0
    for i in range(10):
        gt_set = set(I_gt[i, :10])
        bf_set = set(I_bf[i, :10])
        matches = len(gt_set & bf_set)
        match_count += matches
        if i < 3:  # 只打印前 3 个
            print(f'   查询 {i}: GT={I_gt[i, :5]}, BF={I_bf[i, :5]}, 匹配={matches}/10')
    
    gt_accuracy = match_count / 100.0
    print(f'   Ground Truth 准确率: {gt_accuracy * 100:.2f}%')
    print()

    # 6. 测试 IVFPQ 搜索
    print('6. 测试 IVFPQ 搜索 (前 10 个查询, nprobe=64)')
    index.nprobe = 64
    D_ivfpq, I_ivfpq = index.search(Q[:10], 10)
    
    recall_sum = 0
    for i in range(10):
        gt_set = set(I_gt[i, :10])
        ivfpq_set = set(I_ivfpq[i, :10])
        matches = len(gt_set & ivfpq_set)
        recall_sum += matches
        if i < 3:
            print(f'   查询 {i}: GT={I_gt[i, :5]}, IVFPQ={I_ivfpq[i, :5]}, recall={matches}/10')
    
    recall = recall_sum / 100.0
    print(f'   IVFPQ Recall@10: {recall * 100:.2f}%')
    print()

    # 7. 比较距离
    print('7. 比较距离 (查询 0)')
    q0 = Q[0:1]
    D_flat, I_flat = index_flat.search(q0, 10)
    index.nprobe = 64
    D_ivf, I_ivf = index.search(q0, 10)
    
    print(f'   暴力搜索: IDs={I_flat[0][:5]}')
    print(f'   暴力搜索: Dists={D_flat[0][:5]}')
    print(f'   IVFPQ:    IDs={I_ivf[0][:5]}')
    print(f'   IVFPQ:    Dists={D_ivf[0][:5]}')
    print()

    # 8. 检查索引是否正确添加了向量
    print('8. 验证索引完整性')
    if index.ntotal != X.shape[0]:
        print(f'   ⚠️ 警告: 索引中的向量数 ({index.ntotal}) != base 向量数 ({X.shape[0]})')
    else:
        print(f'   ✓ 索引向量数量匹配')
    
    # 检查维度
    if index.d != X.shape[1]:
        print(f'   ⚠️ 警告: 索引维度 ({index.d}) != base 维度 ({X.shape[1]})')
    else:
        print(f'   ✓ 索引维度匹配')

    print('\n=== 诊断完成 ===')

