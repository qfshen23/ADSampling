import numpy as np
from tqdm import tqdm
import os
import gc

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

def write_ivecs(filename, array):
    """将数组写入ivecs格式文件"""
    with open(filename, 'wb') as f:
        for row in array:
            dim = len(row)
            f.write(np.array([dim], dtype='int32').tobytes())
            f.write(row.astype('int32').tobytes())

def compute_groundtruth_simple(dataset='sift', k=100):
    """简单单线程版本，内存使用最小"""
    base_path = f'/data/vector_datasets/{dataset}'
    base_fvecs_path = f'{base_path}/{dataset}_base.fvecs'
    query_fvecs_path = f'{base_path}/{dataset}_query.fvecs'
    gt_path = f'{base_path}/{dataset}_groundtruth.ivecs'

    print(f"处理数据集: {dataset}")
    
    # 读取数据
    print("读取基础数据...")
    base = read_fvecs(base_fvecs_path)
    print("读取查询数据...")
    query = read_fvecs(query_fvecs_path)
    
    nb, dim = base.shape
    nq = query.shape[0]
    print(f"基础向量: {nb}, 查询向量: {nq}, 维度: {dim}")
    
    # 逐个处理查询
    groundtruth = np.zeros((nq, k), dtype=np.int32)
    
    for i in tqdm(range(nq), desc="计算groundtruth"):
        # 计算距离（使用平方距离避免开方）
        distances = np.sum((base - query[i]) ** 2, axis=1)
        
        # 找到top-k（使用argpartition更高效）
        if k < nb:
            indices = np.argpartition(distances, k-1)[:k]
            # 对top-k结果排序
            indices = indices[np.argsort(distances[indices])]
        else:
            indices = np.argsort(distances)[:k]
        
        groundtruth[i] = indices
        
        # 每1000个查询清理一次内存
        if (i + 1) % 1000 == 0:
            gc.collect()
    
    print(f"保存结果到: {gt_path}")
    write_ivecs(gt_path, groundtruth)
    print("完成!")

if __name__ == '__main__':
    datasets = ['deep10m', 'spacev10m', 'bigann10m']
    for dataset in datasets:
        try:
            compute_groundtruth_simple(dataset, k=100)
        except Exception as e:
            print(f"处理 {dataset} 时出错: {e}")
            continue
