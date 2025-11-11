# build_ivfpq.py
import os
import time
import struct
import numpy as np
import faiss

# ======= 配置 =======
source = '/data/vector_datasets/'
datasets = ['sift', 'gist']

# IVF & PQ 参数
K = 1024          # nlist
M = 16            # PQ m（子空间个数）
nbits = 8         # 每个子空间的编码bit数（常用8）
metric = faiss.METRIC_L2  # 向量度量，L2 / IP

# 训练数据量（可选下采样）
TRAIN_MAX = 2_000_000     # 训练最大样本数（根据内存调整）
RANDOM_SEED = 123

# 索引文件名
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

def write_index(index, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    faiss.write_index(index, path)

# ======= 主流程 =======
if __name__ == '__main__':
    np.random.seed(RANDOM_SEED)

    for dataset in datasets:
        path = os.path.join(source, dataset)
        base_path = os.path.join(path, f'{dataset}_base.fvecs')
        out_index = idx_path_of(dataset)

        print(f'=== Building IVFPQ for {dataset} ===')
        print(f'Base vectors: {base_path}')
        if not os.path.exists(base_path):
            raise FileNotFoundError(base_path)

        # 读 base
        X = read_fvecs(base_path)
        nb, d = X.shape
        print(f'Loaded base: nb={nb}, d={d}')

        # 准备训练数据
        if nb > TRAIN_MAX:
            train_sel = np.random.choice(nb, TRAIN_MAX, replace=False)
            Xtrain = X[train_sel].copy()
            print(f'Training on subsampled {len(Xtrain)} vectors')
        else:
            Xtrain = X

        # 创建索引
        index_str = f'IVF{K},PQ{M}x{nbits}'
        print(f'Index factory spec: {index_str}')
        index = faiss.index_factory(d, index_str, metric)

        # 训练
        t0 = time.perf_counter()
        print('Training...')
        index.train(Xtrain)
        t1 = time.perf_counter()
        print(f'Trained in {t1 - t0:.2f}s')

        # 添加数据
        print('Adding base vectors...')
        index.add(X)
        t2 = time.perf_counter()
        print(f'Added in {t2 - t1:.2f}s; ntotal={index.ntotal}')

        # 保存
        print(f'Writing index to: {out_index}')
        write_index(index, out_index)
        print('Done.\n')
