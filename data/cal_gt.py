import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

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

def process_batch(args):
    """处理一批查询向量"""
    query_batch, base, k = args
    batch_size = len(query_batch)
    results = np.zeros((batch_size, k), dtype=np.int32)
    
    for i, q in enumerate(query_batch):
        distances = np.linalg.norm(base - q, axis=1)
        results[i] = np.argsort(distances)[:k]
    
    return results

def compute_groundtruth(dataset='sift', k=100):
    # 文件路径
    base_path = f'/data/vector_datasets/{dataset}'
    base_fvecs_path = f'{base_path}/{dataset}_base.fvecs'
    query_fvecs_path = f'{base_path}/{dataset}_query.fvecs'
    gt_path = f'{base_path}/{dataset}_groundtruth.ivecs'

    print(f"处理数据集: {dataset}")
    print("读取数据...")
    base = read_fvecs(base_fvecs_path)
    query = read_fvecs(query_fvecs_path)
    
    nq = query.shape[0]
    print(f"计算 {nq} 个查询向量的 top-{k} 近邻...")
    
    # 设置并行参数
    num_workers = cpu_count()
    batch_size = max(1, nq // (num_workers * 4))  # 每个批次的大小
    print(f"使用 {num_workers} 个工作进程，批次大小: {batch_size}")
    
    # 准备批次数据
    batches = []
    for i in range(0, nq, batch_size):
        end = min(i + batch_size, nq)
        batches.append((query[i:end], base, k))
    
    # 并行处理
    groundtruth = np.zeros((nq, k), dtype=np.int32)
    with Pool(num_workers) as pool:
        results = list(tqdm(pool.imap(process_batch, batches), total=len(batches)))
    
    # 合并结果
    current_idx = 0
    for batch_result in results:
        batch_size = len(batch_result)
        groundtruth[current_idx:current_idx + batch_size] = batch_result
        current_idx += batch_size
    
    print(f"保存结果到: {gt_path}")
    write_ivecs(gt_path, groundtruth)
    print("完成!")

if __name__ == '__main__':
    datasets = ['openai1536', 'openai3072']
    for dataset in datasets:
        compute_groundtruth(dataset, k=100)
