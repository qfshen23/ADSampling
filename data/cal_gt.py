import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import gc
import psutil
import os

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
    """处理一批查询向量 - 优化版本"""
    query_batch, base, k = args
    batch_size = len(query_batch)
    results = np.zeros((batch_size, k), dtype=np.int32)
    
    for i, q in enumerate(query_batch):
        # 使用平方距离避免开方运算
        distances = np.sum((base - q) ** 2, axis=1)
        # 使用argpartition而不是argsort，更高效
        if k < len(distances):
            indices = np.argpartition(distances, k-1)[:k]
            # 对top-k结果排序
            indices = indices[np.argsort(distances[indices])]
        else:
            indices = np.argsort(distances)
        results[i] = indices[:k]
    
    return results

def get_memory_usage():
    """获取当前内存使用情况（GB）"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024 / 1024

def compute_groundtruth(dataset='sift', k=100, use_multiprocessing=False):
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
    nb = base.shape[0]
    print(f"基础向量: {nb}, 查询向量: {nq}, 维度: {base.shape[1]}")
    print(f"当前内存使用: {get_memory_usage():.2f}GB")
    
    # 检查可用内存
    available_memory = psutil.virtual_memory().available / 1024 / 1024 / 1024
    print(f"系统可用内存: {available_memory:.2f}GB")
    
    if not use_multiprocessing or available_memory < 8:
        print("使用单线程模式（内存受限或手动指定）")
        # 单线程处理
        groundtruth = np.zeros((nq, k), dtype=np.int32)
        
        for i in tqdm(range(nq), desc="计算groundtruth"):
            # 使用平方距离避免开方运算
            distances = np.sum((base - query[i]) ** 2, axis=1)
            
            # 使用argpartition更高效
            if k < nb:
                indices = np.argpartition(distances, k-1)[:k]
                indices = indices[np.argsort(distances[indices])]
            else:
                indices = np.argsort(distances)[:k]
            
            groundtruth[i] = indices
            
            # 每1000个查询清理一次内存
            if (i + 1) % 1000 == 0:
                gc.collect()
    else:
        print("使用多进程模式")
        # 减少工作进程数量和批次大小
        num_workers = min(8, cpu_count() // 2)  # 限制最大进程数
        batch_size = max(1, min(50, nq // (num_workers * 2)))  # 减小批次大小
        print(f"使用 {num_workers} 个工作进程，批次大小: {batch_size}")
        
        # 准备批次数据
        batches = []
        for i in range(0, nq, batch_size):
            end = min(i + batch_size, nq)
            batches.append((query[i:end], base, k))
        
        # 并行处理
        groundtruth = np.zeros((nq, k), dtype=np.int32)
        try:
            with Pool(num_workers) as pool:
                results = list(tqdm(pool.imap(process_batch, batches), total=len(batches)))
            
            # 合并结果
            current_idx = 0
            for batch_result in results:
                batch_size_actual = len(batch_result)
                groundtruth[current_idx:current_idx + batch_size_actual] = batch_result
                current_idx += batch_size_actual
        except Exception as e:
            print(f"多进程处理失败: {e}")
            print("回退到单线程模式")
            return compute_groundtruth(dataset, k, use_multiprocessing=False)
    
    print(f"保存结果到: {gt_path}")
    write_ivecs(gt_path, groundtruth)
    print("完成!")

if __name__ == '__main__':
    # 检查系统内存
    available_memory = psutil.virtual_memory().available / 1024 / 1024 / 1024
    print(f"系统可用内存: {available_memory:.2f}GB")
    
    datasets = ['deep10m','spacev10m', 'bigann10m']
    for dataset in datasets:
        try:
            print(f"\n{'='*50}")
            print(f"开始处理数据集: {dataset}")
            
            # 根据内存情况决定是否使用多进程
            use_mp = available_memory > 16
            compute_groundtruth(dataset, k=100, use_multiprocessing=use_mp)
            
        except Exception as e:
            print(f"处理 {dataset} 时出错: {e}")
            continue
