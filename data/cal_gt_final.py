import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import gc
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

def compute_groundtruth_single_thread(dataset='sift', k=100):
    """单线程版本，内存使用最小"""
    base_path = f'/data/vector_datasets/{dataset}'
    base_fvecs_path = f'{base_path}/{dataset}_base.fvecs'
    query_fvecs_path = f'{base_path}/{dataset}_query.fvecs'
    gt_path = f'{base_path}/{dataset}_groundtruth.ivecs'

    print(f"处理数据集: {dataset} (单线程模式)")
    print("读取数据...")
    base = read_fvecs(base_fvecs_path)
    query = read_fvecs(query_fvecs_path)
    
    nq = query.shape[0]
    nb = base.shape[0]
    print(f"基础向量: {nb}, 查询向量: {nq}, 维度: {base.shape[1]}")
    
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
    
    print(f"保存结果到: {gt_path}")
    write_ivecs(gt_path, groundtruth)
    print("完成!")

def compute_groundtruth_multiprocess(dataset='sift', k=100, max_workers=4, batch_size=20):
    """多进程版本，限制内存使用"""
    base_path = f'/data/vector_datasets/{dataset}'
    base_fvecs_path = f'{base_path}/{dataset}_base.fvecs'
    query_fvecs_path = f'{base_path}/{dataset}_query.fvecs'
    gt_path = f'{base_path}/{dataset}_groundtruth.ivecs'

    print(f"处理数据集: {dataset} (多进程模式)")
    print("读取数据...")
    base = read_fvecs(base_fvecs_path)
    query = read_fvecs(query_fvecs_path)
    
    nq = query.shape[0]
    nb = base.shape[0]
    print(f"基础向量: {nb}, 查询向量: {nq}, 维度: {base.shape[1]}")
    
    # 限制工作进程数量和批次大小
    num_workers = min(max_workers, cpu_count() // 2, 4)
    actual_batch_size = min(batch_size, max(1, nq // (num_workers * 2)))
    print(f"使用 {num_workers} 个工作进程，批次大小: {actual_batch_size}")
    
    # 准备批次数据
    batches = []
    for i in range(0, nq, actual_batch_size):
        end = min(i + actual_batch_size, nq)
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
        return compute_groundtruth_single_thread(dataset, k)
    
    print(f"保存结果到: {gt_path}")
    write_ivecs(gt_path, groundtruth)
    print("完成!")

def compute_groundtruth(dataset='sift', k=100, use_multiprocessing=None):
    """主函数，自动选择处理模式"""
    if use_multiprocessing is None:
        # 自动判断：对于大数据集默认使用单线程
        large_datasets = ['deep10m', 'spacev10m', 'bigann10m', 'sift10m', 'tiny5m']
        use_multiprocessing = dataset not in large_datasets
    
    if use_multiprocessing:
        try:
            compute_groundtruth_multiprocess(dataset, k, max_workers=4, batch_size=20)
        except:
            print("多进程模式失败，使用单线程模式")
            compute_groundtruth_single_thread(dataset, k)
    else:
        compute_groundtruth_single_thread(dataset, k)

if __name__ == '__main__':
    datasets = ['deep10m', 'spacev10m', 'bigann10m']
    
    print("推荐使用单线程模式以避免内存问题")
    print("如果要强制使用多进程，请修改 use_multiprocessing=True")
    
    for dataset in datasets:
        try:
            print(f"\n{'='*50}")
            print(f"开始处理数据集: {dataset}")
            
            # 对于大数据集，强制使用单线程
            compute_groundtruth(dataset, k=100, use_multiprocessing=False)
            
        except Exception as e:
            print(f"处理 {dataset} 时出错: {e}")
            continue
