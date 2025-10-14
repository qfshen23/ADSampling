import numpy as np
from tqdm import tqdm
import os
import gc
import psutil

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

def get_memory_usage():
    """获取当前内存使用情况（GB）"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024 / 1024

def compute_knn_batch(queries, base, k, start_idx=0):
    """计算一批查询的k近邻，使用更高效的方法"""
    batch_size = len(queries)
    results = np.zeros((batch_size, k), dtype=np.int32)
    
    # 使用向量化计算距离
    for i, query in enumerate(queries):
        # 计算所有距离
        distances = np.sum((base - query) ** 2, axis=1)
        # 使用argpartition而不是argsort，更高效
        if k < len(distances):
            indices = np.argpartition(distances, k-1)[:k]
            # 对top-k结果排序
            indices = indices[np.argsort(distances[indices])]
        else:
            indices = np.argsort(distances)
        results[i] = indices[:k]
        
        if (i + 1) % 100 == 0:
            print(f"  批次进度: {i+1}/{batch_size}, 内存使用: {get_memory_usage():.2f}GB")
    
    return results

def compute_groundtruth_chunked(dataset='sift', k=100, max_memory_gb=8):
    """分块计算groundtruth，控制内存使用"""
    # 文件路径
    base_path = f'/data/vector_datasets/{dataset}'
    base_fvecs_path = f'{base_path}/{dataset}_base.fvecs'
    query_fvecs_path = f'{base_path}/{dataset}_query.fvecs'
    gt_path = f'{base_path}/{dataset}_groundtruth.ivecs'

    print(f"处理数据集: {dataset}")
    print("读取查询数据...")
    query = read_fvecs(query_fvecs_path)
    nq = query.shape[0]
    
    print("读取基础数据...")
    base = read_fvecs(base_fvecs_path)
    nb, dim = base.shape
    
    print(f"数据集信息: {nb} 个基础向量, {nq} 个查询向量, 维度: {dim}")
    print(f"当前内存使用: {get_memory_usage():.2f}GB")
    
    # 估算内存使用
    base_memory = base.nbytes / 1024 / 1024 / 1024  # GB
    query_memory = query.nbytes / 1024 / 1024 / 1024  # GB
    print(f"基础数据内存: {base_memory:.2f}GB, 查询数据内存: {query_memory:.2f}GB")
    
    # 计算合适的批次大小
    available_memory = max_memory_gb - base_memory - query_memory - 1  # 预留1GB
    if available_memory <= 0:
        print("警告: 内存可能不足，使用最小批次大小")
        batch_size = 1
    else:
        # 估算每个查询处理需要的临时内存
        temp_memory_per_query = nb * 4 / 1024 / 1024 / 1024  # 距离数组的内存
        batch_size = max(1, int(available_memory / temp_memory_per_query))
        batch_size = min(batch_size, nq)  # 不超过查询总数
    
    print(f"使用批次大小: {batch_size}")
    
    # 分批处理
    groundtruth = np.zeros((nq, k), dtype=np.int32)
    
    for start_idx in tqdm(range(0, nq, batch_size), desc="处理批次"):
        end_idx = min(start_idx + batch_size, nq)
        query_batch = query[start_idx:end_idx]
        
        print(f"\n处理查询 {start_idx} 到 {end_idx-1}")
        batch_results = compute_knn_batch(query_batch, base, k, start_idx)
        groundtruth[start_idx:end_idx] = batch_results
        
        # 强制垃圾回收
        del query_batch, batch_results
        gc.collect()
        
        print(f"批次完成，当前内存使用: {get_memory_usage():.2f}GB")
    
    print(f"保存结果到: {gt_path}")
    write_ivecs(gt_path, groundtruth)
    print("完成!")

def compute_groundtruth_streaming(dataset='sift', k=100, query_chunk_size=100):
    """流式处理版本，进一步减少内存使用"""
    base_path = f'/data/vector_datasets/{dataset}'
    base_fvecs_path = f'{base_path}/{dataset}_base.fvecs'
    query_fvecs_path = f'{base_path}/{dataset}_query.fvecs'
    gt_path = f'{base_path}/{dataset}_groundtruth.ivecs'

    print(f"流式处理数据集: {dataset}")
    
    # 读取基础数据
    print("读取基础数据...")
    base = read_fvecs(base_fvecs_path)
    nb, dim = base.shape
    print(f"基础数据: {nb} 个向量, 维度: {dim}")
    
    # 获取查询数据信息
    with open(query_fvecs_path, 'rb') as f:
        dim_bytes = f.read(4)
        if len(dim_bytes) != 4:
            raise IOError("Cannot read dimension")
        query_dim = np.frombuffer(dim_bytes, dtype=np.int32)[0]
        
        # 计算查询数量
        f.seek(0, 2)  # 移到文件末尾
        file_size = f.tell()
        nq = file_size // ((query_dim + 1) * 4)
    
    print(f"查询数据: {nq} 个向量")
    
    # 创建输出文件
    results = []
    
    # 分块读取和处理查询
    with open(query_fvecs_path, 'rb') as f:
        for chunk_start in tqdm(range(0, nq, query_chunk_size), desc="处理查询块"):
            chunk_end = min(chunk_start + query_chunk_size, nq)
            chunk_size = chunk_end - chunk_start
            
            # 读取查询块
            query_chunk = np.zeros((chunk_size, query_dim), dtype=np.float32)
            for i in range(chunk_size):
                dim_bytes = f.read(4)
                if len(dim_bytes) != 4:
                    break
                dim = np.frombuffer(dim_bytes, dtype=np.int32)[0]
                if dim != query_dim:
                    raise IOError("Inconsistent dimension")
                
                vector_bytes = f.read(dim * 4)
                query_chunk[i] = np.frombuffer(vector_bytes, dtype=np.float32)
            
            # 处理这个块
            print(f"\n处理查询 {chunk_start} 到 {chunk_end-1}")
            chunk_results = compute_knn_batch(query_chunk, base, k)
            results.append(chunk_results)
            
            # 清理内存
            del query_chunk, chunk_results
            gc.collect()
            print(f"内存使用: {get_memory_usage():.2f}GB")
    
    # 合并结果
    print("合并结果...")
    groundtruth = np.vstack(results)
    
    print(f"保存结果到: {gt_path}")
    write_ivecs(gt_path, groundtruth)
    print("完成!")

if __name__ == '__main__':
    # 检查可用内存
    available_memory = psutil.virtual_memory().available / 1024 / 1024 / 1024
    print(f"系统可用内存: {available_memory:.2f}GB")
    
    datasets = ['deep10m', 'spacev10m', 'bigann10m']
    
    for dataset in datasets:
        try:
            print(f"\n{'='*50}")
            print(f"开始处理数据集: {dataset}")
            
            # 根据可用内存选择处理方式
            if available_memory > 16:
                print("使用分块处理模式")
                compute_groundtruth_chunked(dataset, k=100, max_memory_gb=int(available_memory * 0.8))
            else:
                print("使用流式处理模式")
                compute_groundtruth_streaming(dataset, k=100, query_chunk_size=50)
                
        except Exception as e:
            print(f"处理 {dataset} 时出错: {e}")
            continue
