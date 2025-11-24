import numpy as np
import faiss
import struct
import os
import argparse
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

def read_vecs_fast(filename, show_progress=True):
    """
    快速读取 .vecs 格式的向量文件（优化版本）
    一次性读取所有数据并重新组织，比逐个向量读取快很多
    
    参数:
        filename: 文件路径
        show_progress: 是否显示读取进度
    """
    # 根据文件扩展名推断数据类型
    if filename.endswith(".fvecs"):
        dtype = np.float32
        dtype_size = 4
    elif filename.endswith(".ivecs"):
        dtype = np.int32
        dtype_size = 4
    elif filename.endswith(".bvecs"):
        dtype = np.uint8
        dtype_size = 1
    else:
        raise ValueError(f"未知的 vecs 文件类型: {filename}")
    
    if show_progress:
        print("  📊 分析文件结构...")
    
    # 获取文件大小
    file_size = os.path.getsize(filename)
    
    with open(filename, "rb") as f:
        # 读取第一个向量的维度
        dim = struct.unpack('i', f.read(4))[0]
        
        # 计算每个向量占用的字节数：4字节(维度) + dim * dtype_size
        vec_size = 4 + dim * dtype_size
        
        # 计算总向量数
        n = file_size // vec_size
        
        if show_progress:
            print(f"  📏 检测到 {n:,} 个向量，每个维度 {dim}")
            print(f"  💾 文件大小: {file_size / (1024**3):.2f} GB")
            print(f"  🚀 开始快速读取...")
        
        # 回到文件开头
        f.seek(0)
        
        # 一次性读取所有数据
        all_data = np.fromfile(f, dtype=np.uint8, count=file_size)
    
    if show_progress:
        print(f"  🔄 重组数据结构...")
    
    # 高效方法：使用numpy的视图和切片操作，避免Python循环
    # 将字节数据重新解释为结构化数组
    all_data = all_data.reshape(n, vec_size)
    
    # 跳过每个向量前4字节的维度信息，提取向量数据
    # all_data[:, 4:] 跳过前4列（维度信息）
    vec_data = all_data[:, 4:].copy()  # copy()确保数据连续
    
    # 将字节数据重新解释为目标数据类型
    vectors = np.frombuffer(vec_data.tobytes(), dtype=dtype).reshape(n, dim)
    
    if show_progress:
        print(f"  ✅ 完成！读取了 {n:,} 个 {dim} 维向量")
    
    return vectors

def read_vectors(filename, c_contiguous=True):
    return read_vecs_fast(filename, c_contiguous)
    
def process_batch(args):
    batch, centroids, k, metric = args
    if metric == 'l2':
        scores = faiss.pairwise_distances(batch, centroids, metric=faiss.METRIC_L2)
        top_clusters = np.argsort(scores, axis=1)[:, :k]
    elif metric == 'ip':
        scores = faiss.pairwise_distances(batch, centroids, metric=faiss.METRIC_INNER_PRODUCT)
        top_clusters = np.argsort(-scores, axis=1)[:, :k]
    else:
        raise ValueError(f"Unsupported metric: {metric}")
    
    # Convert to bytes
    result = bytearray()
    for cluster_ids in top_clusters:
        result.extend(struct.pack('i', k))  # Write dimension
        result.extend(struct.pack(f'{k}i', *cluster_ids.astype(np.int32)))
    return result

def compute_and_save_top_clusters(X, centroids_path, output_path, k, batch_size=10000, metric='l2'):
    print("Loading centroids...")
    centroids = read_vectors(centroids_path)
    
    print("Computing and saving top clusters...")
    num_workers = cpu_count()
    print(f"Using {num_workers} workers")
    
    # Prepare batches
    batches = []
    for i in range(0, X.shape[0], batch_size):
        end = min(i + batch_size, X.shape[0])
        batches.append((X[i:end], centroids, k, metric))
    
    # Process batches in parallel with progress bar
    with Pool(num_workers) as pool:
        results = list(tqdm(pool.imap(process_batch, batches), total=len(batches)))
    
    # Write results to file
    print("Writing results to file...")
    with open(output_path, 'wb') as f:
        for result in tqdm(results):
            f.write(result)
                
    print(f"Top clusters saved to: {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="计算向量的 top-k 聚类编号")
    parser.add_argument('--metric', choices=['l2', 'ip'], default='l2',
                        help='选择距离度量，l2 为欧氏距离，ip 为 inner product')
    args = parser.parse_args()

    metric = args.metric

    # Parameters
    source = '/data/vector_datasets/'
    datasets = ['openai1536', 'openai3072', 'glove2m_normalized', 'word2vec_normalized']
    K = 1024  # Total number of clusters
    batch_size = 2000
    k = 512  # Number of top clusters to keep

    for dataset in datasets:
        print(f"\n=== Processing dataset: {dataset} ===")
        path = os.path.join(source, dataset)
        
        # Auto-detect file format (.fvecs or .bvecs)
        data_path_fvecs = os.path.join(path, f'{dataset}_base.fvecs')
        data_path = data_path_fvecs
        
        centroids_path_fvecs = os.path.join(path, f'{dataset}_centroid_{K}.fvecs')
        centroids_path = centroids_path_fvecs
        
        output_path = os.path.join(path, f'{dataset}_top_clusters_{k}_of_{K}.ivecs')

        print(f"Data file: {data_path}")
        print(f"Centroids file: {centroids_path}")

        # clear the output file
        if os.path.exists(output_path):
            os.remove(output_path)

        # Load base vectors
        X = read_vectors(data_path)
        
        # Compute and save top clusters
        compute_and_save_top_clusters(X, centroids_path, output_path, k, batch_size, metric)