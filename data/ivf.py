import numpy as np
import faiss
import struct
import os

# ============ 配置参数 ============
source = '/data/vector_datasets/'
datasets = ['glove2m_normalized', 'word2vec_normalized']

# 聚类数量
K = 1024

# 距离度量
# 'L2': 欧氏距离（L2 distance）
# 'IP': 内积（Inner Product，对于归一化向量等同于余弦相似度）
metric = 'IP'
# ==================================

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

def to_fvecs(filename, data):
    print(f"Writing File - {filename}")
    with open(filename, 'wb') as fp:
        for y in data:
            d = struct.pack('I', len(y))
            fp.write(d)
            for x in y:
                a = struct.pack('f', x)
                fp.write(a)

if __name__ == '__main__':

    for dataset in datasets:
        print("=" * 80)
        print(f"Clustering - {dataset}")
        print(f"Metric: {metric}, K: {K}")
        print("=" * 80)
        
        # path
        path = os.path.join(source, dataset)
        
        # 自动检测数据文件格式 (.fvecs 或 .bvecs)
        data_path_fvecs = os.path.join(path, f'{dataset}_base.fvecs')
        data_path = data_path_fvecs
        
        # 根据 metric 命名聚类中心文件
        centroids_path = os.path.join(path, f'{dataset}_centroid_{K}_{metric}.fvecs')
        # transformation_path = os.path.join(path, 'O.fvecs')

        # read data vectors (自动选择读取函数)
        X = read_vectors(data_path)
        # P = read_fvecs(transformation_path)
        # C = read_fvecs(centroids_path)
        D = X.shape[1]
        
        print(f"数据维度: {D}, 向量数量: {X.shape[0]:,}")
        
        # 根据 metric 创建不同的索引
        if metric == 'L2':
            # L2 距离（欧氏距离）
            print(f"使用 L2 (欧氏距离) 进行聚类")
            quantizer = faiss.IndexFlatL2(D)
            index = faiss.IndexIVFFlat(quantizer, D, K, faiss.METRIC_L2)
        elif metric == 'IP':
            # 内积距离（对于归一化向量等同于余弦相似度）
            print(f"使用 IP (内积/余弦相似度) 进行聚类")
            quantizer = faiss.IndexFlatIP(D)
            index = faiss.IndexIVFFlat(quantizer, D, K, faiss.METRIC_INNER_PRODUCT)
        else:
            raise ValueError(f"不支持的 metric: {metric}，请使用 'L2' 或 'IP'")
        
        index.verbose = True
        print(f"开始训练聚类...")
        index.train(X)
        print(f"聚类训练完成！")
        
        # 提取聚类中心
        centroids = index.quantizer.reconstruct_n(0, index.nlist)
        print(f"提取了 {centroids.shape[0]} 个聚类中心")
        
        # 保存聚类中心
        to_fvecs(centroids_path, centroids)
        print(f"✓ 聚类中心已保存到: {centroids_path}")
        print()

        # randomized centroids
        # centroids = np.dot(C, P)
        # to_fvecs(randomzized_cluster_path, centroids)
