import numpy as np
import faiss
import struct
import os

source = '/data/vector_datasets/'
datasets = ['glove2m']
# the number of clusters
K = 1024

def read_fvecs(filename, c_contiguous=True):
    fv = np.fromfile(filename, dtype=np.float32)
    if fv.size == 0:
        return np.zeros((0, 0), dtype=np.float32)
    dim = fv.view(np.int32)[0]
    assert dim > 0
    fv = fv.reshape(-1, 1 + dim)
    if not all(fv.view(np.int32)[:, 0] == dim):
        raise IOError("Non-uniform vector sizes in " + filename)
    fv = fv[:, 1:]
    if c_contiguous:
        fv = fv.copy()
    return fv

if __name__ == '__main__':

    for dataset in datasets:
        print(f"Processing dataset: {dataset}")

        # 路径
        path = os.path.join(source, dataset)
        data_path = os.path.join(path, f'{dataset}_base.fvecs')
        centroids_path = os.path.join(path, f'{dataset}_centroid_{K}.fvecs')
        cluster_size_path = os.path.join(path, f'{dataset}_cluster_sizes_{K}.txt')

        # 读取 base 向量 & centroids
        print(f"Reading base vectors from: {data_path}")
        X = read_fvecs(data_path)
        print(f"Reading centroids from: {centroids_path}")
        C = read_fvecs(centroids_path)

        if X.shape[0] == 0:
            print("Warning: no base vectors found, skip dataset.")
            continue
        if C.shape[0] != K:
            raise ValueError(f"Expected {K} centroids, but got {C.shape[0]}")

        D = X.shape[1]
        assert C.shape[1] == D, "Dimension mismatch between base vectors and centroids"

        print(f"Base vectors: {X.shape[0]}, dim={D}")
        print(f"Centroids: {C.shape[0]}, dim={C.shape[1]}")

        # 用 L2 距离把每个 base vector 分配到最近的 centroid
        print("Building FAISS index on centroids...")
        index = faiss.IndexFlatL2(D)  # 只在 centroids 上建一个简单的 Flat L2 index
        index.add(C)                  # 添加所有 centroids

        print("Assigning base vectors to nearest centroids...")
        # I: (nb, 1) 每个 base vector 对应的 cluster id
        # Dists: (nb, 1) 对应的最小距离（这里其实用不到）
        Dists, I = index.search(X, 1)
        I = I.reshape(-1)  # 转成一维，长度 = num_base_vectors

        # 用 numpy.bincount 统计每个 cluster 的 size
        print("Counting vectors in each cluster...")
        # minlength=K 确保即使某个 cluster 没有向量也会有一个 0 计数
        cluster_sizes = np.bincount(I, minlength=K)

        # 按 cluster size 从大到小排序
        print("Sorting clusters by size (descending)...")
        # indices_sorted 是 0..K-1 的一个排列
        indices_sorted = np.argsort(-cluster_sizes)  # 负号表示从大到小

        # 输出到文本文件：每行 "cluster_id size"
        print(f"Writing cluster sizes to: {cluster_size_path}")
        with open(cluster_size_path, 'w') as f:
            f.write("# cluster_id size\n")
            for cid in indices_sorted:
                f.write(f"{cid} {int(cluster_sizes[cid])}\n")

        print("Done.\n")
