import numpy as np
import faiss
import struct
import os

source = '/data/vector_datasets/'
datasets = ['sift10m']
K = 4096
batch_size = 10000

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

def to_fvecs(filename, data):
    print(f"Writing File - {filename}")
    num, dim = data.shape
    with open(filename, 'wb') as f:
        for i in range(num):
            f.write(struct.pack('i', dim))  # 写入维度
            f.write(data[i].astype(np.float32, copy=False).tobytes())  # 写入向量

def extract_and_save_centroids(X, D, K, centroids_path):
    print("Training IVF index...")
    index = faiss.index_factory(D, f"IVF{K},Flat")
    index.verbose = True
    index.train(X)
    centroids = index.quantizer.reconstruct_n(0, index.nlist)
    to_fvecs(centroids_path, centroids)
    print(f"Centroids saved to: {centroids_path}")

def compute_and_save_distances(X, centroids_path, distances_path, batch_size):
    print("Loading centroids...")
    centroids = read_fvecs(centroids_path)
    K = centroids.shape[0]
    with open(distances_path, 'wb') as f:
        for i in range(0, X.shape[0], batch_size):
            end = min(i + batch_size, X.shape[0])
            D_batch = faiss.pairwise_distances(X[i:end], centroids)
            for row in D_batch:
                f.write(struct.pack('i', K))
                f.write(row.astype(np.float32, copy=False).tobytes())
    print(f"Distances saved to: {distances_path}")

if __name__ == '__main__':
    for dataset in datasets:
        print(f"\n=== Processing dataset: {dataset} ===")
        path = os.path.join(source, dataset)
        data_path = os.path.join(path, f'{dataset}_base.fvecs')
        centroids_path = os.path.join(path, f'{dataset}_centroid_{K}.fvecs')
        distances_path = os.path.join(path, f'{dataset}_distances_{K}.fvecs')

        # Step 1: Load base vectors
        X = read_fvecs(data_path)
        D = X.shape[1]

        # Step 2: Extract and save centroids
        extract_and_save_centroids(X, D, K, centroids_path)

        # Step 3: Compute and save distances (in low memory)
        compute_and_save_distances(X, centroids_path, distances_path, batch_size)
