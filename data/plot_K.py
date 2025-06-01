import numpy as np
import faiss
from tqdm import tqdm
import os
source = '/data/vector_datasets/'
datasets = ['tiny5m', 'sift10m']

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

def knn_indegree_faiss(X, K=100, batch_size=10000, verbose=1):
    N, D = X.shape
    index = faiss.IndexFlatL2(D)
    index.add(X)
    indegree = np.zeros(N, dtype=np.int32)

    for start in tqdm(range(0, N, batch_size), desc="kNN Search", disable=not verbose):
        end = min(start + batch_size, N)
        Dists, Ids = index.search(X[start:end], K + 1)
        for i, row in enumerate(Ids):
            filtered = row[row != (start + i)]
            filtered = filtered[:K]
            indegree[filtered] += 1

    return indegree

if __name__ == "__main__":
    for dataset in datasets:
        print(f"Processing dataset - {dataset}")
        # Paths
        path = os.path.join(source, dataset)
        base_path = os.path.join(path, f'{dataset}_base.fvecs')
        base = read_fvecs(base_path)
        print("Loaded data of shape:", base.shape)
        K = 100
        indegree = knn_indegree_faiss(base, K=K, batch_size=100000) 
        np.save(f"{dataset}_faiss_knn_indegree.npy", indegree)
