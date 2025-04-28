import numpy as np
import faiss
import struct
import os

source = '/data/vector_datasets/'
datasets = ['sift']
K = 64

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

def to_ivecs(filename, data):
    print(f"Writing File - {filename}")
    with open(filename, 'wb') as fp:
        for y in data:
            d = struct.pack('I', 1)  # only 1 int per entry (cluster id)
            fp.write(d)
            a = struct.pack('i', y)
            fp.write(a)

if __name__ == '__main__':
    for dataset in datasets:
        print(f"Processing dataset - {dataset}")
        # Paths
        path = os.path.join(source, dataset)
        data_path = os.path.join(path, f'{dataset}_base.fvecs')
        centroids_path = os.path.join(path, f'{dataset}_centroid_{K}.fvecs')
        cluster_ids_path = os.path.join(path, f'{dataset}_cluster_id_{K}.ivecs')

        # Read data
        X = read_fvecs(data_path)
        D = X.shape[1]

        # Train index and extract centroids
        index = faiss.index_factory(D, f"IVF{K},Flat")
        index.verbose = True
        index.train(X)
        centroids = index.quantizer.reconstruct_n(0, index.nlist)

        # Assign each point to nearest centroid
        batch_size = 10000
        cluster_ids = np.empty(X.shape[0], dtype=np.int32)
        for i in range(0, X.shape[0], batch_size):
            end = min(i + batch_size, X.shape[0])
            D_batch = faiss.pairwise_distances(X[i:end], centroids)
            cluster_ids[i:end] = np.argmin(D_batch, axis=1)

        # Save cluster ids
        to_ivecs(cluster_ids_path, cluster_ids)
