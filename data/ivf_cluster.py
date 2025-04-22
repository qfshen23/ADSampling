import numpy as np
import faiss
import struct
import os

source = '/data/vector_datasets/'
datasets = ['gist', 'sift']
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
        print(f"Processing dataset - {dataset}")
        # Paths
        path = os.path.join(source, dataset)
        data_path = os.path.join(path, f'{dataset}_base.fvecs')
        centroids_path = os.path.join(path, f'{dataset}_centroid_{K}.fvecs')
        distances_path = os.path.join(path, f'{dataset}_distances_{K}.fvecs')

        # Read data and centroids
        X = read_fvecs(data_path)
        D = X.shape[1]

        # Train index and extract centroids
        index = faiss.index_factory(D, f"IVF{K},Flat")
        index.verbose = True
        index.train(X)
        centroids = index.quantizer.reconstruct_n(0, index.nlist)
        to_fvecs(centroids_path, centroids)

        # Compute distances from each base vector to all centroids
        distances = np.empty((X.shape[0], K), dtype=np.float32)
        batch_size = 10000  # avoid memory overflow, process in batches
        for i in range(0, X.shape[0], batch_size):
            end = min(i + batch_size, X.shape[0])
            distances[i:end] = faiss.pairwise_distances(X[i:end], centroids)

        # Save distances
        to_fvecs(distances_path, distances)
