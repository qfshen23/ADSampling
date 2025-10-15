import numpy as np
import faiss
import struct
import os

source = '/data/vector_datasets/'
datasets = ['spacev10m', 'bigann10m', 'deep10m']
# the number of clusters
K = 2048

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

def read_vectors(filename, c_contiguous=True):
    return read_fvecs(filename, c_contiguous)

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
        print(f"Clustering - {dataset}")
        # path
        path = os.path.join(source, dataset)
        
        # 自动检测数据文件格式 (.fvecs 或 .bvecs)
        data_path_fvecs = os.path.join(path, f'{dataset}_base.fvecs')
        data_path = data_path_fvecs
        
        centroids_path = os.path.join(path, f'{dataset}_centroid_{K}.fvecs')
        randomzized_cluster_path = os.path.join(path, f"{dataset}_centroid_{K}.fvecs")
        # transformation_path = os.path.join(path, 'O.fvecs')

        # read data vectors (自动选择读取函数)
        X = read_vectors(data_path)
        # P = read_fvecs(transformation_path)
        # C = read_fvecs(centroids_path)
        D = X.shape[1]
        
        
        # cluster data vectors
        index = faiss.index_factory(D, f"IVF{K},Flat")
        index.verbose = True
        index.train(X)
        centroids = index.quantizer.reconstruct_n(0, index.nlist)
        to_fvecs(centroids_path, centroids)

        # randomized centroids
        # centroids = np.dot(C, P)
        # to_fvecs(randomzized_cluster_path, centroids)
