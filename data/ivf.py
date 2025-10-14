import numpy as np
import faiss
import struct
import os

source = '/data/vector_datasets/'
datasets = ['spacev10m', 'bigann10m']
# the number of clusters
K = 4096

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

def read_bvecs(filename, c_contiguous=True):
    bv = np.fromfile(filename, dtype=np.uint8)
    if bv.size == 0:
        return np.zeros((0, 0))
    dim = bv[:4].view(np.int32)[0]
    assert dim > 0
    bv = bv.reshape(-1, 4 + dim)
    if not all(bv[:, :4].view(np.int32).flatten() == dim):
        raise IOError("Non-uniform vector sizes in " + filename)
    bv = bv[:, 4:]
    if c_contiguous:
        bv = bv.copy()
    return bv.astype(np.float32)

def read_vectors(filename, c_contiguous=True):
    """自动根据文件扩展名选择读取函数"""
    if filename.endswith('.fvecs'):
        return read_fvecs(filename, c_contiguous)
    elif filename.endswith('.bvecs'):
        return read_bvecs(filename, c_contiguous)
    else:
        raise ValueError(f"不支持的文件格式: {filename}，仅支持 .fvecs 和 .bvecs")

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
        data_path_bvecs = os.path.join(path, f'{dataset}_base.bvecs')
        
        if os.path.exists(data_path_fvecs):
            data_path = data_path_fvecs
            print(f"使用 .fvecs 格式: {data_path}")
        elif os.path.exists(data_path_bvecs):
            data_path = data_path_bvecs
            print(f"使用 .bvecs 格式: {data_path}")
        else:
            raise FileNotFoundError(f"找不到数据文件: {data_path_fvecs} 或 {data_path_bvecs}")
        
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
