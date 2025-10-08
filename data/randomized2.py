import os
import numpy as np
import struct

source = '/data/vector_datasets/'
datasets = ['tiny5m']

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

if __name__ == "__main__":
    
    for dataset in datasets:
        # path
        path = os.path.join(source, dataset)
        
        # read O.fvecs (orthogonal transformation matrix)
        O_path = os.path.join(path, 'O.fvecs')
        print(f"Reading O.fvecs from {O_path}")
        O = read_fvecs(O_path)
        
        # read centroids
        centroids_path = os.path.join(path, f'{dataset}_centroid_2048.fvecs')
        print(f"Reading centroids from {centroids_path}")
        centroids = read_fvecs(centroids_path)
        
        # apply transformation to centroids
        print(f"Transforming centroids for {dataset}")
        O_centroids = np.dot(centroids, O)
        
        # save transformed centroids
        O_centroids_path = os.path.join(path, f'O{dataset}_centroid_2048.fvecs')
        to_fvecs(O_centroids_path, O_centroids)
        
        print(f"Saved transformed centroids to {O_centroids_path}")
