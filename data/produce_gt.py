import numpy as np
import faiss
import struct
import os

source = '/data/vector_datasets/'
datasets = ['gist', 'sift', 'nuswide', 'msong']
K = 10000

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
            d = struct.pack('I', len(y))
            fp.write(d)
            for x in y:
                a = struct.pack('i', x)
                fp.write(a)

if __name__ == '__main__':
    for dataset in datasets:
        print(f"Processing dataset - {dataset}")
        # Paths
        path = os.path.join(source, dataset)
        base_path = os.path.join(path, f'{dataset}_base.fvecs')
        query_path = os.path.join(path, f'{dataset}_query.fvecs')
        gt_path = os.path.join(path, f'{dataset}_groundtruth_{K}.ivecs')

        # Read data
        print("Loading base vectors...")
        base = read_fvecs(base_path)
        print(f"  base: {base.shape}")

        print("Loading query vectors...")
        query = read_fvecs(query_path)
        print(f"  query: {query.shape}")

        # Build groundtruth
        print(f"Computing {K}-NN groundtruth (L2)...")
        index = faiss.IndexFlatL2(base.shape[1])
        index.add(base)
        _, I = index.search(query, K)

        # Save groundtruth
        print(f"Writing groundtruth to {gt_path}")
        to_ivecs(gt_path, I)
        print("Done.")