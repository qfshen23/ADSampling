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

def compute_and_save_cluster_ids(X, centroids_path, ids_path, batch_size):
    print("Loading centroids...")
    centroids = read_fvecs(centroids_path)
    
    print("Computing cluster assignments...")
    with open(ids_path, 'wb') as f:
        for i in range(0, X.shape[0], batch_size):
            end = min(i + batch_size, X.shape[0])
            batch = X[i:end]
            
            # Compute distances to all centroids
            distances = faiss.pairwise_distances(batch, centroids)
            
            # Get nearest centroid indices
            cluster_ids = np.argmin(distances, axis=1)
            
            # Save cluster IDs
            for cluster_id in cluster_ids:
                f.write(struct.pack('i', 1))  # Write dimension (always 1)
                f.write(struct.pack('i', int(cluster_id)))  # Write cluster ID as int32
                
    print(f"Cluster IDs saved to: {ids_path}")

if __name__ == '__main__':
    for dataset in datasets:
        print(f"\n=== Processing dataset: {dataset} ===")
        path = os.path.join(source, dataset)
        data_path = os.path.join(path, f'{dataset}_base.fvecs')
        centroids_path = os.path.join(path, f'{dataset}_centroid_{K}.fvecs')
        ids_path = os.path.join(path, f'{dataset}_cluster_id_{K}.ivecs')

        # Load base vectors
        X = read_fvecs(data_path)
        
        # Compute and save cluster assignments
        compute_and_save_cluster_ids(X, centroids_path, ids_path, batch_size)