import numpy as np
import faiss
import struct
import os
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

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

def process_batch(args):
    batch, centroids, k = args
    # Compute distances to all centroids
    distances = faiss.pairwise_distances(batch, centroids)
    # Sort cluster IDs by distance and take top k
    top_clusters = np.argsort(distances, axis=1)[:,:k]
    
    # Convert to bytes
    result = bytearray()
    for cluster_ids in top_clusters:
        result.extend(struct.pack('i', k))  # Write dimension
        result.extend(struct.pack(f'{k}i', *cluster_ids.astype(np.int32)))
    return result

def compute_and_save_top_clusters(X, centroids_path, output_path, batch_size=10000):
    print("Loading centroids...")
    centroids = read_fvecs(centroids_path)
    
    print("Computing and saving top clusters...")
    num_workers = cpu_count()
    print(f"Using {num_workers} workers")
    
    # Prepare batches
    batches = []
    for i in range(0, X.shape[0], batch_size):
        end = min(i + batch_size, X.shape[0])
        batches.append((X[i:end], centroids, k))
    
    # Process batches in parallel with progress bar
    with Pool(num_workers) as pool:
        results = list(tqdm(pool.imap(process_batch, batches), total=len(batches)))
    
    # Write results to file
    print("Writing results to file...")
    with open(output_path, 'wb') as f:
        for result in tqdm(results):
            f.write(result)
                
    print(f"Top clusters saved to: {output_path}")

if __name__ == '__main__':
    # Parameters
    source = '/data/vector_datasets/'
    datasets = ['sift']
    K = 256  # Total number of clusters
    batch_size = 2000
    k = 256  # Number of top clusters to keep

    for dataset in datasets:
        print(f"\n=== Processing dataset: {dataset} ===")
        path = os.path.join(source, dataset)
        data_path = os.path.join(path, f'{dataset}_base.fvecs')
        centroids_path = os.path.join(path, f'{dataset}_centroid_{K}.fvecs')
        output_path = os.path.join(path, f'{dataset}_top_clusters_{k}_of_{K}.ivecs')

        # clear the output file
        if os.path.exists(output_path):
            os.remove(output_path)

        # Load base vectors
        X = read_fvecs(data_path)
        
        # Compute and save top clusters
        compute_and_save_top_clusters(X, centroids_path, output_path, batch_size)