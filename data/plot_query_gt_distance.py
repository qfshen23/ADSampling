import numpy as np
import matplotlib.pyplot as plt
import os

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

def read_ivecs(filename):
    iv = np.fromfile(filename, dtype=np.int32)
    if iv.size == 0:
        return np.zeros((0, 0))
    dim = iv[0]
    assert dim > 0
    iv = iv.reshape(-1, 1 + dim)
    if not all(iv[:, 0] == dim):
        raise IOError("Non-uniform vector sizes in " + filename)
    return iv[:, 1:]

data_path = '/data/vector_datasets'

# Configuration for different datasets
datasets = ['gist', 'sift', 'msong', 'tiny5m', 'sift10m']

os.makedirs('query-gt-distance', exist_ok=True)

for dataset in datasets:
    dataset_path = os.path.join(data_path, dataset)
    base_file = os.path.join(dataset_path, f"{dataset}_base.fvecs")
    query_file = os.path.join(dataset_path, f"{dataset}_query.fvecs")
    gt_file = os.path.join(dataset_path, f"{dataset}_groundtruth_10000.ivecs")
    
    # Check if all files exist
    if not all(os.path.exists(f) for f in [base_file, query_file, gt_file]):
        print(f"Skipping {dataset}: missing files")
        continue
    
    print(f"Processing {dataset}...")
    
    base_vectors = read_fvecs(base_file)
    queries = read_fvecs(query_file)
    groundtruth = read_ivecs(gt_file)
    
    n_queries = queries.shape[0]
    k_gt = groundtruth.shape[1]  # Number of ground truth neighbors
    
    # For each query, calculate distances to its ground truth neighbors
    avg_distances = np.zeros(k_gt)
    
    for i in range(n_queries):
        query = queries[i]
        gt_neighbors = groundtruth[i]
        
        # Calculate distances from query to each ground truth neighbor
        distances = []
        for j, neighbor_idx in enumerate(gt_neighbors):
            if neighbor_idx < base_vectors.shape[0]:  # Valid neighbor index
                distance = np.linalg.norm(base_vectors[neighbor_idx] - query)
                distances.append(distance)
            else:
                distances.append(float('inf'))  # Invalid neighbor
        
        avg_distances += np.array(distances)
    
    # Calculate average over all queries
    avg_distances /= n_queries
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, k_gt + 1), avg_distances, marker='o', markersize=3, alpha=0.7)
    plt.title(f'Average Query-GT Neighbor Distances for {dataset}')
    plt.xlabel('k-th Ground Truth Neighbor')
    plt.ylabel('Average Distance')
    # plt.yscale('log')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.savefig(f'query-gt-distance/{dataset}_gt_distances_linear.png', dpi=600, bbox_inches='tight')
    plt.close()
    
    print(f"Saved plot for {dataset}")
