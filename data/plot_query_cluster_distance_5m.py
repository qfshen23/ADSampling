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

K = 2048
datasets = ['tiny5m'] 
data_path = '/data/vector_datasets'
top_k_clusters = 2048

plt.figure(figsize=(10, 6))

for dataset in datasets:
    dataset_path = os.path.join(data_path, dataset)
    centroid_file = os.path.join(dataset_path, f'{dataset}_centroid_{K}.fvecs')
    query_file = os.path.join(dataset_path, f"{dataset}_query.fvecs")

    centroids = read_fvecs(centroid_file)
    queries = read_fvecs(query_file)

    n_queries = queries.shape[0]
    n_clusters = K

    # For each query, calculate distances to all clusters and sort them
    avg_distances = np.zeros(top_k_clusters)
    
    for query in queries:
        # Calculate distances from query to all centroids
        distances = np.linalg.norm(centroids - query, axis=1)
        # Sort distances and get top_k_clusters nearest
        sorted_distances = np.sort(distances)[:top_k_clusters]
        avg_distances += sorted_distances

    # Calculate average over all queries
    avg_distances /= n_queries

    # Plot average distances
    plt.plot(range(1, top_k_clusters + 1), avg_distances, label=dataset, marker='o', markersize=3, alpha=0.7)

plt.title(f'Average Query-Cluster Distances for {top_k_clusters} Nearest Clusters')
plt.xlabel('k-th Nearest Cluster')
plt.ylabel('Average Distance')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

os.makedirs('query-cluster-distance', exist_ok=True)
plt.savefig(f'query-cluster-distance/distances_tiny5m.png', dpi=600, bbox_inches='tight')
plt.close()
