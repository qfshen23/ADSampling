import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
import os
from tqdm import tqdm

def read_fvecs(filename, c_contiguous=True):
    """Load fvecs format file"""
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
    """Load ivecs format file"""
    iv = np.fromfile(filename, dtype=np.int32)
    if iv.size == 0:
        return np.zeros((0, 0))
    dim = iv[0]
    assert dim > 0
    iv = iv.reshape(-1, 1 + dim)
    if not all(iv[:, 0] == dim):
        raise IOError("Non-uniform vector sizes in " + filename)
    iv = iv[:, 1:]
    return iv

# Load data
print("Loading data...")
K = 1024
dataset_path = '/data/vector_datasets/sift'

base_vectors = read_fvecs(os.path.join(dataset_path, 'sift_base.fvecs'))
centroids = read_fvecs(os.path.join(dataset_path, f'sift_centroid_{K}.fvecs'))
query_vectors = read_fvecs(os.path.join(dataset_path, 'sift_query.fvecs'))[:100]  # Only first 100 queries
gt = read_ivecs(os.path.join(dataset_path, 'sift_groundtruth_10000.ivecs'))[:100]  # Only first 100 queries
cluster_ids = read_ivecs(os.path.join(dataset_path, f'sift_cluster_id_{K}.ivecs'))

n_queries = query_vectors.shape[0]  # Will be 100
top_k_clusters_list = [4, 8, 16, 32, 64, 128]
n_neighbors = gt.shape[1]  # Number of ground truth neighbors

# Pre-compute distances between centroids and query vectors
print("Pre-computing query-centroid distances...")
query_centroid_distances = np.linalg.norm(query_vectors[:, np.newaxis] - centroids, axis=2)
query_top_clusters_dict = {}
for top_k in top_k_clusters_list:
    query_top_clusters_dict[top_k] = [set(clusters) for clusters in np.argsort(query_centroid_distances)[:, :top_k]]

# Pre-compute distances between centroids and base vectors for ground truth points
print("Pre-computing base-centroid distances...")
unique_gt_points = np.unique(gt)
base_centroid_distances = {}
for nn in unique_gt_points:
    nn_vector = base_vectors[nn]
    distances = np.linalg.norm(centroids - nn_vector, axis=1)
    sorted_indices = np.argsort(distances)
    base_centroid_distances[nn] = {top_k: set(sorted_indices[:top_k]) for top_k in top_k_clusters_list}

# For storing overlap statistics for each neighbor and each top-k
nn_overlaps = {top_k: [[] for _ in range(n_neighbors)] for top_k in top_k_clusters_list}

print(f"Processing {n_queries} queries...")
for query_idx in tqdm(range(n_queries)):
    gt_points = gt[query_idx]
    
    for top_k in top_k_clusters_list:
        query_clusters = query_top_clusters_dict[top_k][query_idx]
        
        # Process all ground truth neighbors
        for nn_idx, nn in enumerate(gt_points):
            nn_top_clusters = base_centroid_distances[nn][top_k]
            overlap = len(query_clusters.intersection(nn_top_clusters))
            nn_overlaps[top_k][nn_idx].append(overlap / top_k)

# Calculate and plot average overlaps for each neighbor and top-k
plt.figure(figsize=(12, 8))
gt_indices = np.arange(1, n_neighbors + 1)

for top_k in top_k_clusters_list:
    print(f"\nResults for top-{top_k} clusters:")
    avg_nn_overlaps = [np.mean(overlaps) for overlaps in nn_overlaps[top_k]]
    for i, avg_overlap in enumerate(avg_nn_overlaps):
        print(f"Average overlap with {i+1}-th NN clusters: {avg_overlap:.4f}")

    # Plot line for each top_k value
    plt.plot(gt_indices, avg_nn_overlaps, marker='o', label=f'Top-{top_k}')

plt.xlabel('Ground Truth Neighbor Index')
plt.ylabel('Average Overlap Ratio')
plt.title('Average Cluster Overlap Ratios vs Ground Truth Neighbor Index')
plt.grid(True, alpha=0.3)
plt.legend()
plt.savefig('sift_cluster_overlap_distribution_10K.png', dpi=600, bbox_inches='tight')
plt.close()
