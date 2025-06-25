import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
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
    iv = iv[:, 1:]
    return iv
    
K = 1024
datasets = ['sift']
data_path = '/data/vector_datasets'
top_k_clusters = 100
k_values = [1, 10, 50]  # k nearest neighbors to consider (out of 10000 available)

for dataset in datasets:
    dataset_path = os.path.join(data_path, dataset)
    base_file = os.path.join(dataset_path, f"{dataset}_base.fvecs")
    centroid_file = os.path.join(dataset_path, f'{dataset}_centroid_{K}.fvecs')
    query_file = os.path.join(dataset_path, f"{dataset}_query.fvecs")
    gt_file = os.path.join(dataset_path, f'{dataset}_groundtruth_10000.ivecs')
    cluster_ids_file = os.path.join(dataset_path, f'{dataset}_cluster_id_{K}.ivecs')

    base_vectors = read_fvecs(base_file)
    centroids_vectors = read_fvecs(centroid_file)
    query_vectors = read_fvecs(query_file)
    point_to_cluster = read_ivecs(cluster_ids_file)
    all_queries_nn = read_ivecs(gt_file)  # Load all ground truth neighbors

    n_queries = query_vectors.shape[0]  # Number of queries
    n_points = base_vectors.shape[0]  # Dataset size
    n_clusters = K

    print(f"Dataset: {dataset}")
    print(f"n_queries: {n_queries}, n_points: {n_points}, n_clusters: {n_clusters}")

    queries_top_clusters = np.zeros((n_queries, top_k_clusters), dtype=int)

    for i, query in enumerate(query_vectors):
        distances = np.linalg.norm(centroids_vectors - query, axis=1)
        top_k = np.argsort(distances)[:top_k_clusters]
        queries_top_clusters[i] = top_k

    for k in k_values:
        # Get k nearest neighbors for current k value
        queries_nn = all_queries_nn[:,:k]

        # Initialize histogram array
        histogram = np.zeros(top_k_clusters, dtype=float)

        # Count distribution of k-NN in Top-k clusters
        for query_idx in range(n_queries):
            nn_indices = queries_nn[query_idx]
            # Cluster IDs of these NNs
            nn_clusters = point_to_cluster[nn_indices]
            # Top-k clusters for current query
            top_clusters = queries_top_clusters[query_idx]
            
            # For each NN, determine which Top-k cluster it belongs to
            for i, cluster_id in enumerate(top_clusters):
                # Count how many points from k-NN belong to current cluster
                count_in_cluster = np.sum(nn_clusters == cluster_id)
                # Calculate weight
                histogram[i] += count_in_cluster

        # Calculate percentage for each cluster
        percentage = histogram / n_queries

        # Calculate cumulative percentage
        cumulative_percentage = np.cumsum(percentage)

        # Print max cumulative value for k=10000
        if k == 10000:
            print(f"Max cumulative value for {dataset} at k=10000: {cumulative_percentage[-1]:.2f}")

        # Create figure and axis with two y-axes
        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax2 = ax1.twinx()

        # Plot histogram on first y-axis
        bars = ax1.bar(range(1, top_k_clusters + 1), percentage, color='skyblue', alpha=0.7, label='Hit Count')
        ax1.set_xlabel('Top-kth Clusters')
        ax1.set_ylabel('Average Hit Count', color='skyblue')
        ax1.tick_params(axis='y', labelcolor='skyblue')
        ax1.set_xticks(range(1, top_k_clusters + 1, 10))
        ax1.grid(axis='y', linestyle='--', alpha=0.7)

        # Plot cumulative line on second y-axis
        line = ax2.plot(range(1, top_k_clusters + 1), cumulative_percentage, color='red', alpha=0.7, label='Cumulative Count')
        ax2.set_ylabel('Cumulative Average Hit Count', color='red')
        ax2.tick_params(axis='y', labelcolor='red')

        # Add title
        plt.title(f'Distribution and Cumulative Count of {k}-NN\nin Top-{top_k_clusters} Clusters for {dataset}')

        # Add legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

        plt.tight_layout()
        os.makedirs('ivf-hit-count-combined', exist_ok=True)
        plt.savefig(f'ivf-hit-count-combined/combined-{dataset}-{k}-c{K}.png', dpi=600)
        plt.close()
