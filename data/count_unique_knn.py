import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import argparse
from tqdm import tqdm

source = '/data/vector_datasets'
datasets = ['sift']
K = [64, 256, 512, 1024]  # number of clusters

def load_ivecs(filename, c_contiguous=True):
    iv = np.fromfile(filename, dtype=np.int32)
    if iv.size == 0:
        return np.zeros((0, 0))
    dim = iv[0]
    assert dim > 0
    iv = iv.reshape(-1, 1 + dim)
    if not all(iv[:, 0] == dim):
        raise IOError("Non-uniform vector sizes in " + filename)
    iv = iv[:, 1:]
    if c_contiguous:
        iv = iv.copy()
    return iv

def count_unique_clusters(gt_file, cluster_file, k=100):
    gt_results = load_ivecs(gt_file)
    cluster_assignments = load_ivecs(cluster_file).flatten()
    
    print("Counting unique clusters for each query...")
    unique_clusters_counts = []
    for query_nn in tqdm(gt_results):
        # Get cluster IDs for the k nearest neighbors
        nn_clusters = [cluster_assignments[nn] for nn in query_nn]
        # Count unique clusters
        unique_count = len(set(nn_clusters))
        unique_clusters_counts.append(unique_count)
    
    # Create histogram
    max_unique = max(unique_clusters_counts)
    histogram = np.zeros(max_unique + 1, dtype=int)
    for count in unique_clusters_counts:
        histogram[count] += 1
    
    return histogram

def plot_histogram(histogram, output_file, k=100, clusters=100):
    plt.figure(figsize=(12, 6))
    x = np.arange(len(histogram))
    plt.bar(x, histogram, color='skyblue')
    plt.xlabel('Number of Unique Clusters')
    plt.ylabel('Number of Queries')
    plt.title(f'Distribution of Unique Clusters in {k}-NN (K={clusters})')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add values on top of bars
    for i, v in enumerate(histogram):
        if v > 0:  # Only show non-zero values
            plt.text(i, v + max(histogram) * 0.01, str(v), ha='center')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=400)

def main():
    for dataset in datasets:
        for k in K:
            print(f"Processing dataset - {dataset}")
            path = os.path.join(source, dataset)
            gt_path = os.path.join(path, f'{dataset}_groundtruth.ivecs')
            cluster_ids_path = os.path.join(path, f'{dataset}_cluster_id_{k}.ivecs')
            output_path = f'{dataset}_unique_clusters_histogram_{k}.png'
            histogram = count_unique_clusters(gt_path, cluster_ids_path)
            plot_histogram(histogram, output_path, 100, k)

if __name__ == "__main__":
    main()

