import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from collections import defaultdict

source = '/data/vector_datasets'
datasets = ['sift']
K = [64]  # number of clusters

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

def load_fvecs(filename, c_contiguous=True):
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

def load_entry_ids(filename):
    entry_ids = []
    with open(filename, 'r') as f:
        for line in f:
            if line.strip():
                entry_ids.append(int(line.strip()))
    return entry_ids

def compute_distance(v1, v2):
    return np.sqrt(np.sum((v1 - v2) ** 2))

def calculate_hit_ratios(gt_file, cluster_file, entry_ids_file, centroid_file, k=100, num_nearest_clusters=20):
    gt_results = load_ivecs(gt_file)
    cluster_assignments = load_ivecs(cluster_file).flatten()
    entry_ids = load_entry_ids(entry_ids_file)
    centroids = load_fvecs(centroid_file)
    
    print("Calculating hit ratios for each entry point's k-NN...")
    hit_ratios = []
    
    for entry_id in tqdm(entry_ids):
        if entry_id < len(gt_results):
            # Get the k nearest neighbors for this entry point
            entry_nn = gt_results[entry_id]
            
            # Get the entry point's vector (assuming it's at the same index in the dataset)
            entry_cluster = cluster_assignments[entry_id]
            entry_centroid = centroids[entry_cluster]
            
            # Calculate distances from entry's centroid to all other centroids
            centroid_distances = []
            for i in range(len(centroids)):
                dist = compute_distance(entry_centroid, centroids[i])
                centroid_distances.append((dist, i))
            
            # Sort by distance and get the nearest 20 clusters
            centroid_distances.sort()
            nearest_clusters = set([cluster_id for _, cluster_id in centroid_distances[:num_nearest_clusters]])
            
            # Count how many of the k-NN are in these nearest clusters
            nn_in_nearest_clusters = 0
            for nn in entry_nn:
                if cluster_assignments[nn] in nearest_clusters:
                    nn_in_nearest_clusters += 1
            
            # Calculate hit ratio
            hit_ratio = nn_in_nearest_clusters / len(entry_nn) if len(entry_nn) > 0 else 0
            hit_ratios.append(hit_ratio)
    
    # Convert to percentages and create histogram
    hit_ratios_percent = [ratio * 100 for ratio in hit_ratios]
    
    # Create histogram with 10 bins (0-10%, 10-20%, etc.)
    histogram, bin_edges = np.histogram(hit_ratios_percent, bins=10, range=(0, 100))
    
    return histogram, bin_edges, len(entry_ids)

def plot_histogram(histogram, bin_edges, output_file, total_entries, k=100, clusters=100, num_nearest_clusters=20):
    plt.figure(figsize=(12, 6))
    
    # Plot histogram as bars
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    plt.bar(bin_centers, histogram, width=bin_edges[1] - bin_edges[0], color='skyblue', edgecolor='black')
    
    plt.xlabel('Percentage of k-NN in Nearest Clusters (%)')
    plt.ylabel('Number of Entry Points')
    plt.title(f'Hit Ratio of {k}-NN in {num_nearest_clusters} Nearest Clusters (K={clusters})')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(bin_centers)
    
    # Add values on top of bars
    for i, v in enumerate(histogram):
        if v > 0:  # Only show non-zero values
            percentage = (v / total_entries) * 100
            plt.text(bin_centers[i], v + max(histogram) * 0.01, f"{v} ({percentage:.1f}%)", ha='center')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=400)

def main():
    for dataset in datasets:
        for k in K:
            print(f"Processing dataset - {dataset}")
            path = os.path.join(source, dataset)
            gt_path = os.path.join(path, f'{dataset}_groundtruth.ivecs')
            cluster_ids_path = os.path.join(path, f'{dataset}_cluster_id_{k}.ivecs')
            entry_ids_path = f'../sift_entry_ids.txt'  # Path to entry IDs file
            centroid_path = os.path.join(path, f'{dataset}_centroid_{k}.fvecs')
            output_path = f'{dataset}_entry_hit_ratio_histogram_{k}.png'
            
            num_nearest_clusters = 20
            histogram, bin_edges, total_entries = calculate_hit_ratios(
                gt_path, cluster_ids_path, entry_ids_path, centroid_path, 100, num_nearest_clusters)
            
            plot_histogram(histogram, bin_edges, output_path, total_entries, 100, k, num_nearest_clusters)

if __name__ == "__main__":
    main()
