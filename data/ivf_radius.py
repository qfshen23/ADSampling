import numpy as np
import faiss
import struct
import os
import matplotlib.pyplot as plt

source = '/data/vector_datasets/'
datasets = ['sift10m']
K = 4096

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

def calculate_cluster_radius(data, centroids, cluster_ids):
    # Initialize array to store max radius for each cluster
    radii = np.zeros(K)
    empty_clusters = []
    
    # Reshape cluster_ids to 1D if needed
    if len(cluster_ids.shape) > 1:
        cluster_ids = cluster_ids.reshape(-1)
        
    # For each cluster, calculate radius
    for i in range(K):
        # Get vectors assigned to this cluster
        cluster_mask = (cluster_ids == i)
        cluster_vectors = data[cluster_mask]
        
        if len(cluster_vectors) > 0:
            # Calculate distances to centroid
            dists = faiss.pairwise_distances(cluster_vectors, centroids[i:i+1])
            max_dist = dists.max()
            # Handle negative distances that could cause sqrt warning
            if max_dist >= 0:
                radii[i] = np.sqrt(max_dist)
            else:
                radii[i] = 0.0
        else:
            empty_clusters.append(i)
    
    print(f"\nFound {len(empty_clusters)} empty clusters")
    if empty_clusters:
        print("Empty cluster IDs:", empty_clusters)
        
    # Print cluster size distribution
    cluster_sizes = np.bincount(cluster_ids, minlength=K)
    print("\nCluster size statistics:")
    print(f"Mean cluster size: {np.mean(cluster_sizes):.2f}")
    print(f"Max cluster size: {np.max(cluster_sizes)}")
    print(f"Min cluster size: {np.min(cluster_sizes)}")
    print(f"Number of clusters with <2 vectors: {np.sum(cluster_sizes < 2)}")
    
    return radii

def plot_cluster_radii(radii, dataset, save_path):
    plt.figure(figsize=(12, 6))
    
    # Plot radii
    plt.plot(range(K), radii, 'b-', alpha=0.6, label='Cluster Radius')
    
    # Add mean line
    mean_radius = np.mean(radii[radii > 0])  # Only consider non-zero radii in mean calculation
    plt.axhline(y=mean_radius, color='r', linestyle='--', label=f'Mean: {mean_radius:.2f}')
    
    # Add min/max annotations
    min_radius = np.min(radii)
    max_radius = np.max(radii)
    plt.annotate(f'Min: {min_radius:.2f}', 
                xy=(np.argmin(radii), min_radius),
                xytext=(10, 10), textcoords='offset points')
    plt.annotate(f'Max: {max_radius:.2f}',
                xy=(np.argmax(radii), max_radius),
                xytext=(10, -10), textcoords='offset points')
    
    plt.xlabel('Cluster ID')
    plt.ylabel('Radius')
    plt.title(f'Cluster Radii Distribution - {dataset}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    for dataset in datasets:
        print(f"\nProcessing {dataset}...")
        path = os.path.join(source, dataset)
        data_path = os.path.join(path, f'{dataset}_base.fvecs')
        centroids_path = os.path.join(path, f'{dataset}_centroid_{K}.fvecs')
        ids_path = os.path.join(path, f'{dataset}_cluster_id_{K}.ivecs')
        radius_path = os.path.join(path, f'{dataset}_radius_{K}.txt')
        plot_path = f'{dataset}_radius_{K}.png'
        
        # Load data, centroids and cluster IDs
        print("Loading data...")
        data = read_fvecs(data_path)
        print("Loading centroids...")
        centroids = read_fvecs(centroids_path)
        print("Loading cluster IDs...")
        cluster_ids = read_ivecs(ids_path)
        
        # Calculate radii
        print("Calculating cluster radii...")
        radii = calculate_cluster_radius(data, centroids, cluster_ids)
        
        # Save numerical results
        np.savetxt(radius_path, radii)
        print(f"Saved cluster radii to {radius_path}")
        
        # Plot and save figure
        print("Generating plot...")
        plot_cluster_radii(radii, dataset, plot_path)
        print(f"Saved plot to {plot_path}")
        
        # Print statistics
        print(f"\nStatistics for {dataset}:")
        print(f"Average cluster radius: {np.mean(radii):.2f}")
        print(f"Maximum cluster radius: {np.max(radii):.2f}")
        print(f"Minimum cluster radius: {np.min(radii):.2f}")

