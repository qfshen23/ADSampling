import numpy as np
import matplotlib.pyplot as plt
import os
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

K = [64, 256, 512, 1024]

source = '/data/vector_datasets'
datasets = ['sift']
layer1_path = "../sift_layer1_vertices.txt"

with open(layer1_path, 'r') as f:
    layer1_ids = [int(line.strip()) for line in f if line.strip()]

for dataset in datasets:
    for k in K:
        print(f"Processing dataset - {dataset}")
        path = os.path.join(source, dataset)
        cluster_ids_path = os.path.join(path, f'{dataset}_cluster_id_{k}.ivecs')
        cluster_ids = load_ivecs(cluster_ids_path).flatten()
        
        # Original histogram
        hist = np.zeros(k, dtype=int)
        for vid in layer1_ids:
            cid = cluster_ids[vid]
            hist[cid] += 1
            
        # For large K, merge clusters into fewer groups
        if k >= 512:  # Consider 512 and 1024 as large K
            num_merged_clusters = 64  # Merge into 64 groups
            merged_hist = np.zeros(num_merged_clusters, dtype=int)
            merge_factor = k // num_merged_clusters
            
            for i in range(k):
                merged_cluster_id = i // merge_factor
                if merged_cluster_id < num_merged_clusters:  # Ensure we don't go out of bounds
                    merged_hist[merged_cluster_id] += hist[i]
            
            # Plot both original and merged histograms
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # Original histogram
            ax1.bar(np.arange(k), hist)
            ax1.set_xlabel("Cluster ID")
            ax1.set_ylabel("# of Layer-1 Vertices")
            ax1.set_title(f"Layer-1 Vertex Distribution over {k} Clusters in {dataset}")
            
            # Merged histogram
            ax2.bar(np.arange(num_merged_clusters), merged_hist)
            ax2.set_xlabel(f"Merged Cluster ID (each contains ~{merge_factor} original clusters)")
            ax2.set_ylabel("# of Layer-1 Vertices")
            ax2.set_title(f"Layer-1 Vertex Distribution over {num_merged_clusters} Merged Clusters in {dataset}")
            
            plt.tight_layout()
            plt.savefig(f"{dataset}_layer1_vertices_cluster_{k}_with_merged.png", dpi=400)
        else:
            # For smaller K, just plot the original histogram
            plt.figure(figsize=(10, 5))
            plt.bar(np.arange(k), hist)
            plt.xlabel("Cluster ID")
            plt.ylabel("# of Layer-1 Vertices")
            plt.title(f"Layer-1 Vertex Distribution over {k} Clusters in {dataset}")
            plt.tight_layout()
            plt.savefig(f"{dataset}_layer1_vertices_cluster_{k}.png", dpi=400)
