import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import matplotlib.cm as cm
from matplotlib.colors import to_rgba
import struct
import os

source = '/data/vector_datasets'
datasets = ['sift']
K = 64  # number of clusters

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

def read_ivecs(filename, c_contiguous=True):
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

def main():
    for dataset in datasets:
        print(f"Processing dataset - {dataset}")
        # Paths
        path = os.path.join(source, dataset)
        data_path = os.path.join(path, f'{dataset}_base.fvecs')
        query_path = os.path.join(path, f'{dataset}_query.fvecs')
        gt_path = os.path.join(path, f'{dataset}_groundtruth.ivecs')
        cluster_ids_path = os.path.join(path, f'{dataset}_cluster_id_{K}.ivecs')

        # Load data
        print("Loading data...")
        base = read_fvecs(data_path)
        queries = read_fvecs(query_path)
        gt = read_ivecs(gt_path)
        cluster_ids = read_ivecs(cluster_ids_path).flatten()

        assert len(cluster_ids) == base.shape[0]

        # Randomly select one query
        num_queries = queries.shape[0]
        query_index = np.random.randint(0, num_queries)
        
        query_index= 7605
        print(f"Selected query ID: {query_index}")

        query = queries[query_index]
        gt_ids = gt[query_index]
        gt_vecs = base[gt_ids]

        # Step 1: Determine GT-NN clusters
        gt_cluster_ids = np.unique(cluster_ids[gt_ids])

        # Step 2: Select base vectors only from GT-NN clusters
        mask = np.isin(cluster_ids, gt_cluster_ids)
        filtered_base = base[mask]
        filtered_cluster_ids = cluster_ids[mask]

        # Step 3: PCA
        all_points = np.vstack([filtered_base, query[np.newaxis, :], gt_vecs])
        pca = PCA(n_components=2)
        all_points_2d = pca.fit_transform(all_points)

        base_2d = all_points_2d[:len(filtered_base)]
        query_2d = all_points_2d[len(filtered_base)]
        gt_2d = all_points_2d[len(filtered_base) + 1:]

        # Step 4: Plot
        print("Drawing figure...")
        plt.figure(figsize=(10, 8))
        
        # Use a perceptually uniform colormap for better distinction
        unique_clusters = np.unique(filtered_cluster_ids)
        num_clusters = len(unique_clusters)
        
        # Choose better colormaps based on number of clusters
        if num_clusters <= 10:
            cmap = plt.cm.tab10
        elif num_clusters <= 20:
            cmap = plt.cm.tab20
        else:
            cmap = plt.cm.viridis
        
        # Create a color dictionary for consistent colors
        color_dict = {c: cmap(i % cmap.N) for i, c in enumerate(unique_clusters)}
        
        # Plot clusters with consistent colors and better transparency
        for c in unique_clusters:
            idx = np.where(filtered_cluster_ids == c)[0]
            cluster_color = color_dict[c]
            # Adjust alpha for better visibility
            cluster_color_with_alpha = to_rgba(cluster_color, alpha=0.3)
            plt.scatter(base_2d[idx, 0], base_2d[idx, 1],
                        s=10, color=cluster_color_with_alpha, 
                        label=f'cluster {c}')

        # Use more visible colors for important points
        plt.scatter(gt_2d[:, 0], gt_2d[:, 1], s=50,
                    edgecolors='blue', facecolors='none', linewidths=2.0, 
                    label='gt-NN', zorder=10)
        plt.scatter(query_2d[0], query_2d[1], color='red',
                    s=150, marker='*', label='query', zorder=11)

        # Draw connecting lines with better visibility
        for pt in gt_2d:
            plt.plot([query_2d[0], pt[0]], [query_2d[1], pt[1]],
                     linestyle='--', color='darkblue', alpha=0.6, linewidth=1.0)

        plt.title(f"Query #{query_index} with GT-NNs (K={K})")
        plt.axis('equal')
        
        # Create a more readable legend with fewer items
        handles, labels = plt.gca().get_legend_handles_labels()
        # Only show a subset of cluster labels if there are too many
        if len(unique_clusters) > 10:
            cluster_handles = handles[:-2]  # All except gt-NN and query
            cluster_labels = labels[:-2]
            selected_indices = np.linspace(0, len(cluster_handles)-1, 10, dtype=int)
            selected_handles = [cluster_handles[i] for i in selected_indices]
            selected_labels = [cluster_labels[i] for i in selected_indices]
            # Add back gt-NN and query
            selected_handles.extend(handles[-2:])
            selected_labels.extend(labels[-2:])
            plt.legend(selected_handles, selected_labels, loc='best', fontsize=9)
        else:
            plt.legend(loc='best', fontsize=9)
            
        plt.tight_layout()

        filename = f"query_{query_index}_knn_clusters_K{K}.png"
        plt.savefig(filename, dpi=400)
        print(f"Saved to {filename}")

if __name__ == '__main__':
    main()
