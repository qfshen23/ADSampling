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
    iv = iv[:, 1:]
    return iv
    
K = 1024
datasets = ['sift', 'gist']  # Add more datasets as needed: 'tiny5m', 'msong', etc.
data_path = '/data/vector_datasets'
top_k_clusters = 64    
k_values = [10, 100]  # k nearest neighbors to consider (out of 10000 available)

# Set publication-quality matplotlib parameters
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 100,
    'savefig.dpi': 400,
    'savefig.bbox': 'tight',
    'pdf.fonttype': 42,  # TrueType fonts for PDF
    'ps.fonttype': 42,
})

# Pre-compute data for all datasets
print("=" * 60)
print("Loading datasets...")
print("=" * 60)

dataset_data = {}
for dataset in datasets:
    dataset_path = os.path.join(data_path, dataset)
    base_file = os.path.join(dataset_path, f"{dataset}_base.fvecs")
    centroid_file = os.path.join(dataset_path, f'{dataset}_centroid_{K}.fvecs')
    query_file = os.path.join(dataset_path, f"{dataset}_query.fvecs")
    gt_file = os.path.join(dataset_path, f'{dataset}_groundtruth_10000.ivecs')
    cluster_ids_file = os.path.join(dataset_path, f'{dataset}_cluster_id_{K}.ivecs')

    print(f"\nLoading dataset: {dataset}")
    
    try:
        base_vectors = read_fvecs(base_file)
        centroids_vectors = read_fvecs(centroid_file)
        query_vectors = read_fvecs(query_file)
        point_to_cluster = read_ivecs(cluster_ids_file)
        all_queries_nn = read_ivecs(gt_file)

        n_queries = query_vectors.shape[0]
        n_points = base_vectors.shape[0]
        n_clusters = K

        print(f"  ✓ Loaded: {n_queries} queries, {n_points} points, {n_clusters} clusters")

        # Compute top clusters for each query
        queries_top_clusters = np.zeros((n_queries, top_k_clusters), dtype=int)
        for i, query in enumerate(query_vectors):
            distances = np.linalg.norm(centroids_vectors - query, axis=1)
            top_k = np.argsort(distances)[:top_k_clusters]
            queries_top_clusters[i] = top_k

        dataset_data[dataset] = {
            'query_vectors': query_vectors,
            'point_to_cluster': point_to_cluster,
            'all_queries_nn': all_queries_nn,
            'queries_top_clusters': queries_top_clusters,
            'n_queries': n_queries
        }
    except Exception as e:
        print(f"  ✗ Error loading {dataset}: {e}")
        print(f"  Skipping {dataset}...")

# Update datasets to only include successfully loaded ones
datasets = list(dataset_data.keys())
if len(datasets) == 0:
    print("\n✗ No datasets loaded successfully. Exiting.")
    exit(1)

print("\n" + "=" * 60)
print(f"Successfully loaded {len(datasets)} dataset(s): {datasets}")
print("=" * 60)

# Create combined plots for each k value
os.makedirs(f'ivf-hit-count-combined-{K}', exist_ok=True)

for k in k_values:
    print(f"\nGenerating combined plot for k={k}")
    
    n_datasets = len(datasets)
    # Create subplots in a row
    fig, axes = plt.subplots(1, n_datasets, figsize=(5.5*n_datasets, 4.5))
    
    # Handle case when there's only one dataset
    if n_datasets == 1:
        axes = [axes]
    
    for idx, dataset in enumerate(datasets):
        data = dataset_data[dataset]
        n_queries = data['n_queries']
        queries_nn = data['all_queries_nn'][:, :k]
        point_to_cluster = data['point_to_cluster']
        queries_top_clusters = data['queries_top_clusters']
        
        # Compute histogram
        histogram = np.zeros(top_k_clusters, dtype=float)
        for query_idx in range(n_queries):
            nn_indices = queries_nn[query_idx]
            nn_clusters = point_to_cluster[nn_indices]
            top_clusters = queries_top_clusters[query_idx]
            for i, cluster_id in enumerate(top_clusters):
                count_in_cluster = np.sum(nn_clusters == cluster_id)
                histogram[i] += count_in_cluster

        percentage = histogram / n_queries
        cumulative_percentage = np.cumsum(percentage)

        # Key points for annotation
        thresholds = [0.8, 0.9, 0.95, 0.98, 0.99]
        key_points = []
        total_cum = cumulative_percentage[-1]
        for t in thresholds:
            pos = np.searchsorted(cumulative_percentage, t * total_cum)
            if pos >= top_k_clusters:
                pos = top_k_clusters - 1
            key_points.append((pos + 1, cumulative_percentage[pos]))

        # Plot on subplot
        ax1 = axes[idx]
        ax2 = ax1.twinx()

        # Bar plot for hit count distribution
        bars = ax1.bar(range(1, top_k_clusters + 1), percentage, 
                      color='#4A90E2', alpha=0.6, label='Hit Count', width=0.85, edgecolor='none')
        ax1.set_xlabel('Top-$k$-th Cluster', fontsize=12, weight='semibold')
        ax1.set_ylabel('Avg. Hit Count', color='#4A90E2', fontsize=12, weight='semibold')
        ax1.tick_params(axis='y', labelcolor='#4A90E2')
        ax1.set_xticks([1] + list(range(10, top_k_clusters + 1, 10)))
        ax1.grid(axis='y', linestyle='--', alpha=0.3, linewidth=0.5)
        ax1.set_axisbelow(True)

        # Cumulative line plot
        line = ax2.plot(range(1, top_k_clusters + 1), cumulative_percentage, 
                       color='#E74C3C', linewidth=2.5, alpha=0.9, label='Cumulative', marker='')
        ax2.set_ylabel('Cumulative Count', color='#E74C3C', fontsize=12, weight='semibold')
        ax2.tick_params(axis='y', labelcolor='#E74C3C')

        # Annotate key points (select subset to avoid clutter)
        selected_thresholds = [0.8, 0.95, 0.99]
        for i, (x, y) in enumerate(key_points):
            if thresholds[i] in selected_thresholds:
                ax2.scatter(x, y, color='black', s=40, zorder=5, marker='o', edgecolors='white', linewidths=1)
                # Position annotations to avoid overlap
                if thresholds[i] == 0.8:
                    xytext = (0, 10)
                elif thresholds[i] == 0.95:
                    xytext = (0, 10)
                else:  # 0.99
                    xytext = (0, 10)
                    
                ax2.annotate(f'{int(thresholds[i]*100)}%',
                           (x, y),
                           textcoords="offset points",
                           xytext=xytext,
                           ha='center',
                           fontsize=9,
                           color='black',
                           weight='bold',
                           bbox=dict(facecolor='white', edgecolor='gray', 
                                   boxstyle='round,pad=0.25', alpha=0.9, linewidth=0.8))

        # Title for each subplot - use dataset name
        dataset_title = dataset.upper()
        if dataset == 'tiny5m':
            dataset_title = 'Tiny5M'
        elif dataset == 'sift10m':
            dataset_title = 'SIFT10M'
        ax1.set_title(f'{dataset_title}', fontsize=14, weight='bold', pad=10)
        
        # Only show legend on the first subplot
        if idx == 0:
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, 
                      loc='upper left', framealpha=0.95, edgecolor='gray', 
                      fancybox=True, shadow=False)

    # Overall title (optional - can be removed for cleaner look)
    # fig.suptitle(f'{k}-NN Distribution in Top-{top_k_clusters} Clusters', 
    #             fontsize=16, weight='bold', y=0.99)
    
    plt.tight_layout()
    
    # Save with high quality in both PNG and PDF
    output_png = f'ivf-hit-count-combined-{K}/combined-k{k}-c{K}-all_datasets.png'
    output_pdf = f'ivf-hit-count-combined-{K}/combined-k{k}-c{K}-all_datasets.pdf'
    
    plt.savefig(output_png, dpi=400, bbox_inches='tight', pad_inches=0.1)
    plt.savefig(output_pdf, bbox_inches='tight', pad_inches=0.1)
    
    print(f"  ✓ Saved: {output_png}")
    print(f"  ✓ Saved: {output_pdf}")
    plt.close()

print("\n" + "=" * 60)
print("All plots generated successfully!")
print("=" * 60)
