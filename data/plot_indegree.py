import matplotlib.pyplot as plt
import numpy as np

# List of datasets to process
datasets = ['tiny5m', 'sift10m', 'sift', 'gist', 'msong']

for dataset in datasets:
    # Read histogram data for each dataset
    hist_path = f'../{dataset}_indegree_histogram.txt'
    indegree = []
    counts = []
    
    with open(hist_path, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:  # Skip first line
            k, v = map(int, line.strip().split())
            indegree.append(k)
            counts.append(v)
            
    indegree = np.array(indegree)
    counts = np.array(counts)
    
    # Create new figure for each dataset
    plt.figure(figsize=(10,6))
    plt.plot(indegree, counts, marker='o', linestyle='-', markersize=4)
    plt.yscale('log')   # Log scale for y-axis
    plt.xlabel('Indegree')
    plt.ylabel('Number of nodes (log scale)')
    plt.title(f'Indegree Histogram - {dataset}')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(f'indegree_histogram_{dataset}.png', dpi=400)
    plt.close()