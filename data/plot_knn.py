import numpy as np
import matplotlib.pyplot as plt
import os

# List of datasets
# , 'msong', 'gist', 'tiny5m', 'sift10m'
datasets = ['gist']

for dataset in datasets:
    # Load the indegree data
    indegree_file = f"{dataset}_faiss_knn_indegree.npy"
    
    if not os.path.exists(indegree_file):
        print(f"Skipping {dataset} - file not found")
        continue
        
    indegree = np.load(indegree_file)
    
    # Count frequency of each indegree value
    unique_degrees, counts = np.unique(indegree, return_counts=True)
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(unique_degrees, counts, 'b-', linewidth=2)
    
    plt.title(f'In-degree Distribution for {dataset}')
    plt.xlabel('In-degree')
    plt.ylabel('Number of Vertices')
    
    # Use log scale for better visualization
    plt.yscale('log')
    
    # Add grid
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    # Save the plot
    plt.savefig(f"{dataset}_indegree_distribution.png", dpi=400)
    plt.close()
    
    # Print some statistics
    print(f"\nDataset: {dataset}")
    print(f"Average in-degree: {np.mean(indegree):.2f}")
    print(f"Maximum in-degree: {np.max(indegree)}")
    print(f"Minimum in-degree: {np.min(indegree)}")
