import matplotlib.pyplot as plt
import numpy as np

# List of datasets to process
datasets = ['tiny5m', 'sift10m', 'sift', 'gist', 'msong']

percentiles = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]

print("Indegree Statistics:")
print("-" * 50)
print(f"{'Dataset':<10} | {'Percentile':<10} | {'Indegree':<10}")
print("-" * 50)

for dataset in datasets:
    # Read histogram data for each dataset
    hist_path = f'../{dataset}_indegree_histogram.txt'
    indegree = []
    counts = []
    
    with open(hist_path, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:  # Skip first line
            k, v = map(int, line.strip().split())
            # Expand the histogram data
            indegree.extend([k] * v)
    
    indegree = np.array(indegree)
    
    # Calculate percentiles
    for p in percentiles:
        percentile_value = np.percentile(indegree, p)
        print(f"{dataset:<10} | {p:>3}%        | {percentile_value:<10.2f}")
    print("-" * 50)
