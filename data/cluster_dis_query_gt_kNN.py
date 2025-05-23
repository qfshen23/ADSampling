import matplotlib.pyplot as plt
import numpy as np

# Data from the comments
k_values = [64, 256, 512, 1024]
ranks = list(range(10))  # 0 to 9

# Parse the data
data = {
    64: [56.627, 7.854, 5.187, 4.256, 3.81, 3.085, 3.156, 2.072, 2.25, 1.982],
    256: [40.841, 7.154, 4.643, 3.692, 3.88, 2.719, 2.741, 2.406, 2.077, 2.134],
    512: [33.526, 6.002, 4.62, 4.234, 3.238, 2.97, 2.91, 2.503, 2.019, 1.986],
    1024: [28.445, 5.519, 3.947, 3.607, 2.94, 2.61, 2.438, 2.179, 2.23, 1.712]
}

# Create a figure with 2x2 subplots
fig, axs = plt.subplots(2, 2, figsize=(15, 10))
axs = axs.flatten()
fig.suptitle('Groundtruth k-NNs distributed across cluster ranks\n(rank = distance from query cluster)', fontsize=16)

# Plot each K value in a separate subplot
for i, k in enumerate(k_values):
    axs[i].bar(ranks, data[k], color='skyblue')
    axs[i].set_title(f'K = {k}')
    axs[i].set_xlabel('Rank (distance from query\'s nearest centroid)')
    axs[i].set_ylabel('Percentage of ground truth kNNs')
    axs[i].set_xticks(ranks)
    axs[i].grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add percentage values on top of each bar
    for j, value in enumerate(data[k]):
        axs[i].text(j, value + 0.5, f'{value:.1f}%', ha='center')

# Adjust layout
plt.tight_layout()
plt.savefig('cluster_query_gt_kNN_distribution.png', dpi=600)