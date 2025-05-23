import matplotlib.pyplot as plt
import numpy as np

'''
k=64:
Groundtruth k-NNs distributed across cluster ranks (rank = distance from entry cluster):
rank 0: 47.271 rank 1: 8.017 rank 2: 6.355 rank 3: 4.345 rank 4: 4.497 rank 5: 3.921 rank 6: 3.143 rank 7: 3.033 rank 8: 2.937 rank 9: 1.809 rank 10: 1.906 rank 11: 2.002 rank 12: 1.697 rank 13: 1.285 rank 14: 1.405 rank 15: 1.226 rank 16: 0.721 rank 17: 0.716 rank 18: 0.582 rank 19: 0.629 

k=256:
Groundtruth k-NNs distributed across cluster ranks (rank = distance from entry cluster):
rank 0: 30.044 rank 1: 6.559 rank 2: 5.067 rank 3: 3.909 rank 4: 3.917 rank 5: 3.296 rank 6: 3.275 rank 7: 2.683 rank 8: 2.271 rank 9: 2.347 rank 10: 2.246 rank 11: 1.939 rank 12: 1.925 rank 13: 1.497 rank 14: 1.507 rank 15: 1.353 rank 16: 1.169 rank 17: 1.239 rank 18: 1.213 rank 19: 1.273 

K=512:
Groundtruth k-NNs distributed across cluster ranks (rank = distance from entry cluster):
rank 0: 23.534 rank 1: 5.416 rank 2: 3.86 rank 3: 4.518 rank 4: 3.271 rank 5: 2.969 rank 6: 2.577 rank 7: 2.749 rank 8: 2.066 rank 9: 1.977 rank 10: 1.711 rank 11: 2.007 rank 12: 1.909 rank 13: 1.456 rank 14: 1.238 rank 15: 1.327 rank 16: 1.082 rank 17: 1.072 rank 18: 1.272 rank 19: 1.251 

K=1024:
Groundtruth k-NNs distributed across cluster ranks (rank = distance from entry cluster):
rank 0: 18.123 rank 1: 4.557 rank 2: 3.905 rank 3: 3.363 rank 4: 3.168 rank 5: 2.428 rank 6: 2.238 rank 7: 1.955 rank 8: 2.027 rank 9: 1.615 rank 10: 1.794 rank 11: 1.654 rank 12: 1.666 rank 13: 1.542 rank 14: 1.032 rank 15: 1.598 rank 16: 1.238 rank 17: 1.287 rank 18: 1.098 rank 19: 0.94 

'''

# Data from the comments
k_values = [64, 256, 512, 1024]

# Parse the data
data = {
    64: [47.271, 8.017, 6.355, 4.345, 4.497, 3.921, 3.143, 3.033, 2.937, 1.809, 1.906, 2.002, 1.697, 1.285, 1.405, 1.226, 0.721, 0.716, 0.582, 0.629],
    256: [30.044, 6.559, 5.067, 3.909, 3.917, 3.296, 3.275, 2.683, 2.271, 2.347, 2.246, 1.939, 1.925, 1.497, 1.507, 1.353, 1.169, 1.239, 1.213, 1.273],
    512: [23.534, 5.416, 3.86, 4.518, 3.271, 2.969, 2.577, 2.749, 2.066, 1.977, 1.711, 2.007, 1.909, 1.456, 1.238, 1.327, 1.082, 1.072, 1.272, 1.251],
    1024: [18.123, 4.557, 3.905, 3.363, 3.168, 2.428, 2.238, 1.955, 2.027, 1.615, 1.794, 1.654, 1.666, 1.542, 1.032, 1.598, 1.238, 1.287, 1.098, 0.94]
}

# Create a figure with 2x2 subplots with better proportions
plt.figure(figsize=(16, 12))

# Create a 2x2 grid with appropriate spacing
gs = plt.GridSpec(2, 2, wspace=0.3, hspace=0.4)

# Plot each K value in a separate subplot
for i, k in enumerate(k_values):
    ax = plt.subplot(gs[i // 2, i % 2])
    ranks = list(range(len(data[k])))  # 0 to max rank
    ax.bar(ranks, data[k], color='skyblue', width=0.7)
    ax.set_title(f'K = {k}', fontsize=14, pad=10)
    ax.set_xlabel('Rank (distance from entry cluster)', fontsize=12)
    ax.set_ylabel('Percentage of ground truth kNNs', fontsize=12)
    ax.set_xticks(ranks)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add percentage values on top of each bar
    for j, v in enumerate(data[k]):
        # if v > 1.0:  # Only show labels for values above 1% to avoid clutter
        ax.text(j, v + 0.5, f'{v:.1f}', ha='center', fontsize=9)

# Add a main title with better positioning
plt.suptitle('Groundtruth k-NNs distributed across cluster ranks\n(rank = distance from entry cluster)', 
             fontsize=18, y=0.98)

# Adjust layout to make better use of space
# plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('cluster_entry_gt_kNN_distribution_large.png', dpi=600, bbox_inches='tight')
