import matplotlib.pyplot as plt
import numpy as np

# Read data from the histogram file
windows = []
hit_counts = []

with open('../cluster_histogram_per100_sift.txt', 'r') as f:
    for line in f:
        window, count = map(float, line.strip().split())
        windows.append(window)
        hit_counts.append(count)

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(windows, hit_counts, '-o', markersize=4)

plt.xlabel('Window Index (Each Window Contains 100 Visited Points)')
plt.ylabel('Average Number of Points in Top-3 Closest Clusters')
plt.title('Distribution of Top-3 Cluster Hits During Search')

plt.grid(True, linestyle='--', alpha=0.7)

# Add text box with statistics
avg_hits = np.mean(hit_counts)
plt.text(0.95, 0.95, f'Average Hits: {avg_hits:.2f}', 
         transform=plt.gca().transAxes,
         verticalalignment='top',
         horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('cluster_hitcount_distribution.png', dpi=400, bbox_inches='tight')
plt.close()
