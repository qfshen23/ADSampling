import matplotlib.pyplot as plt
import numpy as np

# Read the hop stats data
ranks = []
avg_hops = []
with open('../sift_hop_stats.txt', 'r') as f:
    for line in f:
        rank, hop = line.strip().split('\t')
        ranks.append(int(rank))
        avg_hops.append(float(hop))

# Create the line plot
plt.figure(figsize=(10, 6))
plt.plot(ranks, avg_hops, '-', linewidth=2)

plt.xlabel('Ground Truth Rank')
plt.ylabel('Average Number of Hops')
plt.title('Average Hops vs Ground Truth Rank')
plt.grid(True)

# Save the plot
plt.savefig('hnsw_gt_hops.png', dpi=400)
plt.close()
