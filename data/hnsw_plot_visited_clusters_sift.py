import numpy as np
import matplotlib.pyplot as plt

# Read cluster rank access stats
rank_counts = []
access_counts = []
with open('cluster_rank_access_stats_sift.txt', 'r') as f:
    for line in f:
        rank, count = map(int, line.strip().split())
        rank_counts.append(rank)
        access_counts.append(count)

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,5))

# Plot histogram of access counts by rank
ax1.bar(rank_counts, access_counts, edgecolor='black')
ax1.set_title('Distribution of Access Counts by Cluster Rank')
ax1.set_xlabel('Cluster Rank')
ax1.set_ylabel('Number of Accesses')
ax1.set_yscale('log')  # Use log scale since counts vary widely

# Calculate cumulative percentage
total_accesses = sum(access_counts)
cumulative = np.cumsum(access_counts) / total_accesses * 100

# Plot cumulative percentage
ax2.plot(rank_counts, cumulative, marker='o')
ax2.set_title('Cumulative Percentage of Accesses by Rank')
ax2.set_xlabel('Cluster Rank')
ax2.set_ylabel('Cumulative Percentage')
ax2.grid(True)

# Print statistics
print(f"Statistics for cluster rank accesses:")
print(f"Total accesses: {total_accesses}")
print(f"Number of ranks: {len(rank_counts)}")
print(f"Top 5 ranks account for {cumulative[4]:.1f}% of accesses")
print(f"Top 10 ranks account for {cumulative[9]:.1f}% of accesses")

plt.tight_layout()
plt.savefig('cluster_rank_access_stats_sift.png', dpi=400)
plt.close()
