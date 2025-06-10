import matplotlib.pyplot as plt
import numpy as np


# Data for K=1K with 5K DCO
k1_props = [5, 10, 15, 20, 25, 50]
k1_recalls = [32.1, 42.7, 52.4, 59.1, 65.7, 83.7]

# Data for K=10K with 20K DCO
k10_props = [5, 10, 15, 20, 25, 50]
k10_recalls = [44.1, 51.5, 57.8, 62.2, 66.5, 79.8]

# Set up the figure
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

# Create bar positions
x = np.arange(len(k1_props))
width = 0.35

# Create bars
bars1 = ax.bar(x - width/2, k1_recalls, width, label='K=1K with 5K DCO', color='skyblue', alpha=0.7)
bars2 = ax.bar(x + width/2, k10_recalls, width, label='K=10K with 20K DCO', color='lightcoral', alpha=0.7)

# Customize the plot
ax.set_xlabel('Random Dimension Proportion (%)')
ax.set_ylabel('Recall (%)')
ax.set_title('Rank by Random Dimensions before Exact Dcos on SIFT1M')
ax.set_xticks(x)
ax.set_xticklabels(k1_props)
ax.set_ylim(0, 100)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('random_rerank_ivf.png', dpi=600)
plt.close()