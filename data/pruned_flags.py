import matplotlib.pyplot as plt
import numpy as np

# Parse the data from the results
data = {
    64: [0, 1.7942, 0.8748, 0.5141, 0.3325, 0.2467, 0.1686, 0.0856, 0.0571, 0.0444, 0.0354, 0.0319, 0.0108, 0.0096, 0.0109, 0.0066, 0.0043, 0.0047, 0.0020, 0.0013],
    256: [0, 3.1386, 1.7564, 1.1253, 0.7587, 0.5515, 0.4096, 0.2896, 0.2324, 0.1514, 0.1228, 0.1068, 0.0865, 0.0659, 0.0472, 0.0362, 0.0380, 0.0303, 0.0241, 0.0181],
    512: [0, 3.9058, 2.3068, 1.6384, 1.0374, 0.8118, 0.5825, 0.4362, 0.3333, 0.2656, 0.2307, 0.1627, 0.1311, 0.1141, 0.0956, 0.0701, 0.0614, 0.0556, 0.0489, 0.0428],
    1024: [0, 4.6351, 2.9864, 2.1542, 1.4637, 1.0733, 0.8154, 0.6555, 0.4873, 0.3922, 0.3476, 0.2762, 0.2290, 0.1724, 0.1485, 0.1238, 0.1093, 0.1057, 0.0860, 0.0653]
}

# Create a figure with 4 subplots
fig, axs = plt.subplots(2, 2, figsize=(15, 10))
axs = axs.flatten()

# Create x-axis labels (ranks)
ranks = np.arange(20)

# Plot each bar chart
for i, k in enumerate([64, 256, 512, 1024]):
    axs[i].bar(ranks, data[k], color='skyblue')
    axs[i].set_title(f'K = {k}')
    axs[i].set_xlabel('Rank')
    axs[i].set_ylabel('Count')
    axs[i].set_xticks(ranks)
    axs[i].grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('pruned_flags_rank_distribution.png', dpi=400)
