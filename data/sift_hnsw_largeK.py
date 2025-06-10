import matplotlib.pyplot as plt
import numpy as np

# Data for K=1K
k1_limited_dco = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15]  # in thousands
k1_recall = [62.2, 74.8, 82.6, 87.6, 91, 93.3, 95, 96.2, 97.1, 98.2, 98.9, 99.68]
k1_qps = [1910, 1381, 1083, 917, 764, 660, 578, 520, 483, 409, 368, 301]

# Data for K=10K  
k10_limited_dco = [20, 30, 40, 50, 60, 70, 80]  # in thousands
k10_recall = [85.3, 94.2, 97.6, 98.9, 99.5, 99.79, 99.9]
k10_qps = [186, 130, 101, 83, 72, 64, 58]

# Create two line plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: Recall vs Limited DCO
ax1.plot(k1_limited_dco, k1_recall, 'o-', label='K=1K', linewidth=2, markersize=6)
ax1.plot(k10_limited_dco, k10_recall, 's-', label='K=10K', linewidth=2, markersize=6)
ax1.set_xlabel('Limited #DCO (thousands)')
ax1.set_ylabel('Recall (%)')
ax1.set_title('Recall vs Limited #DCO')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Plot 2: QPS vs Limited DCO
ax2.plot(k1_limited_dco, k1_qps, 'o-', label='K=1K', linewidth=2, markersize=6)
ax2.plot(k10_limited_dco, k10_qps, 's-', label='K=10K', linewidth=2, markersize=6)
ax2.set_xlabel('Limited #DCO (thousands)')
ax2.set_ylabel('QPS (queries per second)')
ax2.set_title('QPS vs Limited #DCO')
ax2.set_yscale('log')
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.savefig('sift_hnsw_largeK.png', dpi=400)
plt.close()