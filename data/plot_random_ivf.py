import matplotlib.pyplot as plt
import numpy as np

'''
10K	60	87	228	62785		60	61761	21024	29.8
								26024	35.5
								30024	41.2
								35024	48.3
								39024	54
								45024	62.5
								50024	69.6
								55024	75.88
								60024	80.6
								65024	83.9
								70024	85.8
								75024	86.6
								80024	86.9
'''

# Data points for top-1K
dcos_1k = [5024, 6024, 7024, 10024, 14024, 18024, 22024, 26024, 30024, 33024, 35024, 40024]
recalls_1k = [11.5, 14.3, 17.2, 25.8, 37.4, 48.9, 60.3, 71.6, 79.9, 83.88, 85.6, 88.0]

# Data points for top-10K
dcos_10k = [11024, 22048, 26048, 30024, 35024, 39024, 45024, 50001, 54486, 57956, 60396, 61830]
recalls_10k = [29.8, 35.5, 41.2, 48.3, 54.0, 62.5, 69.6, 75.88, 80.6, 83.9, 85.8, 86.6]

# Create two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# First subplot for Top-1K
ax1.plot(dcos_1k, recalls_1k, marker='o', linewidth=2, markersize=6)
ax1.set_xlabel('Number of Distance Computations (DCOs)')
ax1.set_ylabel('Recall (%)')
ax1.set_title('Top-1K Random IVF on SIFT1M')
ax1.grid(True, alpha=0.3)
ax1.plot(32510, 88.8, marker='*', color='red', markersize=12, label='Exact Search')
ax1.legend()

# Second subplot for Top-10K
ax2.plot(dcos_10k, recalls_10k, marker='s', linewidth=2, markersize=6)
ax2.set_xlabel('Number of Distance Computations (DCOs)')
ax2.set_ylabel('Recall (%)')
ax2.set_title('Top-10K Random IVF on SIFT1M')
ax2.grid(True, alpha=0.3)
ax2.plot(61761, 86.9, marker='*', color='red', markersize=12, label='Exact Search')
ax2.legend()

plt.tight_layout()
plt.savefig('random_ivf.png', dpi=400)
plt.close()