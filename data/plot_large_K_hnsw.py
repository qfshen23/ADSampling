import matplotlib.pyplot as plt
import numpy as np

# Data for each dataset
K_values = [1, 10, 100, 1000, 10000]
log_K = np.log10(K_values)

# SIFT data
sift_recall = [96.5, 96.7, 96.7, 99.78, 99.98]
sift_qps = [4998, 3998, 1810, 294, 52]

# GIST data 
gist_recall = [90.2, 89.7, 90.0, 96.86, 99.4]
gist_qps = [541, 432, 244, 56, 10]

# MSONG data
msong_recall = [99.7, 99.2, 98.2, 99.7, 99.9]
msong_qps = [2599, 1707, 1058, 175, 34]

# Plot SIFT
fig, ax1 = plt.subplots(figsize=(8, 6))
ax2 = ax1.twinx()

ax1.bar(log_K, sift_qps, color='skyblue', alpha=0.7, label='QPS')
ax2.plot(log_K, sift_recall, color='red', linewidth=2, marker='o', label='Recall')

ax1.set_xlabel('K (Power of 10)')
ax1.set_ylabel('QPS')
ax2.set_ylabel('Recall (%)')
ax1.set_title('SIFT')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

plt.tight_layout()
plt.savefig('hnsw_large_K_sift.png', dpi=400)
plt.close()

# Plot GIST
fig, ax1 = plt.subplots(figsize=(8, 6))
ax2 = ax1.twinx()

ax1.bar(log_K, gist_qps, color='skyblue', alpha=0.7, label='QPS')
ax2.plot(log_K, gist_recall, color='red', linewidth=2, marker='o', label='Recall')

ax1.set_xlabel('K (Power of 10)')
ax1.set_ylabel('QPS')
ax2.set_ylabel('Recall (%)')
ax1.set_title('GIST')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

plt.tight_layout()
plt.savefig('hnsw_large_K_gist.png', dpi=400)    
plt.close()

# Plot MSONG
fig, ax1 = plt.subplots(figsize=(8, 6))
ax2 = ax1.twinx()

ax1.bar(log_K, msong_qps, color='skyblue', alpha=0.7, label='QPS')
ax2.plot(log_K, msong_recall, color='red', linewidth=2, marker='o', label='Recall')

ax1.set_xlabel('K (Power of 10)')
ax1.set_ylabel('QPS')
ax2.set_ylabel('Recall (%)')
ax1.set_title('MSONG')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

plt.tight_layout()
plt.savefig('hnsw_large_K_msong.png', dpi=400)
plt.close()

# Plot Tiny5m
fig, ax1 = plt.subplots(figsize=(8, 6))
ax2 = ax1.twinx()

tiny_recall = [92.1, 92.1, 92.4, 94.0, 99.1]
tiny_qps = [1465, 565, 222, 76, 12]

ax1.bar(log_K, tiny_qps, color='skyblue', alpha=0.7, label='QPS')
ax2.plot(log_K, tiny_recall, color='red', linewidth=2, marker='o', label='Recall')

ax1.set_xlabel('K (Power of 10)')
ax1.set_ylabel('QPS')
ax2.set_ylabel('Recall (%)')
ax1.set_title('Tiny5m')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

plt.tight_layout()
plt.savefig('hnsw_large_K_tiny5m.png', dpi=400)
plt.close()

# Plot SIFT10m
fig, ax1 = plt.subplots(figsize=(8, 6))
ax2 = ax1.twinx()

sift10m_recall = [94.3, 95.1, 93.3, 99.1, 99.8]
sift10m_qps = [2854, 1958, 1082, 156, 24]

ax1.bar(log_K, sift10m_qps, color='skyblue', alpha=0.7, label='QPS')
ax2.plot(log_K, sift10m_recall, color='red', linewidth=2, marker='o', label='Recall')

ax1.set_xlabel('K (Power of 10)')
ax1.set_ylabel('QPS')
ax2.set_ylabel('Recall (%)')
ax1.set_title('SIFT10m')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

plt.tight_layout()
plt.savefig('hnsw_large_K_sift10m.png', dpi=400)
plt.close()