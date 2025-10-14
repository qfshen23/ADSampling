import matplotlib.pyplot as plt
import numpy as np

# Data for top-10 experiments
top10_data = {
    'gist': {
        'Greedy+ADSampling+': {
            'recall': [70.43, 79.23, 84.76, 88.19, 90.46, 92.45, 94.79, 95.67, 96.85, 97.69, 98.64, 99.18],
            'qps': [314, 235, 191, 162, 142, 119, 98, 90, 78, 69, 55, 47]
        },
        'Ours+ADSampling+': {
            'recall': [67.76, 77.15, 82.34, 86.91, 89.07, 91.65, 93.32, 94.83, 95.73, 96.7, 98.06, 98.66],
            'qps': [397.15, 310.08, 259.79, 216.02, 192.01, 164.33, 143.53, 123.93, 112.86, 101.26, 77.27, 68.24]
        }
    },
    'sift': {
        'Greedy+ADSampling+': {
            'recall': [75.02, 87.36, 92.46, 95.14, 96.65, 97.58, 98.27, 98.65, 98.97, 99.18],
            'qps': [3648, 2123, 1550, 1193, 972, 819, 714, 625, 547, 506]
        },
        'Ours+ADSampling+': {
            'recall': [75.36, 87.08, 91.2, 94.57, 96.46, 97.54, 98.24, 98.66, 98.75, 99.01],
            'qps': [3064.26, 1958.81, 1568.59, 1149.66, 915.88, 726.1, 637.65, 573.63, 523.12, 462.79]
        }
    },
    'tiny5m': {
        'Greedy+ADSampling+': {
            'recall': [72.17, 78.97, 82.75, 86.01, 88.21, 89.99, 91.3, 93.29, 95.48, 96.79, 97.25, 98.02, 98.59, 98.92],
            'qps': [209, 147, 114, 95, 81, 73, 68, 56, 42, 34, 31, 26, 23, 20]
        },
        'Ours+ADSampling+': {
            'recall': [72.02, 79.51, 82.35, 85.49, 87.74, 89.63, 91.03, 93.15, 95.39, 96.52, 97.19, 97.75, 98.28, 98.66],
            'qps': [336.28, 225.13, 198.58, 161.08, 133.06, 116.39, 103.21, 78.09, 58.41, 49.72, 44.26, 39.05, 35.4, 30.24]
        }
    }
}

# Data for top-1 experiments
top1_data = {
    'gist': {
        'Greedy+ADSampling+': {
            'recall': [75.5, 82.9, 86.8, 89.3, 91.5, 93.3, 94.5, 95.5, 96.6, 97.6, 98.4, 98.7, 98.9],
            'qps': [393, 298, 244, 204, 177, 159, 142, 127, 98, 86, 70, 64, 55]
        },
        'Ours+ADSampling+': {
            'recall': [73.2, 81.7, 85.5, 88.4, 91.3, 92.6, 93.9, 94.6, 96.2, 97.1, 98, 98.2, 98.7],
            'qps': [455.56, 358.87, 312.02, 267.9, 226.77, 203.98, 174.94, 166, 141.47, 124.42, 101.85, 92.25, 77.74]
        }
    },
    'sift': {
        'Greedy+ADSampling+': {
            'recall': [79.41, 89.91, 94.25, 95.99, 96.96, 97.56, 98.54, 98.88, 99.02],
            'qps': [3979, 2201, 1512, 1149, 922, 763, 514, 363, 294]
        },
        'Ours+ADSampling+': {
            'recall': [75.87, 87.84, 94.02, 95.95, 96.98, 97.52, 98.53, 98.94, 99.01],
            'qps': [3830.72, 2635.7, 1730.64, 1283.5, 1038.46, 940.13, 603.01, 422.93, 305.01]
        }
    }
}

# Plot top-10 experiments - each dataset separately
colors = ['blue', 'orange']
methods = ['Greedy+ADSampling+', 'Ours+ADSampling+']

# Plot GIST dataset
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
data = top10_data['gist']
for j, method in enumerate(methods):
    ax.plot(data[method]['recall'], data[method]['qps'], 
            'o-', color=colors[j], label=method, linewidth=2, markersize=6)
ax.set_xlabel('Recall (%)')
ax.set_ylabel('QPS')
ax.set_title('Top-10: GIST Dataset - QPS vs Recall')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(min(min(data[methods[0]]['recall']), min(data[methods[1]]['recall'])) - 1,
            max(max(data[methods[0]]['recall']), max(data[methods[1]]['recall'])) + 1)
plt.tight_layout()
plt.savefig('exp2_top10_gist_qps_vs_recall.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot SIFT dataset
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
data = top10_data['sift']
for j, method in enumerate(methods):
    ax.plot(data[method]['recall'], data[method]['qps'], 
            'o-', color=colors[j], label=method, linewidth=2, markersize=6)
ax.set_xlabel('Recall (%)')
ax.set_ylabel('QPS')
ax.set_title('Top-10: SIFT Dataset - QPS vs Recall')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(min(min(data[methods[0]]['recall']), min(data[methods[1]]['recall'])) - 1,
            max(max(data[methods[0]]['recall']), max(data[methods[1]]['recall'])) + 1)
plt.tight_layout()
plt.savefig('exp2_top10_sift_qps_vs_recall.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot TINY5M dataset
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
data = top10_data['tiny5m']
for j, method in enumerate(methods):
    ax.plot(data[method]['recall'], data[method]['qps'], 
            'o-', color=colors[j], label=method, linewidth=2, markersize=6)
ax.set_xlabel('Recall (%)')
ax.set_ylabel('QPS')
ax.set_title('Top-10: TINY5M Dataset - QPS vs Recall')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(min(min(data[methods[0]]['recall']), min(data[methods[1]]['recall'])) - 1,
            max(max(data[methods[0]]['recall']), max(data[methods[1]]['recall'])) + 1)
plt.tight_layout()
plt.savefig('exp2_top10_tiny5m_qps_vs_recall.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot top-1 experiments - each dataset separately
# Plot GIST dataset
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
data = top1_data['gist']
for j, method in enumerate(methods):
    ax.plot(data[method]['recall'], data[method]['qps'], 
            'o-', color=colors[j], label=method, linewidth=2, markersize=6)
ax.set_xlabel('Recall (%)')
ax.set_ylabel('QPS')
ax.set_title('Top-1: GIST Dataset - QPS vs Recall')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(min(min(data[methods[0]]['recall']), min(data[methods[1]]['recall'])) - 1,
            max(max(data[methods[0]]['recall']), max(data[methods[1]]['recall'])) + 1)
plt.tight_layout()
plt.savefig('exp2_top1_gist_qps_vs_recall.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot SIFT dataset
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
data = top1_data['sift']
for j, method in enumerate(methods):
    ax.plot(data[method]['recall'], data[method]['qps'], 
            'o-', color=colors[j], label=method, linewidth=2, markersize=6)
ax.set_xlabel('Recall (%)')
ax.set_ylabel('QPS')
ax.set_title('Top-1: SIFT Dataset - QPS vs Recall')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(min(min(data[methods[0]]['recall']), min(data[methods[1]]['recall'])) - 1,
            max(max(data[methods[0]]['recall']), max(data[methods[1]]['recall'])) + 1)
plt.tight_layout()
plt.savefig('exp2_top1_sift_qps_vs_recall.png', dpi=300, bbox_inches='tight')
plt.close()