import matplotlib.pyplot as plt
import numpy as np

# Data for each dataset
K_values = [1, 10, 100, 1000, 10000]
log_K = np.log10(K_values)
x = np.arange(len(log_K))
width = 0.35  # Width of bars

# SIFT data
sift_ivf_recall = [95.99, 95.14, 95.1, 94.95, 94.9]
sift_ivf_qps = [993, 995, 671, 385, 160]
sift_hnsw_recall = [96.5, 96.7, 96.7, 99.78, 99.98]
sift_hnsw_qps = [4998, 3998, 1810, 294, 52]

# GIST data
gist_ivf_recall = [87.5, 88.25, 89.1, 88.3, 86]
gist_ivf_qps = [68, 55, 34, 23, 14]
gist_hnsw_recall = [90.2, 89.7, 90.0, 96.86, 99.4]
gist_hnsw_qps = [541, 432, 244, 56, 10]

# MSONG data
msong_ivf_recall = [100, 96.14, 97.6, 96.6, 95.6]
msong_ivf_qps = [1988, 284, 146, 103, 50]
msong_hnsw_recall = [99.7, 99.2, 98.2, 99.7, 99.9]
msong_hnsw_qps = [2599, 1707, 1058, 175, 34]

# Tiny5m data
tiny_ivf_recall = [100, 91.4, 89, 89.4, 89.7]
tiny_ivf_qps = [1002, 40, 33, 21, 12]
tiny_hnsw_recall = [92.1, 92.1, 92.4, 94.0, 99.1]
tiny_hnsw_qps = [1465, 565, 222, 76, 12]

# SIFT10m data
sift10m_ivf_recall = [96.5, 96.7, 96.1, 95.9, 96.3]
sift10m_ivf_qps = [269, 212, 140, 86, 38]
sift10m_hnsw_recall = [94.3, 95.1, 93.3, 99.1, 99.8]
sift10m_hnsw_qps = [2854, 1958, 1082, 156, 24]

# Plot function
def plot_dataset(ivf_qps, ivf_recall, hnsw_qps, hnsw_recall, title):
    fig, ax1 = plt.subplots(figsize=(8, 6))
    ax2 = ax1.twinx()
    
    # Plot bars
    ax1.bar(x - width/2, ivf_qps, width, color='skyblue', alpha=0.7, label='IVF QPS')
    ax1.bar(x + width/2, hnsw_qps, width, color='lightgreen', alpha=0.7, label='HNSW QPS')
    
    # Plot lines
    ax2.plot(x - width/2, ivf_recall, color='red', linewidth=2, marker='o', label='IVF Recall')
    ax2.plot(x + width/2, hnsw_recall, color='darkred', linewidth=2, marker='s', label='HNSW Recall')
    
    ax1.set_xlabel('K (Power of 10)')
    ax1.set_ylabel('QPS')
    ax2.set_ylabel('Recall (%)')
    ax1.set_title(title)
    ax1.set_xticks(x)
    ax1.set_xticklabels(log_K)
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(f'large_K_{title.lower()}.png', dpi=400)
    plt.close()

# Generate plots for each dataset
plot_dataset(sift_ivf_qps, sift_ivf_recall, sift_hnsw_qps, sift_hnsw_recall, 'SIFT')
plot_dataset(gist_ivf_qps, gist_ivf_recall, gist_hnsw_qps, gist_hnsw_recall, 'GIST')
plot_dataset(msong_ivf_qps, msong_ivf_recall, msong_hnsw_qps, msong_hnsw_recall, 'MSONG')
plot_dataset(tiny_ivf_qps, tiny_ivf_recall, tiny_hnsw_qps, tiny_hnsw_recall, 'Tiny5m')
plot_dataset(sift10m_ivf_qps, sift10m_ivf_recall, sift10m_hnsw_qps, sift10m_hnsw_recall, 'SIFT10m')
