import matplotlib.pyplot as plt
import numpy as np

# GIST数据集
recall_gist_original = [75.6, 83.0, 86.9, 89.4, 91.6, 93.4, 94.6, 95.6, 96.7, 97.7, 98.5, 98.8, 99.0]
qps_gist_original = [133, 90, 68, 55, 46, 39, 34, 31, 23, 20, 15, 14, 12]

recall_gist_topk64 = [75.4, 82.5, 86.3, 88.9, 91.6, 93.2, 94.3, 95.0, 96.3, 97.3, 98.2, 98.6, 98.8]
qps_gist_topk64 = [238, 155, 122, 98, 77, 66, 53, 49, 41, 35, 28, 24, 20]

recall_gist_topk96 = [73.1, 81.7, 85.4, 88.0, 91.2, 92.5, 93.9, 94.5, 95.8, 97.0, 97.9, 98.0, 98.7]
qps_gist_topk96 = [235, 151, 120, 96.7, 74.5, 65.2, 53, 48, 40.1, 35, 27.3, 24, 19.7]

recall_gist_topk48 = [76.0, 82.1, 86.2, 89.1, 91.4, 93.2, 94.5, 95.3, 96.7, 97.6, 98.4, 98.6, 98.9]
qps_gist_topk48 = [230, 149, 118, 95.3, 73.1, 63.8, 52, 47, 39.2, 33.9, 26.6, 23.6, 19.3]

# SIFT数据集
recall_sift_original = [79.4, 89.91, 94.25, 95.99, 96.96, 97.56, 98.54, 98.99, 99.02]
qps_sift_original = [2646, 1455, 1009, 771, 628, 530, 355, 253, 205]

recall_sift_topk64 = [79.0, 89.17, 94.37, 96.1, 97.07, 97.63, 98.56, 98.95, 99.02]
qps_sift_topk64 = [3705, 2445, 1447, 1032, 854, 749, 457, 312, 242]

recall_sift_topk96 = [79.3, 89.0, 94.4, 96.1, 97.0, 97.54, 98.5, 98.88, 99.02]
qps_sift_topk96 = [3664, 2364, 1405, 1017, 804, 712, 433, 301, 221]

recall_sift_topk48 = [79.25, 89.3, 94.4, 96.17, 97.1, 97.66, 98.61, 98.96, 99.02]
qps_sift_topk48 = [3680, 2382, 1401, 1009, 798, 708, 425, 294, 215]

# 绘制图表
def plot_recall_vs_qps():
    # GIST数据集图表
    plt.figure(figsize=(10, 6))
    plt.plot(recall_gist_original, qps_gist_original, 'o-', label='Original IVF')
    plt.plot(recall_gist_topk64, qps_gist_topk64, 's-', label='TopK-64 Clusters')
    plt.plot(recall_gist_topk96, qps_gist_topk96, '^-', label='TopK-96 Clusters')
    plt.plot(recall_gist_topk48, qps_gist_topk48, 'D-', label='TopK-48 Clusters')
    plt.xlabel('Recall (%)')
    plt.ylabel('QPS')
    plt.title('Recall vs QPS for GIST Dataset')
    plt.grid(True)
    plt.legend()
    plt.savefig('recall_vs_qps_gist_top1_few_settings.png', dpi=400)
    plt.close()

    # SIFT数据集图表
    plt.figure(figsize=(10, 6))
    plt.plot(recall_sift_original, qps_sift_original, 'o-', label='Original IVF')
    plt.plot(recall_sift_topk64, qps_sift_topk64, 's-', label='TopK-64 Clusters')
    plt.plot(recall_sift_topk96, qps_sift_topk96, '^-', label='TopK-96 Clusters')
    plt.plot(recall_sift_topk48, qps_sift_topk48, 'D-', label='TopK-48 Clusters')
    plt.xlabel('Recall (%)')
    plt.ylabel('QPS')
    plt.title('Recall vs QPS for SIFT Dataset')
    plt.grid(True)
    plt.legend()
    plt.savefig('recall_vs_qps_sift_top1_few_settings.png', dpi=400)
    plt.close()

# 执行绘图
plot_recall_vs_qps()