import matplotlib.pyplot as plt
import numpy as np

# GIST数据集
recall_gist_original = [70.4, 79.2, 84.8, 88.2, 90.4, 92.4, 94.8, 95.7, 96.8, 97.7, 98.68, 99.2]
qps_gist_original = [132, 90, 68, 55, 46, 39, 31, 28, 23, 20, 15, 12]

recall_gist_topk64 = [69.46, 78.1, 83.0, 87.46, 89.56, 91.87, 93.65, 95.0, 95.88, 96.8, 98.2, 98.76]
qps_gist_topk64 = [240, 161, 122, 94, 79, 64, 54, 46, 40, 36, 26, 22]

recall_gist_topk96 = [68.65, 77.0, 81.83, 86.39, 88.46, 91.04, 92.71, 94.4, 95.37, 96.34, 97.86, 98.41]
qps_gist_topk96 = [237, 160, 121, 92.39, 78.4, 63.26, 54, 45.3, 40.4, 35.1, 25.78, 22.1]

recall_gist_topk48 = [70.4, 78.69, 83.93, 87.68, 89.61, 92.0, 93.82, 95.21, 96.19, 97.14, 98.44, 98.94]
qps_gist_topk48 = [232, 155, 118, 90, 76.7, 61.9, 53, 44.5, 39.7, 34.2, 25, 21.5]

# SIFT数据集
recall_sift_original = [75.0, 87.3, 92.4, 95.1, 96.6, 97.6, 98.2, 98.6, 98.9, 99.1]
qps_sift_original = [2641, 1454, 1009, 772, 625, 530, 458, 404, 361, 327]

recall_sift_topk64 = [77.45, 87.9, 91.8, 94.8, 96.6, 97.64, 98.26, 98.7, 98.78, 99.0]
qps_sift_topk64 = [2979, 1886, 1449, 1047, 790, 673, 546, 468, 435, 381]

recall_sift_topk96 = [77.7, 87.96, 91.93, 94.9, 96.62, 97.66, 98.29, 98.73, 98.8, 99.06]
qps_sift_topk96 = [2966, 1824, 1394, 998, 758, 613, 511, 437.7, 411, 363]

recall_sift_topk48 = [77.0, 87.57, 91.55, 94.67, 96.45, 97.52, 98.19, 98.63, 98.72, 99.0]
qps_sift_topk48 = [2990, 1846, 1402, 1016, 774, 630, 523, 449, 421, 373]

# TINY5M数据集
recall_tiny_original = [72.2, 79.0, 82.8, 86.1, 88.3, 90.1, 91.4, 93.4, 95.6, 96.9, 97.4, 98.1, 98.7, 99.0]
qps_tiny_original = [149, 101, 77, 62, 51, 44, 39, 31, 22, 18, 16, 13, 11, 10]

recall_tiny_topk64 = [72.2, 79.65, 82.54, 85.66, 87.92, 89.79, 91.19, 93.31, 95.56, 96.68, 97.35, 97.91, 98.44, 98.82]
qps_tiny_topk64 = [258, 161, 135, 107, 87, 73, 61, 47, 32, 27, 23, 21, 18, 15.7]

recall_tiny_topk96 = [71.47, 79.37, 81.87, 85.0, 87.3, 89.42, 90.81, 93.12, 95.31, 96.44, 97.2, 97.75, 98.27, 98.75]
qps_tiny_topk96 = [252, 158, 135, 106.6, 86.59, 71.6, 61.3, 43.78, 31.78, 26.4, 23, 20.28, 18.2, 15.1]

recall_tiny_topk48 = [73.0, 80.32, 82.8, 86.03, 88.31, 90.23, 91.46, 93.52, 95.61, 96.73, 97.45, 98.0, 98.48, 98.87]
qps_tiny_topk48 = [246, 154, 130, 103, 84, 69.5, 59.4, 42.7, 31, 25.7, 22.46, 19.67, 17.67, 14.86]

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
    plt.savefig('recall_vs_qps_gist_top10_few_settings.png', dpi=400)
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
    plt.savefig('recall_vs_qps_sift_top10_few_settings.png', dpi=400)
    plt.close()

    # TINY5M数据集图表
    plt.figure(figsize=(10, 6))
    plt.plot(recall_tiny_original, qps_tiny_original, 'o-', label='Original IVF')
    plt.plot(recall_tiny_topk64, qps_tiny_topk64, 's-', label='TopK-64 Clusters')
    plt.plot(recall_tiny_topk96, qps_tiny_topk96, '^-', label='TopK-96 Clusters')
    plt.plot(recall_tiny_topk48, qps_tiny_topk48, 'D-', label='TopK-48 Clusters')
    plt.xlabel('Recall (%)')
    plt.ylabel('QPS')
    plt.title('Recall vs QPS for TINY5M Dataset')
    plt.grid(True)
    plt.legend()
    plt.savefig('recall_vs_qps_tiny_top10_few_settings.png', dpi=400)
    plt.close()

# 执行绘图
plot_recall_vs_qps()