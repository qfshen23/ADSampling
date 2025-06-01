import matplotlib.pyplot as plt

# SIFT dataset
sift_recall = [67.0, 81.9, 88.8, 92.6, 94.9, 96.4, 97.4]
sift_qps = [1455, 850, 603, 466, 387, 334, 286]
# Point without refinement
sift_recall_wo = [16.0]
sift_qps_wo = [15551]

# GIST dataset
gist_recall = [48.4, 65.2, 74.7, 80.7, 85.1, 88.3, 90.7]
gist_qps = [131, 67, 45, 34, 27, 23, 20]
# Point without refinement
gist_recall_wo = [7.9]
gist_qps_wo = [1888]

# MSONG dataset
msong_recall = [82.5, 93.2, 96.6, 98.0, 98.7, 99.1, 99.4]
msong_qps = [288, 148, 101, 77, 64, 54, 46]
# Point without refinement
msong_recall_wo = [18.5]
msong_qps_wo = [4446]

# TINY5M dataset
tiny5m_recall = [64.4, 78.4, 85.3, 89.4, 92.1, 93.9]
tiny5m_qps = [81, 41, 27, 21, 17, 14]
# Point without refinement
tiny5m_recall_wo = [3.9]
tiny5m_qps_wo = [2490]

# SIFT10M dataset
sift10m_recall = [77.0, 87.5, 91.9, 94.4, 95.9, 96.9]
sift10m_qps = [379, 204, 137, 104, 85, 73]
# Point without refinement
sift10m_recall_wo = [6.8]
sift10m_qps_wo = [4808]

plt.figure(figsize=(10, 6))

plt.plot(sift_recall, sift_qps, marker='o', label='SIFT')
plt.scatter(sift_recall_wo, sift_qps_wo, marker='*', color='C0', s=100)

plt.plot(gist_recall, gist_qps, marker='s', label='GIST')
plt.scatter(gist_recall_wo, gist_qps_wo, marker='*', color='C1', s=100)

plt.plot(msong_recall, msong_qps, marker='^', label='MSONG')
plt.scatter(msong_recall_wo, msong_qps_wo, marker='*', color='C2', s=100)

plt.plot(tiny5m_recall, tiny5m_qps, marker='v', label='TINY5M')
plt.scatter(tiny5m_recall_wo, tiny5m_qps_wo, marker='*', color='C3', s=100)

plt.plot(sift10m_recall, sift10m_qps, marker='D', label='SIFT10M')
plt.scatter(sift10m_recall_wo, sift10m_qps_wo, marker='*', color='C4', s=100)

plt.xlabel('Recall (%)')
plt.ylabel('QPS (queries/sec)')
plt.grid(True)
plt.legend()
plt.title('QPS vs Recall Trade-off for Different Datasets (Top-1K)')

plt.yscale('log')
plt.savefig('ivf_recall_qps_1k.png', dpi=400)