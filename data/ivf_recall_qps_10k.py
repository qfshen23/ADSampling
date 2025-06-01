import matplotlib.pyplot as plt

# SIFT dataset
sift_recall = [59.4, 78.0, 87.0, 92.0, 94.9, 96.7]
sift_qps = [468, 298, 228, 187, 159, 139]
# Point without refinement
sift_recall_wo = [38.8]
sift_qps_wo = [5848]

# GIST dataset
gist_recall = [55.6, 63.4, 74.6, 81.8, 86.6, 90.0]
gist_qps = [43, 33, 22, 17, 14, 11]
# Point without refinement
gist_recall_wo = [23.1]
gist_qps_wo = [1609]

# MSONG dataset
msong_recall = [86.2, 91.0, 95.6, 97.6, 98.6, 99.1]
msong_qps = [90, 71, 49, 38, 31, 26]
# Point without refinement
msong_recall_wo = [44.0]
msong_qps_wo = [3114]

# TINY5M dataset
tiny5m_recall = [58.6, 74.3, 82.4, 87.4, 90.7, 92.3]
tiny5m_qps = [49, 26, 18, 13, 11, 10]
# Point without refinement
tiny5m_recall_wo = [15.2]
tiny5m_qps_wo = [2084]

# SIFT10M dataset
sift10m_recall = [73.8, 85.7, 91.0, 93.9, 95.6, 96.3]
sift10m_qps = [152, 89, 64, 49, 41, 37]
# Point without refinement
sift10m_recall_wo = [24.6]
sift10m_qps_wo = [3245]

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
plt.title('QPS vs Recall Trade-off for Different Datasets (Top-10K)')

plt.yscale('log')
plt.savefig('ivf_recall_qps_10k.png', dpi=400)