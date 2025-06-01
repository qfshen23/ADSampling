import matplotlib.pyplot as plt

# SIFT dataset
sift_nprobe = [20, 40, 60, 80, 100, 120]
sift_recall = [59.4, 78.0, 87.0, 92.0, 94.9, 96.7]

# GIST dataset
gist_nprobe = [30, 40, 60, 80, 100, 120]
gist_recall = [55.6, 63.4, 74.6, 81.8, 86.6, 90.0]

# MSONG dataset
msong_nprobe = [30, 40, 60, 80, 100, 120]
msong_recall = [86.2, 91.0, 95.6, 97.6, 98.6, 99.1]

# TINY5M dataset
tiny5m_nprobe = [30, 60, 90, 120, 150, 170]
tiny5m_recall = [58.6, 74.3, 82.4, 87.4, 90.7, 92.3]

# SIFT10M dataset
sift10m_nprobe = [40, 80, 120, 160, 200, 220]
sift10m_recall = [73.8, 85.7, 91.0, 93.9, 95.6, 96.3]

plt.figure(figsize=(10, 6))

plt.plot(sift_nprobe, sift_recall, marker='o', label='SIFT')
plt.plot(gist_nprobe, gist_recall, marker='s', label='GIST')
plt.plot(msong_nprobe, msong_recall, marker='^', label='MSONG')
plt.plot(tiny5m_nprobe, tiny5m_recall, marker='*', label='TINY5M')
plt.plot(sift10m_nprobe, sift10m_recall, marker='D', label='SIFT10M')

plt.xlabel('#nprobe')
plt.ylabel('Recall (%)')
plt.grid(True)
plt.legend()
plt.title('Recall vs #nprobe for Different Datasets (Top-10K)')

plt.savefig('ivf_distribution_top10K.png', dpi=400)