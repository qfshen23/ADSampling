import matplotlib.pyplot as plt

# SIFT dataset
sift_nprobe = [10, 20, 30, 40, 50, 60, 70]
sift_recall = [67.0, 81.9, 88.8, 92.6, 94.9, 96.4, 97.4]

# GIST dataset
gist_nprobe = [10, 20, 30, 40, 50, 60, 70]
gist_recall = [48.4, 65.2, 74.7, 80.7, 85.1, 88.3, 90.7]

# MSONG dataset
msong_nprobe = [10, 20, 30, 40, 50, 60, 70]
msong_recall = [82.5, 93.2, 96.6, 98.0, 98.7, 99.1, 99.4]

# TINY5M dataset
tiny5m_nprobe = [20, 40, 60, 80, 100, 120]
tiny5m_recall = [64.4, 78.4, 85.3, 89.4, 92.1, 93.9]

# SIFT10M dataset
sift10m_nprobe = [20, 40, 60, 80, 100, 120]
sift10m_recall = [77.0, 87.5, 91.9, 94.4, 95.9, 96.9]

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
plt.title('Recall vs #nprobe for Different Datasets (Top-1K)')

plt.savefig('ivf_distribution_top1K.png', dpi=400)
