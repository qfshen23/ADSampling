import matplotlib.pyplot as plt
import numpy as np

# GIST dataset
recall1_gist = [75.6, 83.0, 86.9, 89.4, 91.6, 93.4, 94.6, 95.6, 96.7, 97.7, 98.5, 98.8, 99.0]
qps1_gist = [133, 90, 68, 55, 46, 39, 34, 31, 23, 20, 15, 14, 12]
dco1_gist = [14890, 21881, 28872, 35799, 42717, 49649, 56546, 63380, 83819, 97272, 124059, 137297, 163545]

recall2_gist = [75.4, 82.5, 86.3, 88.9, 91.6, 93.2, 94.3, 95.0, 96.3, 97.3, 98.2, 98.6, 98.8]
qps2_gist = [238, 155, 122, 98, 77, 66, 53, 49, 41, 35, 28, 24, 20]
dco2_gist = [5567, 9307, 11965, 15174, 20329, 23550, 29743, 32754, 39284, 45425, 58991, 67338, 84663]

# SIFT dataset
recall1_sift = [79.4, 89.91, 94.25, 95.99, 96.96, 97.56, 98.54, 98.99, 99.02]
qps1_sift = [2646, 1455, 1009, 771, 628, 530, 355, 253, 205]
dco1_sift = [6464, 11773, 17010, 22208, 27376, 32510, 47744, 67761, 82649]

recall2_sift = [79.0, 89.17, 94.37, 96.1, 97.07, 97.63, 98.56, 98.95, 99.02]
qps2_sift = [3705, 2445, 1447, 1032, 854, 749, 457, 312, 242]
dco2_sift = [1521, 2324, 4413, 6557, 8694, 9765, 18134, 27496, 42686]

# Plot for GIST dataset
plt.figure(figsize=(10, 6))
plt.plot(recall1_gist, qps1_gist, 'b-o', label='Original')
plt.plot(recall2_gist, qps2_gist, 'r-o', label='New')
plt.xlabel('Recall (%)')
plt.ylabel('QPS (queries/s)')
plt.title('Recall vs QPS Comparison (GIST)')
plt.grid(True)
plt.legend()
plt.savefig('recall_vs_qps_gist_top1.png', dpi=400)
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(recall1_gist, [d/1000 for d in dco1_gist], 'b-o', label='Original')
plt.plot(recall2_gist, [d/1000 for d in dco2_gist], 'r-o', label='New')
plt.xlabel('Recall (%)')
plt.ylabel('Distance Computations (K)')
plt.title('Recall vs Distance Computations (GIST)')
plt.grid(True)
plt.legend()
plt.savefig('recall_vs_dco_gist_top1.png', dpi=400)
plt.close()

# Plot for SIFT dataset
plt.figure(figsize=(10, 6))
plt.plot(recall1_sift, qps1_sift, 'b-o', label='Original')
plt.plot(recall2_sift, qps2_sift, 'r-o', label='New')
plt.xlabel('Recall (%)')
plt.ylabel('QPS (queries/s)')
plt.title('Recall vs QPS Comparison (SIFT)')
plt.grid(True)
plt.legend()
plt.savefig('recall_vs_qps_sift_top1.png', dpi=400)
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(recall1_sift, [d/1000 for d in dco1_sift], 'b-o', label='Original')
plt.plot(recall2_sift, [d/1000 for d in dco2_sift], 'r-o', label='New')
plt.xlabel('Recall (%)')
plt.ylabel('Distance Computations (K)')
plt.title('Recall vs Distance Computations (SIFT)')
plt.grid(True)
plt.legend()
plt.savefig('recall_vs_dco_sift_top1.png', dpi=400)
plt.close()