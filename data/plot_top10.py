import matplotlib.pyplot as plt
import numpy as np

# GIST dataset
recall1_gist = [70.4, 79.2, 84.8, 88.2, 90.4, 92.4, 94.8, 95.7, 96.8, 97.7, 98.68, 99.2]
qps1_gist = [132, 90, 68, 55, 46, 39, 31, 28, 23, 20, 15, 12]
dco1_gist = [14890, 21881, 28872, 35799, 42717, 49649, 63380, 70195, 83819, 97272, 124059, 150511]

recall2_gist = [69.46, 78.1, 83.0, 87.46, 89.56, 91.87, 93.65, 95.0, 95.88, 96.8, 98.2, 98.76]
qps2_gist = [240, 161, 122, 94, 79, 64, 54, 46, 40, 36, 26, 22]
dco2_gist = [5567, 8773, 11965, 16152, 19273, 24481, 28783, 34897, 39284, 45425, 64065, 74485]

# SIFT dataset
recall1_sift = [75.0, 87.3, 92.4, 95.1, 96.6, 97.6, 98.2, 98.6, 98.9, 99.1]
qps1_sift = [2641, 1454, 1009, 772, 625, 530, 458, 404, 361, 327]
dco1_sift = [6464, 11773, 17010, 22208, 27376, 32510, 37617, 42695, 47744, 52776]

recall2_sift = [77.45, 87.9, 91.8, 94.8, 96.6, 97.64, 98.26, 98.7, 98.78, 99.0]
qps2_sift = [2979, 1886, 1449, 1047, 790, 673, 546, 468, 435, 381]
dco2_sift = [1977, 3311, 4413, 6557, 9201, 11828, 14953, 18060, 20082, 23179]

# TINY5M dataset
recall1_tiny = [72.2, 79.0, 82.8, 86.1, 88.3, 90.1, 91.4, 93.4, 95.6, 96.9, 97.4, 98.1, 98.7, 99.0]
qps1_tiny = [149, 101, 77, 62, 51, 44, 39, 31, 22, 18, 16, 13, 11, 10]
dco1_tiny = [34178, 50397, 66362, 82383, 98251, 114013, 129714, 161010, 223052, 284207, 314401, 374912, 434381, 493165]

recall2_tiny = [72.2, 79.65, 82.54, 85.66, 87.92, 89.79, 91.19, 93.31, 95.56, 96.68, 97.35, 97.91, 98.44, 98.82]
qps2_tiny = [258, 161, 135, 107, 87, 73, 61, 47, 32, 27, 23, 21, 18, 15.7]
dco2_tiny = [10012, 17577, 20787, 27263, 34767, 43111, 51705, 76730, 108168, 129137, 149860, 171151, 192108, 233564]

# Plot for GIST dataset
plt.figure(figsize=(10, 6))
plt.plot(recall1_gist, qps1_gist, 'b-o', label='Original')
plt.plot(recall2_gist, qps2_gist, 'r-o', label='New')
plt.xlabel('Recall (%)')
plt.ylabel('QPS (queries/s)')
plt.title('Recall vs QPS Comparison (GIST)')
plt.grid(True)
plt.legend()
plt.savefig('recall_vs_qps_gist_top10.png', dpi=400)
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(recall1_gist, [d/1000 for d in dco1_gist], 'b-o', label='Original')
plt.plot(recall2_gist, [d/1000 for d in dco2_gist], 'r-o', label='New')
plt.xlabel('Recall (%)')
plt.ylabel('Distance Computations (K)')
plt.title('Recall vs Distance Computations (GIST)')
plt.grid(True)
plt.legend()
plt.savefig('recall_vs_dco_gist_top10.png', dpi=400)
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
plt.savefig('recall_vs_qps_sift_top10.png', dpi=400)
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(recall1_sift, [d/1000 for d in dco1_sift], 'b-o', label='Original')
plt.plot(recall2_sift, [d/1000 for d in dco2_sift], 'r-o', label='New')
plt.xlabel('Recall (%)')
plt.ylabel('Distance Computations (K)')
plt.title('Recall vs Distance Computations (SIFT)')
plt.grid(True)
plt.legend()
plt.savefig('recall_vs_dco_sift_top10.png', dpi=400)
plt.close()

# Plot for TINY5M dataset
plt.figure(figsize=(10, 6))
plt.plot(recall1_tiny, qps1_tiny, 'b-o', label='Original')
plt.plot(recall2_tiny, qps2_tiny, 'r-o', label='New')
plt.xlabel('Recall (%)')
plt.ylabel('QPS (queries/s)')
plt.title('Recall vs QPS Comparison (TINY5M)')
plt.grid(True)
plt.legend()
plt.savefig('recall_vs_qps_tiny5m_top10.png', dpi=400)
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(recall1_tiny, [d/1000 for d in dco1_tiny], 'b-o', label='Original')
plt.plot(recall2_tiny, [d/1000 for d in dco2_tiny], 'r-o', label='New')
plt.xlabel('Recall (%)')
plt.ylabel('Distance Computations (K)')
plt.title('Recall vs Distance Computations (TINY5M)')
plt.grid(True)
plt.legend()
plt.savefig('recall_vs_dco_tiny5m_top10.png', dpi=400)
plt.close()