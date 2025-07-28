import matplotlib.pyplot as plt
import numpy as np

# Data for original method
recall1 = [83.5, 86.07, 88.08, 89.7, 91.03, 92.14, 93.06, 93.86, 95.11, 96.07, 96.45, 97.35, 97.6, 98.0, 98.17]
qps1 = [1224, 1063, 902, 809, 741, 673, 621, 576, 504, 463, 446, 367, 352, 324, 312]
dco1 = [13872, 15969, 18055, 20140, 22208, 24277, 26341, 28404, 32510, 36598, 38635, 44721, 46738, 50763, 52776]

# Data for new method
recall2 = [83.07, 86.21, 88.12, 89.33, 91.0, 92.3, 93.07, 93.91, 94.6, 95.68, 96.12, 96.95, 97.22, 97.67, 97.96]
qps2 = [1340, 1136, 1028, 994, 862, 812, 737, 684, 624, 552, 528, 454, 456, 406, 386]
dco2 = [4912, 6009, 6653, 7101, 8098, 9099, 10149, 11139, 12162, 14172, 15202, 17237, 18259, 20290, 22332]

# Plot Recall vs QPS
plt.figure(figsize=(10, 6))
plt.plot(recall1, qps1, 'b-o', label='Original')
plt.plot(recall2, qps2, 'r-o', label='New')
plt.xlabel('Recall (%)')
plt.ylabel('QPS (queries/s)')
plt.title('Recall vs QPS Comparison (SIFT)')
plt.grid(True)
plt.legend()
plt.savefig('recall_vs_qps_sift.png', dpi=400)
plt.close()

# Plot Distance Computations vs Recall
plt.figure(figsize=(10, 6))
plt.plot(recall1, [d/1000 for d in dco1], 'b-o', label='Original')
plt.plot(recall2, [d/1000 for d in dco2], 'r-o', label='New')
plt.xlabel('Recall (%)')
plt.ylabel('Distance Computations (K)')
plt.title('Recall vs Distance Computations (SIFT)')
plt.grid(True)
plt.legend()
plt.savefig('recall_vs_dco_sift.png', dpi=400)
plt.close()