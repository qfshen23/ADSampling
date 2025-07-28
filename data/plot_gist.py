import matplotlib.pyplot as plt
import numpy as np

# Data for original method
recall1 = [71.05, 77.31, 81.5, 84.78, 87.2, 89.11, 90.63, 92.02, 93.16, 94.07, 94.82, 95.44, 96.0, 96.469, 96.85, 97.18, 97.5, 97.75, 97.99, 98.22]
qps1 = [89, 67, 53, 45, 38, 33.8, 30.9, 27.6, 25.8, 22.6, 21, 19.5, 18.9, 17.8, 16.1, 15.2, 14.8, 13.8, 13.3, 12.6]
dco1 = [21881, 28872, 35799, 42717, 49649, 56546, 63380, 70195, 76998, 83819, 90559, 97272, 104017, 110692, 117369, 124059, 130696, 137297, 143911, 150511]

# Data for new method
recall2 = [71.42, 77.35, 80.64, 84.72, 86.93, 88.83, 90.14, 91.57, 92.55, 93.25, 93.69, 94.62, 95.23, 95.72, 96.587, 96.981, 97.3, 97.6, 97.857, 98.2]
qps2 = [130, 100, 87, 63, 55, 49.3, 45.8, 39.3, 37.6, 33.6, 31.3, 30.4, 27.58, 25.7, 23.6, 21.1, 20.3, 18.9, 18, 16.4]
dco2 = [12216, 15163, 19051, 24871, 28887, 32952, 35934, 40938, 45121, 48105, 51400, 55414, 59743, 63711, 73681, 78991, 83971, 89124, 94321, 104190]

# Plot Recall vs QPS
plt.figure(figsize=(10, 6))
plt.plot(recall1, qps1, 'b-o', label='Original')
plt.plot(recall2, qps2, 'r-o', label='New')
plt.xlabel('Recall (%)')
plt.ylabel('QPS (queries/s)')
plt.title('Recall vs QPS Comparison (GIST)')
plt.grid(True)
plt.legend()
plt.savefig('recall_vs_qps_gist.png', dpi=400)
plt.close()

# Plot Distance Computations vs Recall
plt.figure(figsize=(10, 6))
plt.plot(recall1, [d/1000 for d in dco1], 'b-o', label='Original')
plt.plot(recall2, [d/1000 for d in dco2], 'r-o', label='New')
plt.xlabel('Recall (%)')
plt.ylabel('Distance Computations (K)')
plt.title('Recall vs Distance Computations (GIST)')
plt.grid(True)
plt.legend()
plt.savefig('recall_vs_dco_gist.png', dpi=400)
plt.close()