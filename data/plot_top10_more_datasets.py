import matplotlib.pyplot as plt

# openai-1536 dataset
recall1_1536 = [79.6, 86.0, 88.9, 90.8, 93.1, 95.5, 96.1, 97.09, 97.61, 98.21, 98.47, 98.66]
qps1_1536 = [184, 102, 71.4, 55, 37.2, 22.7, 19, 14.3, 11.6, 9.71, 8.39, 7.81]
dco1_1536 = [6580, 11853, 16957, 22080, 32363, 52884, 63169, 83633, 103991, 124335, 144521, 154574]

recall2_1536 = [73.04, 85.41, 87.44, 88.78, 90.59, 92.77, 94.01, 95.15, 96.21, 96.89, 97.24, 97.47]
qps2_1536 = [296.151, 110.829, 85.7731, 69.9158, 51.2942, 34.7255, 28.2176, 21.9608, 17.8327, 15.3166, 13.4296, 12.6338]
dco2_1536 = [2250, 7617, 10261, 12865, 18100, 27466, 34657, 45039, 55462, 66977, 74246, 79405]

# openai-3072 dataset
recall1_3072 = [77.06, 83.42, 86.87, 88.93, 90.47, 91.52, 93.84, 94.56, 95.86, 96.75, 97.35, 97.86, 98.07]
qps1_3072 = [89.24, 49.67, 34.3, 26.29, 21.32, 17.89, 10.97, 9.2, 6.94, 5.57, 4.65, 3.97, 3.73]
dco1_3072 = [6685, 12053, 17393, 22680, 28010, 33340, 54370, 64872, 85943, 106966, 127894, 148744, 159132]

recall2_3072 = [69.89, 82.48, 84.36, 86.42, 87.73, 88.52, 91.39, 92.43, 93.69, 94.74, 95.42, 96.13, 96.35]
qps2_3072 = [157.087, 58.4202, 46.7633, 36.5273, 29.9641, 26.7192, 17.3788, 14.9292, 11.5561, 9.4647, 8.00579, 6.87193, 6.44801]
dco2_3072 = [2251, 7648, 9794, 12914, 16065, 18202, 29562, 34782, 45277, 55613, 66085, 77467, 82670]

# Plot for openai-1536 dataset
plt.figure(figsize=(10, 6))
plt.plot(recall1_1536, qps1_1536, 'b-o', label='Original')
plt.plot(recall2_1536, qps2_1536, 'r-o', label='New')
plt.xlabel('Recall (%)')
plt.ylabel('QPS (queries/s)')
plt.title('Recall vs QPS Comparison (openai-1536)')
plt.grid(True)
plt.legend()
plt.savefig('recall_vs_qps_openai1536_top10.png', dpi=400)
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(recall1_1536, [d/1000 for d in dco1_1536], 'b-o', label='Original')
plt.plot(recall2_1536, [d/1000 for d in dco2_1536], 'r-o', label='New')
plt.xlabel('Recall (%)')
plt.ylabel('Distance Computations (K)')
plt.title('Recall vs Distance Computations (openai-1536)')
plt.grid(True)
plt.legend()
plt.savefig('recall_vs_dco_openai1536_top10.png', dpi=400)
plt.close()

# Plot for openai-3072 dataset
plt.figure(figsize=(10, 6))
plt.plot(recall1_3072, qps1_3072, 'b-o', label='Original')
plt.plot(recall2_3072, qps2_3072, 'r-o', label='New')
plt.xlabel('Recall (%)')
plt.ylabel('QPS (queries/s)')
plt.title('Recall vs QPS Comparison (openai-3072)')
plt.grid(True)
plt.legend()
plt.savefig('recall_vs_qps_openai3072_top10.png', dpi=400)
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(recall1_3072, [d/1000 for d in dco1_3072], 'b-o', label='Original')
plt.plot(recall2_3072, [d/1000 for d in dco2_3072], 'r-o', label='New')
plt.xlabel('Recall (%)')
plt.ylabel('Distance Computations (K)')
plt.title('Recall vs Distance Computations (openai-3072)')
plt.grid(True)
plt.legend()
plt.savefig('recall_vs_dco_openai3072_top10.png', dpi=400)
plt.close()