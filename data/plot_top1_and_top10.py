import matplotlib.pyplot as plt
import numpy as np

# Data for top-10
# GIST dataset
recall_gist_top10 = [69.46, 78.1, 83.0, 87.46, 89.56, 91.87, 93.65, 95.0, 95.88, 96.8, 98.2, 98.76]
a_gist_top10 = [221856, 333712, 445568, 556400, 667088, 778000, 997696, 1106736, 1324720, 1539968, 1968560, 2391792]
b_gist_top10 = [16032960, 25266240, 34459200, 46517760, 55506240, 70505280, 82895040, 100503360, 113137920, 130824000, 184507200, 214516800]

# SIFT dataset
recall_sift_top10 = [77.45, 87.9, 91.8, 94.8, 96.6, 97.64, 98.26, 98.7, 98.78, 99.0]
a_sift_top10 = [87040, 171984, 255776, 338944, 421632, 503776, 585488, 666736, 747520, 828032]
b_sift_top10 = [759168, 1271424, 1694592, 2517888, 3533184, 4541952, 5741952, 6935040, 7711488, 8900736]

# TINY5M dataset
recall_tiny_top10 = [72.2, 79.65, 82.54, 85.66, 87.92, 89.79, 91.19, 93.31, 95.56, 96.68, 97.35, 97.91, 98.44, 98.82]
a_tiny_top10 = [1028160, 1547168, 2058048, 2570720, 3078496, 3582880, 4085312, 5086784, 7072128, 9029088, 9995296, 11931648, 13834656, 15715744]
b_tiny_top10 = [11533824, 20248704, 23946624, 31406976, 40051584, 49663872, 59564160, 88392960, 124609536, 148765824, 172638720, 197165952, 221308416, 269065728]

# Data for top-1
# GIST dataset
recall_gist_top1 = [75.4, 82.5, 86.3, 88.9, 91.6, 93.2, 94.3, 95.0, 96.3, 97.3, 98.2, 98.6, 98.8]
a_gist_top1 = [221856, 333712, 445568, 556400, 667088, 778000, 888352, 997696, 1324720, 1539968, 1968560, 2180368, 2600336]
b_gist_top1 = [16032960, 26804160, 34459200, 43701120, 58547520, 67824000, 85659840, 94331520, 113137920, 130824000, 169894080, 193933440, 243829440]

# SIFT dataset
recall_sift_top1 = [79.0, 89.17, 94.37, 96.1, 97.07, 97.63, 98.56, 98.95, 99.02]
a_sift_top1 = [87040, 171984, 255776, 338944, 421632, 503776, 747520, 1067792, 1306000]
b_sift_top1 = [584064, 892416, 1694592, 2517888, 3338496, 3749760, 6963456, 10558464, 16391424]

# Plot stacked bar charts
plt.figure(figsize=(10, 6))
x = np.arange(len(recall_gist_top10))
plt.bar(x, a_gist_top10, label='overlap ratio', color='skyblue')
plt.bar(x, b_gist_top10, bottom=a_gist_top10, label='distance', color='lightcoral')
plt.xticks(x, [f'{r:.1f}' for r in recall_gist_top10])
plt.xlabel('Recall (%)')
plt.ylabel('Atomic Operations')
plt.title('Recall vs Atomic Operations - GIST (Top-10)')
plt.legend()
plt.grid(True)
plt.savefig('recall_vs_atom_gist_top10_stacked.png', dpi=400)
plt.close()

plt.figure(figsize=(10, 6))
x = np.arange(len(recall_sift_top10))
plt.bar(x, a_sift_top10, label='overlap ratio', color='skyblue')
plt.bar(x, b_sift_top10, bottom=a_sift_top10, label='distance', color='lightcoral')
plt.xticks(x, [f'{r:.1f}' for r in recall_sift_top10])
plt.xlabel('Recall (%)')
plt.ylabel('Atomic Operations')
plt.title('Recall vs Atomic Operations - SIFT (Top-10)')
plt.legend()
plt.grid(True)
plt.savefig('recall_vs_atom_sift_top10_stacked.png', dpi=400)
plt.close()

plt.figure(figsize=(10, 6))
x = np.arange(len(recall_tiny_top10))
plt.bar(x, a_tiny_top10, label='overlap ratio', color='skyblue')
plt.bar(x, b_tiny_top10, bottom=a_tiny_top10, label='distance', color='lightcoral')
plt.xticks(x, [f'{r:.1f}' for r in recall_tiny_top10])
plt.xlabel('Recall (%)')
plt.ylabel('Atomic Operations')
plt.title('Recall vs Atomic Operations - TINY5M (Top-10)')
plt.legend()
plt.grid(True)
plt.savefig('recall_vs_atom_tiny_top10_stacked.png', dpi=400)
plt.close()

plt.figure(figsize=(10, 6))
x = np.arange(len(recall_gist_top1))
plt.bar(x, a_gist_top1, label='overlap ratio', color='skyblue')
plt.bar(x, b_gist_top1, bottom=a_gist_top1, label='distance', color='lightcoral')
plt.xticks(x, [f'{r:.1f}' for r in recall_gist_top1])
plt.xlabel('Recall (%)')
plt.ylabel('Atomic Operations')
plt.title('Recall vs Atomic Operations - GIST (Top-1)')
plt.legend()
plt.grid(True)
plt.savefig('recall_vs_atom_gist_top1_stacked.png', dpi=400)
plt.close()

plt.figure(figsize=(10, 6))
x = np.arange(len(recall_sift_top1))
plt.bar(x, a_sift_top1, label='overlap ratio', color='skyblue')
plt.bar(x, b_sift_top1, bottom=a_sift_top1, label='distance', color='lightcoral')
plt.xticks(x, [f'{r:.1f}' for r in recall_sift_top1])
plt.xlabel('Recall (%)')
plt.ylabel('Atomic Operations')
plt.title('Recall vs Atomic Operations - SIFT (Top-1)')
plt.legend()
plt.grid(True)
plt.savefig('recall_vs_atom_sift_top1_stacked.png', dpi=400)
plt.close()