import matplotlib.pyplot as plt
import numpy as np

'''
top-1														
dataset	dim	K	nprobe	recall	dco	qps	#atom op	topk-clusters	nprobe	#re-rank vectors	dco	recall	qps	#atom op
														
gist	960	1024	10	75.6	14890	133	42883200	64	20	4000	5567	75.4	238	16254816
			15	83	21881	90	63017280		25	7500	9307	82.5	155	27137872
			20	86.9	28872	68	83151360		30	10000	11965	86.3	122	34904768
			25	89.4	35799	55	103101120		35	13000	15174	88.9	98	44257520
			30	91.6	42717	46	123024960		40	18000	20329	91.6	77	59214608
			35	93.4	49649	39	142989120		45	21000	23550	93.2	66	68602000
			40	94.6	56546	34	162852480		50	27000	29743	94.3	53	86548192
			45	95.6	63380	31	182534400		55	30000	32754	95	49	95329216
			60	96.7	83819	23	241398720		70	36000	39284	96.3	41	114462640
			70	97.7	97272	20	280143360		80	42000	45425	97.3	35	132363968
			90	98.5	124059	15	357289920		100	55000	58991	98.2	28	171862640
			100	98.8	137297	14	395415360		110	63000	67338	98.6	24	196113808
			120	99	163545	12	471009600		120	80000	84663	98.8	20	246429776
														
sift	128	1024	5	79.4	6464	2646	2482176		10	400	1521	79	3705	671104
			10	89.91	11773	1455	4520832		15	1100	2324	89.17	2445	1064400
			15	94.25	17010	1009	6531840		20	3000	4413	94.37	1447	1950368
			20	95.99	22208	771	8527872		25	5000	6557	96.1	1032	2856832
			25	96.96	27376	628	10512384		30	7000	8694	97.07	854	3760128
			30	97.56	32510	530	12483840		35	8000	9765	97.63	749	4253536
			45	98.54	47744	355	18333696		50	16000	18134	98.56	457	7710976
			65	98.99	67761	253	26020224		70	25000	27496	98.95	312	11626256
			80	99.02	82649	205	31737216		80	40000	42686	99.02	242	17697424
'''

# GIST dataset
recall1_gist = [75.6, 83.0, 86.9, 89.4, 91.6, 93.4, 94.6, 95.6, 96.7, 97.7, 98.5, 98.8, 99.0]
qps1_gist = [133, 90, 68, 55, 46, 39, 34, 31, 23, 20, 15, 14, 12]
dco1_gist = [14890, 21881, 28872, 35799, 42717, 49649, 56546, 63380, 83819, 97272, 124059, 137297, 163545]
atom1_gist = [42883200, 63017280, 83151360, 103101120, 123024960, 142989120, 162852480, 182534400, 241398720, 280143360, 357289920, 395415360, 471009600]

recall2_gist = [75.4, 82.5, 86.3, 88.9, 91.6, 93.2, 94.3, 95.0, 96.3, 97.3, 98.2, 98.6, 98.8]
qps2_gist = [238, 155, 122, 98, 77, 66, 53, 49, 41, 35, 28, 24, 20]
dco2_gist = [5567, 9307, 11965, 15174, 20329, 23550, 29743, 32754, 39284, 45425, 58991, 67338, 84663]
atom2_gist = [16254816, 27137872, 34904768, 44257520, 59214608, 68602000, 86548192, 95329216, 114462640, 132363968, 171862640, 196113808, 246429776]

# SIFT dataset
recall1_sift = [79.4, 89.91, 94.25, 95.99, 96.96, 97.56, 98.54, 98.99, 99.02]
qps1_sift = [2646, 1455, 1009, 771, 628, 530, 355, 253, 205]
dco1_sift = [6464, 11773, 17010, 22208, 27376, 32510, 47744, 67761, 82649]
atom1_sift = [2482176, 4520832, 6531840, 8527872, 10512384, 12483840, 18333696, 26020224, 31737216]

recall2_sift = [79.0, 89.17, 94.37, 96.1, 97.07, 97.63, 98.56, 98.95, 99.02]
qps2_sift = [3705, 2445, 1447, 1032, 854, 749, 457, 312, 242]
dco2_sift = [1521, 2324, 4413, 6557, 8694, 9765, 18134, 27496, 42686]
atom2_sift = [671104, 1064400, 1950368, 2856832, 3760128, 4253536, 7710976, 11626256, 17697424]

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

plt.figure(figsize=(10, 6))
plt.plot(recall1_gist, [a/1000000 for a in atom1_gist], 'b-o', label='Original')
plt.plot(recall2_gist, [a/1000000 for a in atom2_gist], 'r-o', label='New')
plt.xlabel('Recall (%)')
plt.ylabel('Atomic Operations (M)')
plt.title('Recall vs Atomic Operations (GIST)')
plt.grid(True)
plt.legend()
plt.savefig('recall_vs_atom_gist_top1.png', dpi=400)
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

plt.figure(figsize=(10, 6))
plt.plot(recall1_sift, [a/1000000 for a in atom1_sift], 'b-o', label='Original')
plt.plot(recall2_sift, [a/1000000 for a in atom2_sift], 'r-o', label='New')
plt.xlabel('Recall (%)')
plt.ylabel('Atomic Operations (M)')
plt.title('Recall vs Atomic Operations (SIFT)')
plt.grid(True)
plt.legend()
plt.savefig('recall_vs_atom_sift_top1.png', dpi=400)
plt.close()