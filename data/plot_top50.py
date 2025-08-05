'''
top-50													
dataset	dim	K	nprobe	recall	dco	qps		topk-clusters	nprobe	#re-rank vectors	dco	recall	qps
													
gist	960	1024	10	64.8	14890	129		64	20	5000	6608	64.9	185
			15	73.8	21881	88			25	7500	9307	72	137
			20	79.8	28872	66			30	12000	14029	79.1	96
			25	83.7	35799	53			35	15000	17216	82.4	80
			30	86.7	42717	45			40	20000	22401	86	63
			35	88.9	49649	38.8			45	24000	26515	88.3	54
			40	90.6	56546	34			50	28000	30628	90.1	48
			45	92	63380	30.4			55	32000	34921	91.5	42.2
			50	93.2	70195	27.4			60	35000	37993	92.5	39.2
			55	94.2	76998	25			65	38000	41093	93.3	36.3
			60	95	83819	23			70	41000	44221	94	33.8
			70	96.2	97272	19.8			80	46000	49434	95	30.3
			80	97.1	110692	17.4			90	55000	58662	96.2	26
			95	97.9	130696	14.7			100	65000	68985	97.1	22.4
			110	98.5	150511	12.8			115	75000	79475	98	19.6
			125	99	170082	11.3			130	85000	89857	98.5	17.3
													
sift	128	1024	5	69	6464	2535			10	800	1977	66.8	2530
			10	82.8	11773	1399			15	2000	3311	81	1560
			15	89.1	17010	973			20	3600	5047	88	1108
			20	92.6	22208	747			25	5000	6557	91.3	862
			25	94.7	27376	607			30	7000	8694	94	683
			30	96.1	32510	516			35	9000	10809	95.5	574
			35	97	37617	441			40	12000	13936	96.8	474
			40	97.7	42695	388			45	15000	17050	97.6	400
			50	98.6	52776	315			50	18000	20166	98.1	348
			55	98.8	57780	287			55	22000	24258	98.6	300
			60	99	62785	264			60	26000	28352	98.9	270
													
													
tiny5m	384	2048	15	71.8	50397	98			25	14000	17577	71.3	137
			20	76.8	66362	75			30	17000	20787	74.57	116
			25	80.7	82383	60			35	23000	27263	78.9	91.7
			30	83.5	98251	50.7			40	30000	34767	82	74.2
			35	85.8	114013	43.7			45	38000	43111	84.7	61.8
			40	87.6	129714	38			50	46000	51705	86.8	53
			45	89.1	145358	34.2			55	60000	66120	89	43
			50	90.3	161010	30.9			60	70000	76730	90.4	38.3
			60	92.3	192202	25.9			70	80000	87061	91.92	33.8
			75	94.3	238454	20.9			85	100000	108057	93.87	27.5
			90	95.6	284207	17.5			100	130000	139309	95.4	22.3
			120	97.3	374912	13.3			125	180000	191076	97.1	16.8
			140	98	434381	11.4			145	210000	222787	97.8	14.5
			160	98.5	493165	10.1			165	250000	264340	98.4	12.5
			200	99.07	609615	8			200	300000	316630	98.9	10.5
'''
import matplotlib.pyplot as plt
import numpy as np

# GIST dataset
recall1_gist = [64.8, 73.8, 79.8, 83.7, 86.7, 88.9, 90.6, 92.0, 93.2, 94.2, 95.0, 96.2, 97.1, 97.9, 98.5, 99.0]
qps1_gist = [129, 88, 66, 53, 45, 38.8, 34, 30.4, 27.4, 25, 23, 19.8, 17.4, 14.7, 12.8, 11.3]
dco1_gist = [14890, 21881, 28872, 35799, 42717, 49649, 56546, 63380, 70195, 76998, 83819, 97272, 110692, 130696, 150511, 170082]

recall2_gist = [64.9, 72.0, 79.1, 82.4, 86.0, 88.3, 90.1, 91.5, 92.5, 93.3, 94.0, 95.0, 96.2, 97.1, 98.0, 98.5]
qps2_gist = [185, 137, 96, 80, 63, 54, 48, 42.2, 39.2, 36.3, 33.8, 30.3, 26, 22.4, 19.6, 17.3]
dco2_gist = [6608, 9307, 14029, 17216, 22401, 26515, 30628, 34921, 37993, 41093, 44221, 49434, 58662, 68985, 79475, 89857]

# SIFT dataset
recall1_sift = [69.0, 82.8, 89.1, 92.6, 94.7, 96.1, 97.0, 97.7, 98.6, 98.8, 99.0]
qps1_sift = [2535, 1399, 973, 747, 607, 516, 441, 388, 315, 287, 264]
dco1_sift = [6464, 11773, 17010, 22208, 27376, 32510, 37617, 42695, 52776, 57780, 62785]

recall2_sift = [66.8, 81.0, 88.0, 91.3, 94.0, 95.5, 96.8, 97.6, 98.1, 98.6, 98.9]
qps2_sift = [2530, 1560, 1108, 862, 683, 574, 474, 400, 348, 300, 270]
dco2_sift = [1977, 3311, 5047, 6557, 8694, 10809, 13936, 17050, 20166, 24258, 28352]

# TINY5M dataset
recall1_tiny = [71.8, 76.8, 80.7, 83.5, 85.8, 87.6, 89.1, 90.3, 92.3, 94.3, 95.6, 97.3, 98.0, 98.5, 99.07]
qps1_tiny = [98, 75, 60, 50.7, 43.7, 38, 34.2, 30.9, 25.9, 20.9, 17.5, 13.3, 11.4, 10.1, 8]
dco1_tiny = [50397, 66362, 82383, 98251, 114013, 129714, 145358, 161010, 192202, 238454, 284207, 374912, 434381, 493165, 609615]

recall2_tiny = [71.3, 74.57, 78.9, 82.0, 84.7, 86.8, 89.0, 90.4, 91.92, 93.87, 95.4, 97.1, 97.8, 98.4, 98.9]
qps2_tiny = [137, 116, 91.7, 74.2, 61.8, 53, 43, 38.3, 33.8, 27.5, 22.3, 16.8, 14.5, 12.5, 10.5]
dco2_tiny = [17577, 20787, 27263, 34767, 43111, 51705, 66120, 76730, 87061, 108057, 139309, 191076, 222787, 264340, 316630]

# Plot for GIST dataset
plt.figure(figsize=(10, 6))
plt.plot(recall1_gist, qps1_gist, 'b-o', label='Original')
plt.plot(recall2_gist, qps2_gist, 'r-o', label='New')
plt.xlabel('Recall (%)')
plt.ylabel('QPS (queries/s)')
plt.title('Recall vs QPS Comparison (GIST)')
plt.grid(True)
plt.legend()
plt.savefig('recall_vs_qps_gist_top50.png', dpi=400)
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(recall1_gist, [d/1000 for d in dco1_gist], 'b-o', label='Original')
plt.plot(recall2_gist, [d/1000 for d in dco2_gist], 'r-o', label='New')
plt.xlabel('Recall (%)')
plt.ylabel('Distance Computations (K)')
plt.title('Recall vs Distance Computations (GIST)')
plt.grid(True)
plt.legend()
plt.savefig('recall_vs_dco_gist_top50.png', dpi=400)
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
plt.savefig('recall_vs_qps_sift_top50.png', dpi=400)
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(recall1_sift, [d/1000 for d in dco1_sift], 'b-o', label='Original')
plt.plot(recall2_sift, [d/1000 for d in dco2_sift], 'r-o', label='New')
plt.xlabel('Recall (%)')
plt.ylabel('Distance Computations (K)')
plt.title('Recall vs Distance Computations (SIFT)')
plt.grid(True)
plt.legend()
plt.savefig('recall_vs_dco_sift_top50.png', dpi=400)
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
plt.savefig('recall_vs_qps_tiny5m_top50.png', dpi=400)
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(recall1_tiny, [d/1000 for d in dco1_tiny], 'b-o', label='Original')
plt.plot(recall2_tiny, [d/1000 for d in dco2_tiny], 'r-o', label='New')
plt.xlabel('Recall (%)')
plt.ylabel('Distance Computations (K)')
plt.title('Recall vs Distance Computations (TINY5M)')
plt.grid(True)
plt.legend()
plt.savefig('recall_vs_dco_tiny5m_top50.png', dpi=400)
plt.close()
