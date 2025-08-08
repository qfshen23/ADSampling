import matplotlib.pyplot as plt
import numpy as np

'''
top-10														
dataset	dim	K	nprobe	recall	dco	qps	#atom op	topk-clusters	nprobe	#re-rank vectors	dco	recall	qps	#atom op
														
gist	960	1024	10	70.4	14890	132	42883200	64	20	4000	5567	69.46	240	16254816
			15	79.2	21881	90	63017280		25	7000	8773	78.1	161	25599952
			20	84.8	28872	68	83151360		30	10000	11965	83	122	34904768
			25	88.2	35799	55	103101120		35	14000	16152	87.46	94	47074160
			30	90.4	42717	46	123024960		40	17000	19273	89.56	79	56173328
			35	92.4	49649	39	142989120		45	22000	24481	91.87	64	71283280
			45	94.8	63380	31	182534400		55	26000	28783	93.65	54	83892736
			50	95.7	70195	28	202161600		60	32000	34897	95	46	101610096
			60	96.8	83819	23	241398720		70	36000	39284	95.88	40	114462640
			70	97.7	97272	20	280143360		80	42000	45425	96.8	36	132363968
			90	98.68	124059	15	357289920		100	60000	64065	98.2	26	186475760
			110	99.2	150511	12	433471680		120	70000	74485	98.76	22	216908592
														
sift	128	1024	5	75	6464	2641	2482176		10	800	1977	77.45	2979	846208
			10	87.3	11773	1454	4520832		15	2000	3311	87.9	1886	1443408
			15	92.4	17010	1009	6531840		20	3000	4413	91.8	1449	1950368
			20	95.1	22208	772	8527872		25	5000	6557	94.8	1047	2856832
			25	96.6	27376	625	10512384		30	7500	9201	96.6	790	3954816
			30	97.6	32510	530	12483840		35	10000	11828	97.64	673	5045728
			35	98.2	37617	458	14444928		40	13000	14953	98.26	546	6327440
			40	98.6	42695	404	16394880		45	16000	18060	98.7	468	7601776
			45	98.9	47744	361	18333696		45	18000	20082	98.78	435	8459008
			50	99.1	52776	327	20265984		50	21000	23179	99	381	9728768
														
tiny5m	384	2048	10	72.2	34178	149	39373056		20	7000	10012	72.2	258	12561984
			15	79	50397	101	58057344		25	14000	17577	79.65	161	21795872
			20	82.8	66362	77	76449024		30	17000	20787	82.54	135	26004672
			25	86.1	82383	62	94905216		35	23000	27263	85.66	107	33977696
			30	88.3	98251	51	113185152		40	30000	34767	87.92	87	43130080
			35	90.1	114013	44	131342976		45	38000	43111	89.79	73	53246752
			40	91.4	129714	39	149430528		50	46000	51705	91.19	61	63649472
			50	93.4	161010	31	185483520		60	70000	76730	93.31	47	93479744
			70	95.6	223052	22	256955904		80	100000	108168	95.56	32	131681664
			90	96.9	284207	18	327406464		100	120000	129137	96.68	27	157794912
			100	97.4	314401	16	362189952		110	140000	149860	97.35	23	182634016
			120	98.1	374912	13	431898624		130	160000	171151	97.91	21	209097600
			140	98.7	434381	11	500406912		145	180000	192108	98.44	18	235143072
			160	99	493165	10	568126080		160	220000	233564	98.82	15.7	284781472
'''

# GIST dataset
recall1_gist = [70.4, 79.2, 84.8, 88.2, 90.4, 92.4, 94.8, 95.7, 96.8, 97.7, 98.68, 99.2]
qps1_gist = [132, 90, 68, 55, 46, 39, 31, 28, 23, 20, 15, 12]
dco1_gist = [14890, 21881, 28872, 35799, 42717, 49649, 63380, 70195, 83819, 97272, 124059, 150511]
atom1_gist = [42883200, 63017280, 83151360, 103101120, 123024960, 142989120, 182534400, 202161600, 241398720, 280143360, 357289920, 433471680]

recall2_gist = [69.46, 78.1, 83.0, 87.46, 89.56, 91.87, 93.65, 95.0, 95.88, 96.8, 98.2, 98.76]
qps2_gist = [240, 161, 122, 94, 79, 64, 54, 46, 40, 36, 26, 22]
dco2_gist = [5567, 8773, 11965, 16152, 19273, 24481, 28783, 34897, 39284, 45425, 64065, 74485]
atom2_gist = [16254816, 25599952, 34904768, 47074160, 56173328, 71283280, 83892736, 101610096, 114462640, 132363968, 186475760, 216908592]

# SIFT dataset
recall1_sift = [75.0, 87.3, 92.4, 95.1, 96.6, 97.6, 98.2, 98.6, 98.9, 99.1]
qps1_sift = [2641, 1454, 1009, 772, 625, 530, 458, 404, 361, 327]
dco1_sift = [6464, 11773, 17010, 22208, 27376, 32510, 37617, 42695, 47744, 52776]
atom1_sift = [2482176, 4520832, 6531840, 8527872, 10512384, 12483840, 14444928, 16394880, 18333696, 20265984]

recall2_sift = [77.45, 87.9, 91.8, 94.8, 96.6, 97.64, 98.26, 98.7, 98.78, 99.0]
qps2_sift = [2979, 1886, 1449, 1047, 790, 673, 546, 468, 435, 381]
dco2_sift = [1977, 3311, 4413, 6557, 9201, 11828, 14953, 18060, 20082, 23179]
atom2_sift = [846208, 1443408, 1950368, 2856832, 3954816, 5045728, 6327440, 7601776, 8459008, 9728768]

# TINY5M dataset
recall1_tiny = [72.2, 79.0, 82.8, 86.1, 88.3, 90.1, 91.4, 93.4, 95.6, 96.9, 97.4, 98.1, 98.7, 99.0]
qps1_tiny = [149, 101, 77, 62, 51, 44, 39, 31, 22, 18, 16, 13, 11, 10]
dco1_tiny = [34178, 50397, 66362, 82383, 98251, 114013, 129714, 161010, 223052, 284207, 314401, 374912, 434381, 493165]
atom1_tiny = [39373056, 58057344, 76449024, 94905216, 113185152, 131342976, 149430528, 185483520, 256955904, 327406464, 362189952, 431898624, 500406912, 568126080]

recall2_tiny = [72.2, 79.65, 82.54, 85.66, 87.92, 89.79, 91.19, 93.31, 95.56, 96.68, 97.35, 97.91, 98.44, 98.82]
qps2_tiny = [258, 161, 135, 107, 87, 73, 61, 47, 32, 27, 23, 21, 18, 15.7]
dco2_tiny = [10012, 17577, 20787, 27263, 34767, 43111, 51705, 76730, 108168, 129137, 149860, 171151, 192108, 233564]
atom2_tiny = [12561984, 21795872, 26004672, 33977696, 43130080, 53246752, 63649472, 93479744, 131681664, 157794912, 182634016, 209097600, 235143072, 284781472]

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

plt.figure(figsize=(10, 6))
plt.plot(recall1_gist, [a/1e6 for a in atom1_gist], 'b-o', label='Original')
plt.plot(recall2_gist, [a/1e6 for a in atom2_gist], 'r-o', label='New')
plt.xlabel('Recall (%)')
plt.ylabel('Atomic Operations (M)')
plt.title('Recall vs Atomic Operations (GIST)')
plt.grid(True)
plt.legend()
plt.savefig('recall_vs_atom_gist_top10.png', dpi=400)
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

plt.figure(figsize=(10, 6))
plt.plot(recall1_sift, [a/1e6 for a in atom1_sift], 'b-o', label='Original')
plt.plot(recall2_sift, [a/1e6 for a in atom2_sift], 'r-o', label='New')
plt.xlabel('Recall (%)')
plt.ylabel('Atomic Operations (M)')
plt.title('Recall vs Atomic Operations (SIFT)')
plt.grid(True)
plt.legend()
plt.savefig('recall_vs_atom_sift_top10.png', dpi=400)
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

plt.figure(figsize=(10, 6))
plt.plot(recall1_tiny, [a/1e6 for a in atom1_tiny], 'b-o', label='Original')
plt.plot(recall2_tiny, [a/1e6 for a in atom2_tiny], 'r-o', label='New')
plt.xlabel('Recall (%)')
plt.ylabel('Atomic Operations (M)')
plt.title('Recall vs Atomic Operations (TINY5M)')
plt.grid(True)
plt.legend()
plt.savefig('recall_vs_atom_tiny5m_top10.png', dpi=400)
plt.close()