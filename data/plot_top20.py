'''
top-20													
dataset	dim	K	nprobe	recall	dco	qps		topk-clusters	nprobe	#re-rank vectors	dco	recall	qps
													
gist	960	1024	10	68.5	14890	129		64	20	4200	5784	67.3	207
			15	77.2	21881	88			25	7500	9307	76.5	137
			20	82.98	28872	66			30	11000	13025	81.8	102
			25	86.5	35799	53.8			35	14000	16152	84.3	84.3
			30	89	42717	45			35	19000	21257	87.98	67.2
			35	91.1	49649	38			40	24000	26352	90.2	56
			40	92.5	56456	34			45	27000	29467	91.55	50.2
			45	93.7	63380	30			50	30000	32741	92.76	45.1
			50	94.87	70195	27			55	35000	37744	93.89	40
			55	95.62	76998	25			60	38000	40938	94.8	36.5
			60	96.35	83819	23			65	41000	44118	95.3	35
			70	97.28	97272	19			75	45000	48369	96.1	31
			80	97.9	110692	17			85	55000	58592	97.2	26
			100	98.6	137297	14			105	65000	69179	98	21.8
			110	99	150511	13.4			110	80000	84320	98.6	19
													
													
sift	128	1024	5	72.6	6464	2563			10	500	1637	67.3	3020
			10	85.6	11773	1410			15	2000	3311	85.4	1584
			15	91.2	17010	976			20	3600	5047	91	1102
			20	94.1	22208	747			25	5000	6557	93.6	875
			25	95.9	27376	607			30	7000	8694	95.6	694
			30	97	32510	511			35	9000	10809	96.7	579
			35	97.8	37617	442			40	11000	12909	97.6	500
			40	98.3	42695	390			45	13000	15019	98.1	433
			45	98.7	47744	348			45	16000	18060	98.4	386
			50	98.9	52776	315			50	20000	22165	98.8	337
			55	99.1	57780	288			55	23000	25246	99	299
													
tiny5m	384	2048	10	68	34178	145			20	7000	10012	67.3	217
			15	75.4	50397	98			25	14000	17577	75.94	137
			20	80	66362	75			30	17000	20787	79.12	116
			25	83.7	82383	60			35	23000	27263	83	91
			30	86.3	98251	50			40	30000	34767	85.5	74
			35	88.4	114013	43			45	38000	43111	88	62
			40	89.9	129714	38			50	46000	51705	89.5	52.5
			45	91.3	145358	34			55	60000	66120	91.2	43.6
			50	92.2	161010	31			60	70000	76730	92.3	38.4
			60	93.8	192202	26			70	80000	87061	93.5	34
			70	94.9	223052	22			80	90000	97848	94.6	30
			80	95.9	253835	19			90	100000	108418	95.4	27
			90	96.5	284207	17			100	120000	129137	96.2	23.3
			100	97.1	314401	15.8			110	140000	149860	96.9	20.7
			110	97.5	344699	14.4			115	160000	170419	97.3	18.6
			120	97.9	374912	13			125	180000	191076	97.7	17
			130	98.2	404754	12.3			135	190000	202020	98.1	16
			140	98.5	434381	11.4			145	200000	212397	98.3	15.1
			160	98.8	493165	10			165	240000	254090	98.8	13
			180	99.1	551590	9			185	260000	275456	99	11.8
'''
import matplotlib.pyplot as plt
import numpy as np

# GIST dataset
recall1_gist = [68.5, 77.2, 82.98, 86.5, 89.0, 91.1, 92.5, 93.7, 94.87, 95.62, 96.35, 97.28, 97.9, 98.6, 99.0]
qps1_gist = [129, 88, 66, 53.8, 45, 38, 34, 30, 27, 25, 23, 19, 17, 14, 13.4]
dco1_gist = [14890, 21881, 28872, 35799, 42717, 49649, 56456, 63380, 70195, 76998, 83819, 97272, 110692, 137297, 150511]

recall2_gist = [67.3, 76.5, 81.8, 84.3, 87.98, 90.2, 91.55, 92.76, 93.89, 94.8, 95.3, 96.1, 97.2, 98.0, 98.6]
qps2_gist = [207, 137, 102, 84.3, 67.2, 56, 50.2, 45.1, 40, 36.5, 35, 31, 26, 21.8, 19]
dco2_gist = [5784, 9307, 13025, 16152, 21257, 26352, 29467, 32741, 37744, 40938, 44118, 48369, 58592, 69179, 84320]

# SIFT dataset
recall1_sift = [72.6, 85.6, 91.2, 94.1, 95.9, 97.0, 97.8, 98.3, 98.7, 98.9, 99.1]
qps1_sift = [2563, 1410, 976, 747, 607, 511, 442, 390, 348, 315, 288]
dco1_sift = [6464, 11773, 17010, 22208, 27376, 32510, 37617, 42695, 47744, 52776, 57780]

recall2_sift = [67.3, 85.4, 91.0, 93.6, 95.6, 96.7, 97.6, 98.1, 98.4, 98.8, 99.0]
qps2_sift = [3020, 1584, 1102, 875, 694, 579, 500, 433, 386, 337, 299]
dco2_sift = [1637, 3311, 5047, 6557, 8694, 10809, 12909, 15019, 18060, 22165, 25246]

# TINY5M dataset
recall1_tiny = [68.0, 75.4, 80.0, 83.7, 86.3, 88.4, 89.9, 91.3, 92.2, 93.8, 94.9, 95.9, 96.5, 97.1, 97.5, 97.9, 98.2, 98.5, 98.8, 99.1]
qps1_tiny = [145, 98, 75, 60, 50, 43, 38, 34, 31, 26, 22, 19, 17, 15.8, 14.4, 13, 12.3, 11.4, 10, 9]
dco1_tiny = [34178, 50397, 66362, 82383, 98251, 114013, 129714, 145358, 161010, 192202, 223052, 253835, 284207, 314401, 344699, 374912, 404754, 434381, 493165, 551590]

recall2_tiny = [67.3, 75.94, 79.12, 83.0, 85.5, 88.0, 89.5, 91.2, 92.3, 93.5, 94.6, 95.4, 96.2, 96.9, 97.3, 97.7, 98.1, 98.3, 98.8, 99.0]
qps2_tiny = [217, 137, 116, 91, 74, 62, 52.5, 43.6, 38.4, 34, 30, 27, 23.3, 20.7, 18.6, 17, 16, 15.1, 13, 11.8]
dco2_tiny = [10012, 17577, 20787, 27263, 34767, 43111, 51705, 66120, 76730, 87061, 97848, 108418, 129137, 149860, 170419, 191076, 202020, 212397, 254090, 275456]

# Plot for GIST dataset
plt.figure(figsize=(10, 6))
plt.plot(recall1_gist, qps1_gist, 'b-o', label='Original')
plt.plot(recall2_gist, qps2_gist, 'r-o', label='New')
plt.xlabel('Recall (%)')
plt.ylabel('QPS (queries/s)')
plt.title('Recall vs QPS Comparison (GIST)')
plt.grid(True)
plt.legend()
plt.savefig('recall_vs_qps_gist_top20.png', dpi=400)
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(recall1_gist, [d/1000 for d in dco1_gist], 'b-o', label='Original')
plt.plot(recall2_gist, [d/1000 for d in dco2_gist], 'r-o', label='New')
plt.xlabel('Recall (%)')
plt.ylabel('Distance Computations (K)')
plt.title('Recall vs Distance Computations (GIST)')
plt.grid(True)
plt.legend()
plt.savefig('recall_vs_dco_gist_top20.png', dpi=400)
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
plt.savefig('recall_vs_qps_sift_top20.png', dpi=400)
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(recall1_sift, [d/1000 for d in dco1_sift], 'b-o', label='Original')
plt.plot(recall2_sift, [d/1000 for d in dco2_sift], 'r-o', label='New')
plt.xlabel('Recall (%)')
plt.ylabel('Distance Computations (K)')
plt.title('Recall vs Distance Computations (SIFT)')
plt.grid(True)
plt.legend()
plt.savefig('recall_vs_dco_sift_top20.png', dpi=400)
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
plt.savefig('recall_vs_qps_tiny5m_top20.png', dpi=400)
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(recall1_tiny, [d/1000 for d in dco1_tiny], 'b-o', label='Original')
plt.plot(recall2_tiny, [d/1000 for d in dco2_tiny], 'r-o', label='New')
plt.xlabel('Recall (%)')
plt.ylabel('Distance Computations (K)')
plt.title('Recall vs Distance Computations (TINY5M)')
plt.grid(True)
plt.legend()
plt.savefig('recall_vs_dco_tiny5m_top20.png', dpi=400)
plt.close()
