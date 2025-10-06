import matplotlib.pyplot as plt

'''
openai-1536	1536	1024	5	80.7	6580	182			15	1000	2250	80.6	327
			10	88	11853	101			20	4000	5532	86.4	163
			15	91	16957	70.6			25	7000	8721	89	110
			20	92.6	22080	54.2			30	10000	11873	90	84.4
			30	94.6	32363	37			40	14000	16055	92	64.53
			50	96.6	52884	22.7			60	20000	22430	93.8	46.54
			60	97.5	63169	19			70	30000	32682	95.6	32.73
			70	97.7	73406	16.5			80	34000	36888	95.8	29.12
			80	97.9	83633	14.3			90	39000	42071	96.1	25.6
			90	98.1	93844	12.9			100	44000	47171	96.5	23
			100	98.3	103991	11.7			105	49000	52298	96.8	20.9
			110	98.5	114142	10.6			115	54000	57525	97.2	19
			120	98.8	124335	9.6			125	59000	62722	97.3	17.29
			130	98.9	134471	9			135	64000	68239	97.6	15.9
													
													
openai-3072	3072	1024	5	78.1	6685	90			15	1000	2251	74.3	166.6
			10	84.9	12053	49.7			20	4500	6113	83	75.7
			15	89.3	17393	34			25	7500	9272	86	52.8
			20	91.2	22680	26.4			30	10000	11948	87.5	42
			30	92.9	33340	18			40	15000	17150	89.3	30.4
			50	95.5	54370	11			60	25000	27634	92.2	19.6
			60	96.2	64872	9.18			70	30000	32832	93.1	16.5
			80	97.2	85943	6.95			90	40000	43184	94.6	12.6
			100	97.9	106966	5.58			105	50000	53490	95.5	10.3
			120	98.4	127894	4.68			125	60000	63968	96.2	8.7
			140	98.8	148744	4			145	70000	74453	96.6	7.46
			150	98.9	159132	3.75			155	80000	84779	97.1	6.5
'''

# OpenAI-1536 dataset
recall1_openai1536 = [80.7, 88.0, 91.0, 92.6, 94.6, 96.6, 97.5, 97.7, 97.9, 98.1, 98.3, 98.5, 98.8, 98.9]
qps1_openai1536 = [182, 101, 70.6, 54.2, 37, 22.7, 19, 16.5, 14.3, 12.9, 11.7, 10.6, 9.6, 9]
dco1_openai1536 = [6580, 11853, 16957, 22080, 32363, 52884, 63169, 73406, 83633, 93844, 103991, 114142, 124335, 134471]

recall2_openai1536 = [80.6, 86.4, 89.0, 90.0, 92.0, 93.8, 95.6, 95.8, 96.1, 96.5, 96.8, 97.2, 97.3, 97.6]
qps2_openai1536 = [327, 163, 110, 84.4, 64.53, 46.54, 32.73, 29.12, 25.6, 23, 20.9, 19, 17.29, 15.9]
dco2_openai1536 = [2250, 5532, 8721, 11873, 16055, 22430, 32682, 36888, 42071, 47171, 52298, 57525, 62722, 68239]

# OpenAI-3072 dataset
recall1_openai3072 = [78.1, 84.9, 89.3, 91.2, 92.9, 95.5, 96.2, 97.2, 97.9, 98.4, 98.8, 98.9]
qps1_openai3072 = [90, 49.7, 34, 26.4, 18, 11, 9.18, 6.95, 5.58, 4.68, 4, 3.75]
dco1_openai3072 = [6685, 12053, 17393, 22680, 33340, 54370, 64872, 85943, 106966, 127894, 148744, 159132]

recall2_openai3072 = [74.3, 83.0, 86.0, 87.5, 89.3, 92.2, 93.1, 94.6, 95.5, 96.2, 96.6, 97.1]
qps2_openai3072 = [166.6, 75.7, 52.8, 42, 30.4, 19.6, 16.5, 12.6, 10.3, 8.7, 7.46, 6.5]
dco2_openai3072 = [2251, 6113, 9272, 11948, 17150, 27634, 32832, 43184, 53490, 63968, 74453, 84779]

# Plot for OpenAI-1536 dataset
plt.figure(figsize=(10, 6))
plt.plot(recall1_openai1536, qps1_openai1536, 'b-o', label='Original')
plt.plot(recall2_openai1536, qps2_openai1536, 'r-o', label='New')
plt.xlabel('Recall (%)')
plt.ylabel('QPS (queries/s)')
plt.title('Recall vs QPS Comparison (OpenAI-1536)')
plt.grid(True)
plt.legend()
plt.savefig('recall_vs_qps_openai1536_top1.png', dpi=400)
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(recall1_openai1536, [d/1000 for d in dco1_openai1536], 'b-o', label='Original')
plt.plot(recall2_openai1536, [d/1000 for d in dco2_openai1536], 'r-o', label='New')
plt.xlabel('Recall (%)')
plt.ylabel('Distance Computations (K)')
plt.title('Recall vs Distance Computations (OpenAI-1536)')
plt.grid(True)
plt.legend()
plt.savefig('recall_vs_dco_openai1536_top1.png', dpi=400)
plt.close()

# Plot for OpenAI-3072 dataset
plt.figure(figsize=(10, 6))
plt.plot(recall1_openai3072, qps1_openai3072, 'b-o', label='Original')
plt.plot(recall2_openai3072, qps2_openai3072, 'r-o', label='New')
plt.xlabel('Recall (%)')
plt.ylabel('QPS (queries/s)')
plt.title('Recall vs QPS Comparison (OpenAI-3072)')
plt.grid(True)
plt.legend()
plt.savefig('recall_vs_qps_openai3072_top1.png', dpi=400)
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(recall1_openai3072, [d/1000 for d in dco1_openai3072], 'b-o', label='Original')
plt.plot(recall2_openai3072, [d/1000 for d in dco2_openai3072], 'r-o', label='New')
plt.xlabel('Recall (%)')
plt.ylabel('Distance Computations (K)')
plt.title('Recall vs Distance Computations (OpenAI-3072)')
plt.grid(True)
plt.legend()
plt.savefig('recall_vs_dco_openai3072_top1.png', dpi=400)
plt.close()