'''
dataset	D	K	#nprobe	Recall	QPS
					
sift	128	1	20	95.99	993
		10	20	95.14	995
		100	30	95.1	671
		1000	50	94.95	385
		10000	100	94.9	160
					
gist	960	1	20	87.5	68
		10	25	88.25	55
		100	40	89.1	34
		1000	60	88.3	23
		10000	100	86	14
					
msong	420	1	1	100	1988
		10	10	96.14	284
		100	20	97.6	146
		1000	30	96.6	103
		10000	60	95.6	50
					
tiny5m	384	1	1	100	1002
		10	40	91.4	40
		100	50	89	33
		1000	80	89.4	21
		10000	140	89.7	12
					
sift10m	128	1	30	96.5	269
		10	40	96.7	212
		100	60	96.1	140
		1000	100	95.9	86
		10000	220	96.3	38
'''
import matplotlib.pyplot as plt
import numpy as np

# Data for each dataset
K_values = [1, 10, 100, 1000, 10000]
log_K = np.log10(K_values)

# Plot SIFT
fig, ax1 = plt.subplots(figsize=(8, 6))
ax2 = ax1.twinx()

sift_recall = [95.99, 95.14, 95.1, 94.95, 94.9]
sift_qps = [993, 995, 671, 385, 160]

ax1.bar(log_K, sift_qps, color='skyblue', alpha=0.7, label='QPS')
ax2.plot(log_K, sift_recall, color='red', linewidth=2, marker='o', label='Recall')

ax1.set_xlabel('K (Power of 10)')
ax1.set_ylabel('QPS')
ax2.set_ylabel('Recall (%)')
ax1.set_title('SIFT')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

plt.tight_layout()
plt.savefig('ivf_large_K_sift.png', dpi=400)
plt.close()

# Plot GIST
fig, ax1 = plt.subplots(figsize=(8, 6))
ax2 = ax1.twinx()

gist_recall = [87.5, 88.25, 89.1, 88.3, 86]
gist_qps = [68, 55, 34, 23, 14]

ax1.bar(log_K, gist_qps, color='skyblue', alpha=0.7, label='QPS')
ax2.plot(log_K, gist_recall, color='red', linewidth=2, marker='o', label='Recall')

ax1.set_xlabel('K (Power of 10)')
ax1.set_ylabel('QPS')
ax2.set_ylabel('Recall (%)')
ax1.set_title('GIST')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

plt.tight_layout()
plt.savefig('ivf_large_K_gist.png', dpi=400)
plt.close()

# Plot MSONG
fig, ax1 = plt.subplots(figsize=(8, 6))
ax2 = ax1.twinx()

msong_recall = [100, 96.14, 97.6, 96.6, 95.6]
msong_qps = [1988, 284, 146, 103, 50]

ax1.bar(log_K, msong_qps, color='skyblue', alpha=0.7, label='QPS')
ax2.plot(log_K, msong_recall, color='red', linewidth=2, marker='o', label='Recall')

ax1.set_xlabel('K (Power of 10)')
ax1.set_ylabel('QPS')
ax2.set_ylabel('Recall (%)')
ax1.set_title('MSONG')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

plt.tight_layout()
plt.savefig('ivf_large_K_msong.png', dpi=400)
plt.close()

# Plot Tiny5m
fig, ax1 = plt.subplots(figsize=(8, 6))
ax2 = ax1.twinx()

tiny_recall = [100, 91.4, 89, 89.4, 89.7]
tiny_qps = [1002, 40, 33, 21, 12]

ax1.bar(log_K, tiny_qps, color='skyblue', alpha=0.7, label='QPS')
ax2.plot(log_K, tiny_recall, color='red', linewidth=2, marker='o', label='Recall')

ax1.set_xlabel('K (Power of 10)')
ax1.set_ylabel('QPS')
ax2.set_ylabel('Recall (%)')
ax1.set_title('Tiny5m')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

plt.tight_layout()
plt.savefig('ivf_large_K_tiny5m.png', dpi=400)
plt.close()

# Plot SIFT10m
fig, ax1 = plt.subplots(figsize=(8, 6))
ax2 = ax1.twinx()

sift10m_recall = [96.5, 96.7, 96.1, 95.9, 96.3]
sift10m_qps = [269, 212, 140, 86, 38]

ax1.bar(log_K, sift10m_qps, color='skyblue', alpha=0.7, label='QPS')
ax2.plot(log_K, sift10m_recall, color='red', linewidth=2, marker='o', label='Recall')

ax1.set_xlabel('K (Power of 10)')
ax1.set_ylabel('QPS')
ax2.set_ylabel('Recall (%)')
ax1.set_title('SIFT10m')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

plt.tight_layout()
plt.savefig('ivf_large_K_sift10m.png', dpi=400)
plt.close()
