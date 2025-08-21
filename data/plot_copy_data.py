import matplotlib.pyplot as plt
import numpy as np

# 提取数据
nprobes = [45, 55, 60, 70, 80]
time1 = [537281, 543097, 542769, 539530, 541113]  # Centroids DCO
time2 = [297711, 384310, 411703, 461061, 528602]  # Candidate Selection
time3 = [2.82904e+07, 3.44164e+07, 4.33687e+07, 4.86438e+07, 5.50858e+07]  # Memory Copy
time4 = [1.21174e+07, 1.44411e+07, 1.75901e+07, 1.97236e+07, 2.29177e+07]  # Exact DCO

# 设置柱状图宽度
width = 0.15

# 设置x轴位置
x = np.arange(len(nprobes))

# 创建图形
fig, ax = plt.subplots(figsize=(12, 6))

# 绘制四组柱状图
rects1 = ax.bar(x - width*1.5, time1, width, label='Centroids DCO')
rects2 = ax.bar(x - width/2, time2, width, label='Candidate Selection')
rects3 = ax.bar(x + width/2, time3, width, label='Memory Copy')
rects4 = ax.bar(x + width*1.5, time4, width, label='Exact DCO')

# 设置图形标题和标签
ax.set_ylabel('Time (microseconds)')
ax.set_title('Time Breakdown for Different nprobe Values (GIST Top-10)')
ax.set_xticks(x)
ax.set_xticklabels([f'nprobe={n}' for n in nprobes])
ax.legend()

# 自动调整布局
plt.tight_layout()

# 显示图形
plt.savefig('copy_data_gist_top10.png', dpi=400)
plt.close()
