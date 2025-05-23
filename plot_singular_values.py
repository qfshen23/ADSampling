import matplotlib.pyplot as plt
import numpy as np

# Data for 10% pruned clusters
pruned_10_percent = {
    64: [1.1113, 0.3899, 0.3281, 0.2375, 0.2417, 0.2376],
    256: [1.4344, 0.5771, 0.441, 0.376, 0.3744, 0.3276, 0.3007, 0.2236, 0.2358, 0.2244, 0.2036, 0.2037, 0.1971, 0.1716, 0.1632, 0.1667, 0.164, 0.1445, 0.1457, 0.1458, 0.1389, 0.1163, 0.1188, 0.1159, 0.1027],
    512: [1.5947, 0.5767, 0.4926, 0.4587, 0.4033, 0.3748, 0.3359, 0.282, 0.2764, 0.2627, 0.283, 0.2523, 0.2081, 0.2105, 0.1975, 0.1909, 0.1944, 0.1991, 0.1778, 0.1801, 0.1492, 0.1568, 0.1347, 0.1339, 0.144, 0.1385, 0.1302, 0.1131, 0.1313, 0.1313, 0.1164, 0.1019, 0.0931, 0.1052, 0.0966, 0.0945, 0.0948, 0.0847, 0.0706, 0.0859, 0.0713, 0.0844, 0.0601, 0.075, 0.0732, 0.0738, 0.0641, 0.0603, 0.06, 0.0573, 0.063],
    1024: [1.7068, 0.6332, 0.5299, 0.4743, 0.4148, 0.3879, 0.3977, 0.3265, 0.3225, 0.2938, 0.2911, 0.276, 0.2912, 0.241, 0.2347, 0.2434, 0.2304, 0.2206, 0.2215, 0.1796, 0.1892, 0.1825, 0.1744, 0.1597, 0.1599, 0.165, 0.1565, 0.1276, 0.1415, 0.1454, 0.1442, 0.1294, 0.133, 0.1256, 0.1126, 0.1052, 0.1183, 0.115, 0.1013, 0.1023, 0.1084, 0.1119, 0.0942, 0.0982, 0.1078, 0.0952, 0.0963, 0.0845, 0.0908, 0.0736, 0.0851, 0.0781, 0.081, 0.071, 0.0677, 0.0834, 0.0678, 0.0639, 0.0684, 0.0729, 0.0704, 0.0658, 0.0649, 0.0604, 0.061, 0.0544, 0.0645, 0.0602, 0.0533, 0.051, 0.0583, 0.0489, 0.0529, 0.0552, 0.0534, 0.0517, 0.0439, 0.044, 0.0555, 0.0475, 0.0459, 0.0424, 0.0483, 0.044, 0.0386, 0.0392, 0.0409, 0.0394, 0.0366, 0.0419, 0.0418, 0.0431, 0.0347, 0.0386, 0.0342, 0.0374, 0.0367, 0.032, 0.0375, 0.0346, 0.0305, 0.0301]
}

# Data for 20% pruned clusters
pruned_20_percent = {
    64: [1.1113, 0.3899, 0.3281, 0.2375, 0.2417, 0.2376, 0.2198, 0.1498, 0.1904, 0.1397, 0.1345, 0.1511],
    256: [1.4344, 0.5771, 0.441, 0.376, 0.3744, 0.3276, 0.3007, 0.2236, 0.2358, 0.2244, 0.2036, 0.2037, 0.1971, 0.1716, 0.1632, 0.1667, 0.164, 0.1445, 0.1457, 0.1458, 0.1389, 0.1163, 0.1188, 0.1159, 0.1027, 0.0955, 0.0913, 0.0805, 0.0839, 0.0806, 0.0717, 0.088, 0.0711, 0.0765, 0.061, 0.0567, 0.0677, 0.0538, 0.0568, 0.0461, 0.0609, 0.0455, 0.0523, 0.0367, 0.0388, 0.0427, 0.0428, 0.0394, 0.0352, 0.0297, 0.0331],
    512: [1.5947, 0.5767, 0.4926, 0.4587, 0.4033, 0.3748, 0.3359, 0.282, 0.2764, 0.2627, 0.283, 0.2523, 0.2081, 0.2105, 0.1975, 0.1909, 0.1944, 0.1991, 0.1778, 0.1801, 0.1492, 0.1568, 0.1347, 0.1339, 0.144, 0.1385, 0.1302, 0.1131, 0.1313, 0.1313, 0.1164, 0.1019, 0.0931, 0.1052, 0.0966, 0.0945, 0.0948, 0.0847, 0.0706, 0.0859, 0.0713, 0.0844, 0.0601, 0.075, 0.0732, 0.0738, 0.0641, 0.0603, 0.06, 0.0573, 0.063, 0.0574, 0.0474, 0.0575, 0.0508, 0.0478, 0.0469, 0.0511, 0.0408, 0.0442, 0.0458, 0.0407, 0.0391, 0.042, 0.0449, 0.0377, 0.0368, 0.0365, 0.0286, 0.038, 0.0312, 0.031, 0.0277, 0.0338, 0.0327, 0.0394, 0.0281, 0.026, 0.0301, 0.03, 0.0288, 0.0283, 0.0251, 0.0253, 0.0255, 0.0293, 0.0289, 0.022, 0.0256, 0.0233, 0.0192, 0.0167, 0.0264, 0.0213, 0.0171, 0.0219, 0.0166, 0.0193, 0.0142, 0.019, 0.0204, 0.0181],
    1024: [1.7068, 0.6332, 0.5299, 0.4743, 0.4148, 0.3879, 0.3977, 0.3265, 0.3225, 0.2938, 0.2911, 0.276, 0.2912, 0.241, 0.2347, 0.2434, 0.2304, 0.2206, 0.2215, 0.1796, 0.1892, 0.1825, 0.1744, 0.1597, 0.1599, 0.165, 0.1565, 0.1276, 0.1415, 0.1454, 0.1442, 0.1294, 0.133, 0.1256, 0.1126, 0.1052, 0.1183, 0.115, 0.1013, 0.1023, 0.1084, 0.1119, 0.0942, 0.0982, 0.1078, 0.0952, 0.0963, 0.0845, 0.0908, 0.0736, 0.0851, 0.0781, 0.081, 0.071, 0.0677, 0.0834, 0.0678, 0.0639, 0.0684, 0.0729, 0.0704, 0.0658, 0.0649, 0.0604, 0.061, 0.0544, 0.0645, 0.0602, 0.0533, 0.051, 0.0583, 0.0489, 0.0529, 0.0552, 0.0534, 0.0517, 0.0439, 0.044, 0.0555, 0.0475, 0.0459, 0.0424, 0.0483, 0.044, 0.0386, 0.0392, 0.0409, 0.0394, 0.0366, 0.0419, 0.0418, 0.0431, 0.0347, 0.0386, 0.0342, 0.0374, 0.0367, 0.032, 0.0375, 0.0346, 0.0305, 0.0301, 0.0402, 0.0272, 0.0306, 0.0339, 0.0258, 0.0261, 0.0241, 0.0301, 0.0257, 0.0272, 0.0264, 0.026, 0.0299, 0.0243, 0.0228, 0.0263, 0.0257, 0.0271, 0.0226, 0.0266, 0.0239, 0.0243, 0.0233, 0.0224, 0.0226, 0.0248, 0.0201, 0.0211, 0.0191, 0.0161, 0.0268, 0.0239, 0.0181, 0.0224, 0.0205, 0.0191, 0.0185, 0.0208, 0.0226, 0.0186, 0.0163, 0.0171, 0.0136, 0.0224, 0.0229, 0.0163, 0.0178, 0.0199, 0.0165, 0.0195, 0.0157, 0.0157, 0.0168, 0.0111, 0.0158, 0.0139, 0.0181, 0.0145, 0.0143, 0.014, 0.0145, 0.0112, 0.0144, 0.0136, 0.0144, 0.0128, 0.0098, 0.0129, 0.014, 0.0185, 0.0149, 0.0126, 0.0126, 0.0127, 0.0168, 0.0099, 0.0141, 0.0142, 0.0077, 0.0116, 0.013, 0.0099, 0.0085, 0.0121, 0.008, 0.0116, 0.01, 0.0101, 0.0105, 0.0105, 0.0102, 0.0105, 0.0112, 0.0088, 0.0101, 0.0082, 0.0075, 0.0129, 0.0124, 0.0082, 0.0084, 0.01]
}

# Sums for each K
sums_10_percent = {
    64: 2.5461,
    256: 6.0735,
    512: 9.7057,
    1024: 14.9834
}

sums_20_percent = {
    64: 3.5314,
    256: 8.0135,
    512: 12.0414,
    1024: 17.9984
}

# Create plots for 10% pruned clusters
plt.figure(figsize=(20, 15))
plt.suptitle("Singular Values for 10% Pruned Clusters", fontsize=20)

for i, k in enumerate([64, 256, 512, 1024]):
    plt.subplot(2, 2, i+1)
    values = pruned_10_percent[k]
    ranks = list(range(len(values)))
    
    plt.bar(ranks, values, width=0.7)
    plt.title(f"K={k}, Sum={sums_10_percent[k]:.4f}", fontsize=16)
    plt.xlabel("Rank", fontsize=14)
    plt.ylabel("Singular Value", fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Add a text annotation for the first few singular values
    for j in range(min(5, len(values))):
        plt.text(j, values[j], f"{values[j]:.4f}", 
                 ha='center', va='bottom', fontsize=10)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("singular_values_10percent.png", dpi=300)

# Create plots for 20% pruned clusters
plt.figure(figsize=(20, 15))
plt.suptitle("Singular Values for 20% Pruned Clusters", fontsize=20)

for i, k in enumerate([64, 256, 512, 1024]):
    plt.subplot(2, 2, i+1)
    values = pruned_20_percent[k]
    ranks = list(range(len(values)))
    
    plt.bar(ranks, values, width=0.7)
    plt.title(f"K={k}, Sum={sums_20_percent[k]:.4f}", fontsize=16)
    plt.xlabel("Rank", fontsize=14)
    plt.ylabel("Singular Value", fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Add a text annotation for the first few singular values
    for j in range(min(5, len(values))):
        plt.text(j, values[j], f"{values[j]:.4f}", 
                 ha='center', va='bottom', fontsize=10)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("singular_values_20percent.png", dpi=300)

# Create a more detailed version with log scale for better visualization of smaller values
plt.figure(figsize=(20, 15))
plt.suptitle("Singular Values for 10% Pruned Clusters (Log Scale)", fontsize=20)

for i, k in enumerate([64, 256, 512, 1024]):
    plt.subplot(2, 2, i+1)
    values = pruned_10_percent[k]
    ranks = list(range(len(values)))
    
    plt.bar(ranks, values, width=0.7)
    plt.yscale('log')
    plt.title(f"K={k}, Sum={sums_10_percent[k]:.4f}", fontsize=16)
    plt.xlabel("Rank", fontsize=14)
    plt.ylabel("Singular Value (log scale)", fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Add a text annotation for the first few singular values
    for j in range(min(5, len(values))):
        plt.text(j, values[j], f"{values[j]:.4f}", 
                 ha='center', va='bottom', fontsize=10)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("singular_values_10percent_log.png", dpi=300)

# Create a more detailed version with log scale for better visualization of smaller values
plt.figure(figsize=(20, 15))
plt.suptitle("Singular Values for 20% Pruned Clusters (Log Scale)", fontsize=20)

for i, k in enumerate([64, 256, 512, 1024]):
    plt.subplot(2, 2, i+1)
    values = pruned_20_percent[k]
    ranks = list(range(len(values)))
    
    plt.bar(ranks, values, width=0.7)
    plt.yscale('log')
    plt.title(f"K={k}, Sum={sums_20_percent[k]:.4f}", fontsize=16)
    plt.xlabel("Rank", fontsize=14)
    plt.ylabel("Singular Value (log scale)", fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Add a text annotation for the first few singular values
    for j in range(min(5, len(values))):
        plt.text(j, values[j], f"{values[j]:.4f}", 
                 ha='center', va='bottom', fontsize=10)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("singular_values_20percent_log.png", dpi=300)

print("Plots have been saved as:")
print("- singular_values_10percent.png")
print("- singular_values_20percent.png")
print("- singular_values_10percent_log.png")
print("- singular_values_20percent_log.png") 