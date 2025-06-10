import numpy as np
import matplotlib.pyplot as plt
import struct

# Read the binary file containing lower bounds
with open('/home/qfshen/workspace/vdb/adsampling/data/sift_lower_bounds_10K.bin', 'rb') as f:
    # Read number of vectors
    num_vectors = struct.unpack('Q', f.read(8))[0]
    
    lower_bounds = []
    max_len = 0
    # First pass to get max length
    for i in range(num_vectors):
        vec_size = struct.unpack('Q', f.read(8))[0]
        vec_data = np.frombuffer(f.read(vec_size * 4), dtype=np.float32)
        max_len = max(max_len, len(vec_data))
        lower_bounds.append(vec_data)

    # Pad arrays to same length with NaN
    padded_bounds = []
    for vec in lower_bounds:
        padded = np.full(max_len, np.nan)
        padded[:len(vec)] = vec
        padded_bounds.append(padded)

# Convert to numpy array for easier processing
lower_bounds = np.array(padded_bounds)

# Take the square root of the lower bound values
lower_bounds = np.sqrt(lower_bounds)

# Skip first 10K DCOs
skip_dcos = 10000
if max_len <= skip_dcos:
    print(f"Warning: max_len ({max_len}) is less than or equal to skip_dcos ({skip_dcos})")
    skip_dcos = 0

# Count lower bound changes for each query after skipping first 10K DCOs
change_counts = []
for i in range(len(lower_bounds)):
    query_bounds = lower_bounds[i, skip_dcos:]
    # Remove NaN values
    valid_bounds = query_bounds[~np.isnan(query_bounds)]
    
    # Count changes (where consecutive values are different)
    changes = 0
    for j in range(1, len(valid_bounds)):
        if valid_bounds[j] != valid_bounds[j-1]:
            changes += 1
    
    change_counts.append(changes)

# Convert to numpy array for easier analysis
change_counts = np.array(change_counts)

# Print statistics
print(f"Statistics for lower bound changes across {len(change_counts)} queries after first {skip_dcos} DCOs:")
print(f"Mean changes per query: {np.mean(change_counts):.2f}")
print(f"Median changes per query: {np.median(change_counts):.2f}")
print(f"Min changes: {np.min(change_counts)}")
print(f"Max changes: {np.max(change_counts)}")
print(f"Standard deviation: {np.std(change_counts):.2f}")

# Plot histogram of change counts
plt.figure(figsize=(10, 6))
plt.hist(change_counts, bins=50, edgecolor='black', alpha=0.7)
plt.xlabel('Number of Lower Bound Changes')
plt.ylabel('Number of Queries')
plt.title(f'Distribution of Lower Bound Changes per Query (After First {skip_dcos} DCOs)')
plt.grid(True, alpha=0.3)
plt.axvline(np.mean(change_counts), color='red', linestyle='--', linewidth=2, 
           label=f'Mean: {np.mean(change_counts):.2f}')
plt.axvline(np.median(change_counts), color='orange', linestyle='--', linewidth=2, 
           label=f'Median: {np.median(change_counts):.2f}')
plt.legend()
plt.savefig(f'sift_lower_bounds_changes_histogram_after_{skip_dcos}_dcos.png', dpi=600)

# Plot change counts vs query index
plt.figure(figsize=(12, 6))
plt.plot(range(len(change_counts)), change_counts, alpha=0.7, linewidth=0.8)
plt.xlabel('Query Index')
plt.ylabel('Number of Lower Bound Changes')
plt.title(f'Lower Bound Changes per Query (After First {skip_dcos} DCOs)')
plt.grid(True, alpha=0.3)
plt.axhline(np.mean(change_counts), color='red', linestyle='--', linewidth=2, 
           label=f'Mean: {np.mean(change_counts):.2f}')
plt.legend()
plt.savefig(f'sift_lower_bounds_changes_per_query_after_{skip_dcos}_dcos.png', dpi=600)
plt.close()
