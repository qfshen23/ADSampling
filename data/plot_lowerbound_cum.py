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

# Calculate cumulative lower bound changes after first 10K DCOs
remaining_len = max_len - skip_dcos
cum_changes_per_query = []

for query_idx in range(num_vectors):
    query_bounds = lower_bounds[query_idx, skip_dcos:]
    # Remove NaN values
    valid_bounds = query_bounds[~np.isnan(query_bounds)]
    
    # Calculate cumulative changes
    cum_changes = [0]  # Start with 0 changes
    changes = 0
    
    for j in range(1, len(valid_bounds)):
        if valid_bounds[j] != valid_bounds[j-1]:
            changes += 1
        cum_changes.append(changes)
    
    # Pad with last value if needed to match remaining_len
    while len(cum_changes) < remaining_len:
        cum_changes.append(cum_changes[-1] if cum_changes else 0)
    
    cum_changes_per_query.append(cum_changes[:remaining_len])

# Convert to numpy array and calculate average
cum_changes_per_query = np.array(cum_changes_per_query)
avg_cum_changes = np.mean(cum_changes_per_query, axis=0)

# Create x-axis values (DCO numbers after first 10K)
x_dcos = np.arange(skip_dcos, skip_dcos + len(avg_cum_changes))

# Plot average cumulative lower bound changes
plt.figure(figsize=(12, 8))
plt.plot(x_dcos, avg_cum_changes, linewidth=2, color='blue', label='Average Cumulative Changes')
plt.xlabel('Number of DCOs')
plt.ylabel('Average Lower Bound Updated Times')
plt.title(f'Average Cumulative Lower Bound Changes After First {skip_dcos} DCOs')
plt.grid(True, alpha=0.3)
plt.legend()
plt.savefig('sift_lower_bounds_cumulative_changes.png', dpi=600)
plt.close()

print(f"Processed {num_vectors} queries")
print(f"After {skip_dcos} DCOs, average final update count: {avg_cum_changes[-1]:.2f}")
