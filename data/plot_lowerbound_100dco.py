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

# Skip first 1000 DCOs
skip_dcos = 10000
if max_len <= skip_dcos:
    print(f"Warning: max_len ({max_len}) is less than or equal to skip_dcos ({skip_dcos})")
    skip_dcos = 0

# Calculate decreases in lower bounds for each 100 DCO interval after skipping first 10000
dco_interval = 100
remaining_len = max_len - skip_dcos
num_intervals = remaining_len // dco_interval

decreases_per_query = []

for query_idx in range(min(num_vectors, 100)):  # Only process first 100 queries
    query_decreases = []
    
    for interval in range(num_intervals):
        start_idx = skip_dcos + interval * dco_interval
        end_idx = min(skip_dcos + (interval + 1) * dco_interval, max_len)
        
        # Get the lower bounds for this query in this interval
        interval_bounds = lower_bounds[query_idx, start_idx:end_idx]
        
        # Remove NaN values
        valid_bounds = interval_bounds[~np.isnan(interval_bounds)]
        
        if len(valid_bounds) > 1:
            # Calculate total decrease in this interval (start - end)
            decrease = valid_bounds[0] - valid_bounds[-1]
            query_decreases.append(decrease)
        else:
            # If not enough valid data points, append NaN
            query_decreases.append(np.nan)
    
    decreases_per_query.append(query_decreases)

# Create x-axis values (interval numbers)
x_intervals = np.arange(1, num_intervals + 1)

# Plot the decreases for each interval
plt.figure(figsize=(12, 8))

# Plot line for each query
for query_idx, query_decreases in enumerate(decreases_per_query):
    # Convert to numpy array and handle NaN values
    query_decreases = np.array(query_decreases)
    valid_mask = ~np.isnan(query_decreases)
    
    if np.any(valid_mask):
        plt.plot(x_intervals[valid_mask], query_decreases[valid_mask], alpha=0.3, color='blue', linewidth=0.5)

# Calculate and plot mean decrease for each interval
mean_decreases = []
for interval in range(num_intervals):
    interval_values = [decreases_per_query[q][interval] for q in range(len(decreases_per_query))]
    interval_values = [v for v in interval_values if not np.isnan(v)]
    if interval_values:
        mean_decreases.append(np.mean(interval_values))
    else:
        mean_decreases.append(np.nan)

valid_mean_mask = ~np.isnan(mean_decreases)
plt.plot(x_intervals[valid_mean_mask], np.array(mean_decreases)[valid_mean_mask], 'r-', linewidth=2, label='Mean Decrease')

plt.xlabel('100-DCO Interval (after skipping first 10000 DCOs)')
plt.ylabel('Lower Bound Decrease')
plt.title('Lower Bound Decrease per 100-DCO Interval (First 100 Queries, Skip First 10000 DCOs) Top-10K')
plt.legend()
plt.grid(True)
plt.savefig('sift_lower_bounds_decrease_10K.png', dpi=600)
plt.close()
