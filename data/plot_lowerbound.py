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

# Create x-axis values (number of visited points)
x = np.arange(1, lower_bounds.shape[1] + 1)

# Plot the lower bounds for each query
plt.figure(figsize=(10, 6))
for i in range(min(100, len(lower_bounds))):  # Plot first 100 queries to avoid overcrowding
    plt.plot(x, lower_bounds[i], alpha=0.1, color='blue')

plt.xlabel('Number of Visited Points')
plt.ylabel('Lower Bound Value')
plt.title('Lower Bounds vs Number of Visited Points for Top-10K')
plt.grid(True)
plt.savefig('sift_lower_bounds_plot_10K.png', dpi=600)
plt.close()
