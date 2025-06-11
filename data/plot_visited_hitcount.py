import numpy as np
import matplotlib.pyplot as plt
import struct

def read_fvecs(filename, c_contiguous=True):
    """Load fvecs format file"""
    fv = np.fromfile(filename, dtype=np.float32)
    if fv.size == 0:
        return np.zeros((0, 0))
    dim = fv.view(np.int32)[0]
    assert dim > 0
    fv = fv.reshape(-1, 1 + dim)
    if not all(fv.view(np.int32)[:, 0] == dim):
        raise IOError("Non-uniform vector sizes in " + filename)
    fv = fv[:, 1:]
    if c_contiguous:
        fv = fv.copy()
    return fv

def read_ivecs(filename):
    """Load ivecs format file"""
    iv = np.fromfile(filename, dtype=np.int32)
    if iv.size == 0:
        return np.zeros((0, 0))
    dim = iv[0]
    assert dim > 0
    iv = iv.reshape(-1, 1 + dim)
    if not all(iv[:, 0] == dim):
        raise IOError("Non-uniform vector sizes in " + filename)
    iv = iv[:, 1:]
    return iv

# Load ground truth
print("Loading ground truth...")
gt = read_ivecs('/data/vector_datasets/sift/sift_groundtruth_10000.ivecs')
gt = gt  # Take first 1000 queries and first 1000 ground truth points per query

# Read the binary file containing visited arrays
print("Reading visited arrays...")
with open('/home/qfshen/workspace/vdb/adsampling/data/sift_visited_array_10K.bin', 'rb') as f:
    # Read number of queries
    num_queries = struct.unpack('Q', f.read(8))[0]
    visited_arrays = []
    
    for i in range(num_queries):
        array_size = struct.unpack('Q', f.read(8))[0]
        visited_array = np.frombuffer(f.read(array_size * 4), dtype=np.int32)
        visited_arrays.append(visited_array)

# Process each query
print(f"Processing {num_queries} queries...")
all_cumulative_counts = []

for query_idx in range(min(num_queries, 1000)):  # Limit to 1000 queries for performance
    if query_idx % 100 == 0:
        print(f"Processing query {query_idx}...")
    
    visited_array = visited_arrays[query_idx]
    gt_set = set(gt[query_idx])  # Convert ground truth to set for O(1) lookup
    
    # Track cumulative count of found ground truth points
    cumulative_count = []
    count_found = 0
    
    for i, visited_point_id in enumerate(visited_array):
        if visited_point_id in gt_set:
            count_found += 1
        cumulative_count.append(count_found)
    
    all_cumulative_counts.append(cumulative_count)

# Find the maximum length to pad all arrays
max_length = max(len(arr) for arr in all_cumulative_counts)

# Pad arrays and compute statistics
padded_counts = []
for counts in all_cumulative_counts:
    padded = np.full(max_length, counts[-1] if counts else 0)  # Pad with last value
    padded[:len(counts)] = counts
    padded_counts.append(padded)

padded_counts = np.array(padded_counts)

# Compute mean and percentiles
mean_cumulative = np.mean(padded_counts, axis=0)
p0_cumulative = np.percentile(padded_counts, 0, axis=0)
p25_cumulative = np.percentile(padded_counts, 25, axis=0)
p75_cumulative = np.percentile(padded_counts, 75, axis=0)
p100_cumulative = np.percentile(padded_counts, 100, axis=0)

# Create x-axis (number of visited points)
x = np.arange(1, max_length + 1)

# Plot the results
plt.figure(figsize=(12, 8))

# Plot individual query lines with low alpha
for i in range(min(50, len(padded_counts))):  # Show first 50 queries
    plt.plot(x, padded_counts[i], alpha=0.1, color='lightblue', linewidth=0.5)

# Plot mean line
plt.plot(x, mean_cumulative, color='red', linewidth=2, label='Mean')

# Plot percentile bands
plt.fill_between(x, p25_cumulative, p75_cumulative, alpha=0.3, color='red', 
                 label='25th-75th percentile')

# Plot min-max band
plt.fill_between(x, p0_cumulative, p100_cumulative, alpha=0.1, color='gray', 
                 label='Min-Max range')

plt.xlabel('Number of Visited Points')
plt.ylabel('Cumulative Count of Found Ground Truth')
plt.title('Distribution of Ground Truth Points Found vs Points Visited\n(SIFT dataset) Top-10K')
plt.legend()
plt.grid(True, alpha=0.3)

# Add some statistics as text
final_mean = mean_cumulative[-1]
final_p0 = p0_cumulative[-1]
final_p25 = p25_cumulative[-1]
final_p75 = p75_cumulative[-1]
final_p100 = p100_cumulative[-1]
plt.text(0.02, 0.98, f'Final mean: {final_mean:.1f}\nFinal 25th-75th: {final_p25:.1f}-{final_p75:.1f}\nFinal min-max: {final_p0:.1f}-{final_p100:.1f}', 
         transform=plt.gca().transAxes, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Print and mark important points on the plot
important_points = [2000, 5000, 10000, 20000, 50000]
for point in important_points:
    if point < max_length:
        idx = point - 1  # Convert to 0-based index
        print(f"\nAt {point} points visited:")
        print(f"Mean: {mean_cumulative[idx]:.2f}")
        print(f"25th-75th percentile: {p25_cumulative[idx]:.2f}-{p75_cumulative[idx]:.2f}")
        print(f"Min-Max: {p0_cumulative[idx]:.2f}-{p100_cumulative[idx]:.2f}")
        
        # Mark the point on the mean line
        plt.plot(point, mean_cumulative[idx], 'ko', markersize=8)
        # Add annotation
        plt.annotate(f'n={point}\nmean={mean_cumulative[idx]:.1f}',
                    xy=(point, mean_cumulative[idx]),
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

plt.tight_layout()
plt.savefig('sift_visited_hitcount_10K.png', dpi=600, bbox_inches='tight')
plt.close()

print("\nPlot saved as 'sift_visited_hitcount.png'")
print(f"Average final count: {final_mean:.2f}")
print(f"25th-75th percentile range: {final_p25:.2f}-{final_p75:.2f}")
print(f"Min-max range: {final_p0:.2f}-{final_p100:.2f}")
