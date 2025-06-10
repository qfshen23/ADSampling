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

# Parameters 
K = 64  # Number of clusters
c = 8   # Number of nearest centroids to consider

# Load centroids and cluster IDs from disk
print("Loading centroids...")
centroids = read_fvecs(f'/data/vector_datasets/sift/sift_centroid_{K}.fvecs')

print("Loading cluster IDs...")
cluster_ids = read_ivecs(f'/data/vector_datasets/sift/sift_cluster_id_{K}.ivecs')
cluster_assignments = cluster_ids.flatten()  # Convert to 1D array

print("Loading queries...")
queries = read_fvecs('/data/vector_datasets/sift/sift_query.fvecs')

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
    
    query = queries[query_idx]
    visited_array = visited_arrays[query_idx]
    
    # Calculate distances from query to all centroids
    query_distances = np.sum((centroids - query) ** 2, axis=1)
    
    # Get indices of c nearest centroids
    nearest_centroid_indices = np.argsort(query_distances)[:c]
    nearest_centroids_set = set(nearest_centroid_indices)
    
    # Track cumulative count of visited points in nearest centroids
    cumulative_count = []
    count_in_nearest = 0
    
    for i, visited_point_id in enumerate(visited_array):
        # Check if this visited point belongs to one of the nearest centroids
        if visited_point_id < len(cluster_assignments):
            point_cluster = cluster_assignments[visited_point_id]
            if point_cluster in nearest_centroids_set:
                count_in_nearest += 1
        
        cumulative_count.append(count_in_nearest)
    
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
plt.plot(x, mean_cumulative, color='red', linewidth=2, label=f'Mean (c={c} nearest centroids)')

# Plot percentile bands
plt.fill_between(x, p25_cumulative, p75_cumulative, alpha=0.3, color='red', 
                 label='25th-75th percentile')

# Plot min-max band
plt.fill_between(x, p0_cumulative, p100_cumulative, alpha=0.1, color='gray', 
                 label='Min-Max range')

plt.xlabel('Number of Visited Points')
plt.ylabel('Cumulative Count of Points in Nearest Centroids')
plt.title(f'Distribution of Visited Points in {c} Nearest Centroids\n(SIFT dataset, {K} clusters)')
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

plt.tight_layout()
plt.savefig(f'sift_visited_distribution_c{c}_clusters{K}.png', dpi=600, bbox_inches='tight')
plt.close()

print(f"Plot saved as 'sift_visited_distribution_c{c}_clusters{K}.png'")
print(f"Average final count: {final_mean:.2f}")
print(f"25th-75th percentile range: {final_p25:.2f}-{final_p75:.2f}")
print(f"Min-max range: {final_p0:.2f}-{final_p100:.2f}")

# Also create a plot showing the percentage of visited points in nearest centroids
plt.figure(figsize=(12, 8))

# Calculate percentages
total_visited = np.arange(1, max_length + 1)
percentage_in_nearest = (mean_cumulative / total_visited) * 100
p0_percentage = (p0_cumulative / total_visited) * 100
p25_percentage = (p25_cumulative / total_visited) * 100
p75_percentage = (p75_cumulative / total_visited) * 100
p100_percentage = (p100_cumulative / total_visited) * 100

# Plot individual query percentages with low alpha
for i in range(min(50, len(padded_counts))):
    query_percentages = (padded_counts[i] / total_visited) * 100
    plt.plot(x, query_percentages, alpha=0.1, color='lightgreen', linewidth=0.5)

# Plot mean percentage line
plt.plot(x, percentage_in_nearest, color='darkgreen', linewidth=2, 
         label=f'Mean percentage (c={c} nearest centroids)')

# Plot percentile bands
plt.fill_between(x, p25_percentage, p75_percentage, alpha=0.3, color='darkgreen', 
                 label='25th-75th percentile')

# Plot min-max band
plt.fill_between(x, p0_percentage, p100_percentage, alpha=0.1, color='gray', 
                 label='Min-Max range')

plt.xlabel('Number of Visited Points')
plt.ylabel('Percentage of Visited Points in Nearest Centroids (%)')
plt.title(f'Percentage of Visited Points in {c} Nearest Centroids\n(SIFT dataset, {K} clusters)')
plt.legend()
plt.grid(True, alpha=0.3)

# Add some statistics as text
final_percentage = percentage_in_nearest[-1]
final_p0_pct = p0_percentage[-1]
final_p25_pct = p25_percentage[-1]
final_p75_pct = p75_percentage[-1]
final_p100_pct = p100_percentage[-1]
plt.text(0.02, 0.98, f'Final mean: {final_percentage:.1f}%\nFinal 25th-75th: {final_p25_pct:.1f}%-{final_p75_pct:.1f}%\nFinal min-max: {final_p0_pct:.1f}%-{final_p100_pct:.1f}%', 
         transform=plt.gca().transAxes, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig(f'sift_visited_percentage_c{c}_clusters{K}.png', dpi=600, bbox_inches='tight')
plt.close()

print(f"Percentage plot saved as 'sift_visited_percentage_c{c}_clusters{K}.png'")
print(f"Average final percentage: {final_percentage:.2f}%")
print(f"Final percentage range: {final_p0_pct:.2f}%-{final_p100_pct:.2f}%")
