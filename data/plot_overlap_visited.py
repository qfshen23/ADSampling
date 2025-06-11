import numpy as np
import matplotlib.pyplot as plt
import os
import struct
from tqdm import tqdm

def read_fvecs(filename, c_contiguous=True):
    """Load fvecs format file"""
    fv = np.fromfile(filename, dtype=np.float32)
    if fv.size == 0:
        return np.zeros((0, 0))
    dim = fv.view(np.int32)[0]
    fv = fv.reshape(-1, 1 + dim)
    assert np.all(fv.view(np.int32)[:, 0] == dim)
    fv = fv[:, 1:]
    return fv.copy() if c_contiguous else fv

def read_ivecs(filename):
    """Load ivecs format file"""
    iv = np.fromfile(filename, dtype=np.int32)
    if iv.size == 0:
        return np.zeros((0, 0))
    dim = iv[0]
    iv = iv.reshape(-1, 1 + dim)
    assert np.all(iv[:, 0] == dim)
    return iv[:, 1:]

# ---------- 参数设置 ----------
K = 1024
TOP_K_LIST = [4, 8, 16, 32, 64, 128]
NUM_QUERIES = 1
dataset_path = '/data/vector_datasets/sift'
visited_file = '/home/qfshen/workspace/vdb/adsampling/data/sift_visited_array_10K.bin'

# ---------- 数据加载 ----------
print("Loading data...")
base_vectors = read_fvecs(os.path.join(dataset_path, 'sift_base.fvecs'))
centroids = read_fvecs(os.path.join(dataset_path, f'sift_centroid_{K}.fvecs'))
query_vectors = read_fvecs(os.path.join(dataset_path, 'sift_query.fvecs'))[:NUM_QUERIES]

# ---------- 加载 visited arrays ----------
print("Reading visited arrays...")
visited_arrays = []
with open(visited_file, 'rb') as f:
    num_queries = struct.unpack('Q', f.read(8))[0]
    for _ in range(min(num_queries, NUM_QUERIES)):
        array_size = struct.unpack('Q', f.read(8))[0]
        visited_array = np.frombuffer(f.read(array_size * 4), dtype=np.int32)
        visited_arrays.append(visited_array)

assert len(visited_arrays) == NUM_QUERIES

# ---------- 分批计算所有base vectors的距离到所有centroids ----------
print("Pre-computing distances between base vectors and centroids...")
batch_size = 10000
num_batches = (len(base_vectors) + batch_size - 1) // batch_size
base_to_centroid_distances = []

for i in tqdm(range(num_batches)):
    start_idx = i * batch_size
    end_idx = min((i + 1) * batch_size, len(base_vectors))
    batch_distances = np.linalg.norm(base_vectors[start_idx:end_idx, np.newaxis] - centroids, axis=2)
    base_to_centroid_distances.append(batch_distances)

base_to_centroid_distances = np.concatenate(base_to_centroid_distances, axis=0)

# ---------- 预计算所有query vectors的距离到所有centroids ----------
print("Pre-computing distances between query vectors and centroids...")
query_to_centroid_distances = np.linalg.norm(query_vectors[:, np.newaxis] - centroids, axis=2)

# ---------- 针对每个 top-K 分别绘图 ----------
for TOP_K_CLUSTER in TOP_K_LIST:
    print(f"\nProcessing Top-{TOP_K_CLUSTER} cluster overlap...")

    # ----- 使用预计算的距离获取 base vectors 的 top-k cluster -----
    print(f"Computing top-{TOP_K_CLUSTER} clusters for base vectors...")
    base_top_clusters = np.argsort(base_to_centroid_distances, axis=1)[:, :TOP_K_CLUSTER]

    # ----- 使用预计算的距离获取 query vectors 的 top-k cluster -----
    print(f"Computing top-{TOP_K_CLUSTER} clusters for query vectors...")
    query_top_clusters = np.argsort(query_to_centroid_distances, axis=1)[:, :TOP_K_CLUSTER]

    # ----- 计算 overlap 并绘制折线图 -----
    print(f"Plotting Top-{TOP_K_CLUSTER} overlap line plot...")
    plt.figure(figsize=(20, 10))

    for q_idx in tqdm(range(NUM_QUERIES)):
        query_clusters = set(query_top_clusters[q_idx])
        visited = visited_arrays[q_idx]

        x = list(range(len(visited)))
        y = []

        for vid in visited:
            base_clusters = set(base_top_clusters[vid])
            overlap = len(query_clusters & base_clusters) / TOP_K_CLUSTER
            y.append(overlap)

        plt.plot(x, y, alpha=0.5, linewidth=1)

    plt.xlabel("Visited List Index")
    plt.ylabel("Cluster Overlap Ratio")
    plt.title(f"Query-Visited Cluster Overlap (Top-{TOP_K_CLUSTER})")
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.05, 1.05)
    plt.savefig(f'sift_query_visited_lines_top{TOP_K_CLUSTER}.png', dpi=300, bbox_inches='tight')
    plt.close()

print("All plots saved.")
