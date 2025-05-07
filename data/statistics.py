import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

def load_fvecs(filename, c_contiguous=True):
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

def load_ivecs(filename, c_contiguous=True):
    fv = np.fromfile(filename, dtype=np.int32)
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

def calculate_nearest_clusters(query_vectors, centroids, x):
    print(f"Calculating {x} nearest clusters for each query...")
    nearest_clusters = []
    for query in tqdm(query_vectors, desc="Processing queries"):
        distances = np.linalg.norm(centroids - query, axis=1)
        top_x = np.argsort(distances)[:x]
        nearest_clusters.append(top_x.tolist())
    return nearest_clusters

def calculate_avg_collision_rate(query_nearest_clusters, gt_clusters):
    total = 0.0
    for q_clusters, gt in zip(query_nearest_clusters, gt_clusters):
        q_set = set(q_clusters)
        collision = len(q_set & gt) / len(gt) if gt else 0.0
        total += collision
    return total / len(gt_clusters)

def plot_collision_vs_x(results, output_file=None):
    x_vals = sorted(results.keys())
    y_vals = [results[x] for x in x_vals]

    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, y_vals, marker='o')
    plt.xlabel('Number of Nearest Clusters per Query (x)')
    plt.ylabel('Average Collision Rate')
    plt.title('Collision Rate vs. Nearest Cluster Count')
    plt.grid(True)

    if output_file:
        plt.savefig(output_file, dpi=600)
        print(f"Plot saved to {output_file}")
    else:
        plt.show()

def main():
    # Paths
    query_path = "/data/vector_datasets/sift/sift_query.fvecs"
    gt_path = "/data/vector_datasets/sift/sift_groundtruth.ivecs"
    data_clusters_path = "/data/vector_datasets/sift/sift_cluster_id_1024.ivecs"
    centroids_path = "/data/vector_datasets/sift/sift_centroid_1024.fvecs"
    output_file = "../collision_vs_x.png"

    # Load data
    print("Loading query vectors...")
    query_vectors = load_fvecs(query_path)
    print("Loading centroids...")
    centroids = load_fvecs(centroids_path)
    print("Loading ground truth...")
    gt_indices = load_ivecs(gt_path)
    print("Loading data cluster IDs...")
    data_cluster_ids = load_ivecs(data_clusters_path).squeeze()  # shape: [num_base_vectors]

    # Extract GT clusters
    gt_clusters = [set(data_cluster_ids[knn]) for knn in gt_indices]

    # Evaluate for multiple x
    x_values = [1, 4, 8, 16, 32, 64, 128]
    collision_results = {}

    for x in x_values:
        query_nearest_clusters = calculate_nearest_clusters(query_vectors, centroids, x)
        avg_collision = calculate_avg_collision_rate(query_nearest_clusters, gt_clusters)
        collision_results[x] = avg_collision
        print(f"x={x}: Avg Collision Rate = {avg_collision:.4f}")

    # Plot
    plot_collision_vs_x(collision_results, output_file)

if __name__ == "__main__":
    main()
