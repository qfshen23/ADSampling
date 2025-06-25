import numpy as np
import struct
import os
from tqdm import tqdm

def read_fvecs(filename, c_contiguous=True):
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
    fv = np.fromfile(filename, dtype=np.int32)
    if fv.size == 0:
        return np.zeros((0, 0))
    dim = fv[0]
    assert dim > 0
    fv = fv.reshape(-1, 1 + dim)
    if not all(fv[:, 0] == dim):
        raise IOError("Non-uniform vector sizes in " + filename)
    return fv[:, 1:]

def compute_overlap_ratio(clusters1, clusters2, k):
    set1 = set(clusters1[:k])
    set2 = set(clusters2[:k])
    intersection = len(set1.intersection(set2))
    return intersection / k

def compute_recall(candidate_ids, gt_vector_ids):
    candidate_set = set(candidate_ids)
    gt_set = set(gt_vector_ids)
    hit_count = len(candidate_set & gt_set)
    recall = hit_count / len(gt_set) if len(gt_set) > 0 else 0.0
    return recall

def main():
    datasets = ['sift']
    K = 1024
    nprobe = 120
    nprobe_segments = [
        (0, 10, 0.9, 64),
        (10, 20, 0.85, 64),
        (20, 30, 0.8, 64),
        (30, 40, 0.75, 64),
        (40, 50, 0.7, 64),
        (60, 70, 0.65, 64),
        (70, 80, 0.4, 64),
        (80, 90, 0.35, 64),
        (90, 100, 0.3, 64),
        (100, 110, 0.25, 64),
        (110, 120, 0.2, 64),
    ]
    gt_neighbors = 10000

    for dataset in datasets:
        print(f"\n=== Processing dataset: {dataset} ===")
        base_path = f'/data/vector_datasets/{dataset}'
        query_path = f'{base_path}/{dataset}_query.fvecs'
        gt_path = f'{base_path}/{dataset}_groundtruth_10000.ivecs'
        centroids_path = f'{base_path}/{dataset}_centroid_{K}.fvecs'
        top_clusters_path = f'{base_path}/{dataset}_top_clusters_{K}.ivecs'
        cluster_ids_path = f'{base_path}/{dataset}_cluster_id_{K}.ivecs'
        missing_files = [p for p in [query_path, gt_path, centroids_path, top_clusters_path, cluster_ids_path] if not os.path.exists(p)]
        if missing_files:
            print(f"Skipping {dataset} - {missing_files} missing")
            continue

        print("Loading data...")
        queries = read_fvecs(query_path)
        groundtruth = read_ivecs(gt_path)[:, :gt_neighbors]
        centroids = read_fvecs(centroids_path)
        top_clusters = read_ivecs(top_clusters_path)
        cluster_ids = read_ivecs(cluster_ids_path)

        num_queries = queries.shape[0]
        recall_results = []
        num_segments = len(nprobe_segments)
        segment_recalls = [[] for _ in range(num_segments)]
        query_segment_recalls = []
        segment_gt_counts = [[] for _ in range(num_segments)]

        print("\nSegmented nprobe, per-segment candidate selection ...")
        for query_idx in tqdm(range(min(1000, num_queries))):
            query = queries[query_idx:query_idx+1]
            distances = np.sum((query - centroids) ** 2, axis=1)
            nearest_clusters = np.argsort(distances)[:nprobe]
            query_top_clusters = np.argsort(distances)
            gt_vector_ids = set(groundtruth[query_idx])

            segment_candidates = []
            query_recalls = []
            for seg_idx, (seg_start, seg_end, seg_percent, k_overlap) in enumerate(nprobe_segments):
                segment_clusters = nearest_clusters[seg_start:seg_end]
                segment_vector_ids = []
                for cluster_id in segment_clusters:
                    vectors_in_cluster = np.where(cluster_ids.flatten() == cluster_id)[0]
                    segment_vector_ids.extend(vectors_in_cluster)
                if len(segment_vector_ids) == 0:
                    segment_recalls[seg_idx].append(0.0)
                    query_recalls.append(0.0)
                    segment_gt_counts[seg_idx].append(0)
                    continue
                # overlap ratio，k_overlap可变
                segment_overlap_ratios = []
                for vector_id in segment_vector_ids:
                    vector_top_clusters = top_clusters[vector_id]
                    overlap_ratio = compute_overlap_ratio(query_top_clusters, vector_top_clusters, k_overlap)
                    segment_overlap_ratios.append(overlap_ratio)
                
                # Calculate number of vectors to select based on percentage
                seg_top_x = int(len(segment_vector_ids) * seg_percent)
                sorted_indices = np.argsort(segment_overlap_ratios)[::-1]
                selected_ids = [segment_vector_ids[i] for i in sorted_indices[:seg_top_x]]
                segment_candidates.extend(selected_ids)

                # 计算该segment中的recall
                gt_in_segment_clusters = set()
                for cluster_id in segment_clusters:
                    vectors_in_cluster = set(np.where(cluster_ids.flatten() == cluster_id)[0])
                    gt_in_segment_clusters.update(gt_vector_ids & vectors_in_cluster)
                segment_gt_counts[seg_idx].append(len(gt_in_segment_clusters))
                gt_found = gt_vector_ids & set(selected_ids)
                seg_recall = len(gt_found) / len(gt_in_segment_clusters) if len(gt_in_segment_clusters) > 0 else 0.0
                segment_recalls[seg_idx].append(seg_recall)
                query_recalls.append(seg_recall)

            query_segment_recalls.append(query_recalls)
            recall = compute_recall(segment_candidates, gt_vector_ids)
            recall_results.append(recall)
            per_seg_str = ', '.join([f'seg{si}:(recall={query_recalls[si]:.4f}, gts={segment_gt_counts[si][-1]})' for si in range(num_segments)])
            print(f"Query {query_idx} - candidates={len(segment_candidates)}, total_recall={recall:.4f}")
            print(f"  Segment stats: [{per_seg_str}]")
            print(f"  Average recall up to now: {np.mean(recall_results):.4f}")

        print(f"\nAverage recall over {len(recall_results)} queries for {dataset}: {np.mean(recall_results):.4f}")
        for seg_idx, (_, _, seg_percent, k_overlap) in enumerate(nprobe_segments):
            avg_seg_recall = np.mean(segment_recalls[seg_idx])
            avg_seg_gts = np.mean(segment_gt_counts[seg_idx])
            print(f"  Segment {seg_idx+1} ({seg_percent*100:.0f}% of vectors, top-{k_overlap} overlap): avg recall = {avg_seg_recall:.4f}, avg gts = {avg_seg_gts:.1f}")

if __name__ == '__main__':
    main()
