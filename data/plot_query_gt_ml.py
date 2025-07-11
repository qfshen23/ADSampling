import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
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

def fake_data_loader():
    dataset = 'sift'
    K = 1024
    gt_neighbors = 10000
    
    base_path = f'/data/vector_datasets/{dataset}'
    query_path = f'{base_path}/{dataset}_query.fvecs'
    gt_path = f'{base_path}/{dataset}_groundtruth_10000.ivecs'
    centroids_path = f'{base_path}/{dataset}_centroid_{K}.fvecs'
    top_clusters_path = f'{base_path}/{dataset}_top_clusters_{K}.ivecs'
    cluster_ids_path = f'{base_path}/{dataset}_cluster_id_{K}.ivecs'
    
    print("Loading data...")
    queries = read_fvecs(query_path)
    groundtruth = read_ivecs(gt_path)[:,:gt_neighbors]
    centroids = read_fvecs(centroids_path)
    top_clusters = read_ivecs(top_clusters_path)
    cluster_ids = read_ivecs(cluster_ids_path)
    
    print(f"Queries shape: {queries.shape}")
    print(f"Groundtruth shape: {groundtruth.shape}")
    print(f"Centroids shape: {centroids.shape}")
    print(f"Top clusters shape: {top_clusters.shape}")
    print(f"Cluster IDs shape: {cluster_ids.shape}")
    
    return queries, centroids, cluster_ids, top_clusters, groundtruth

# ========== 训练样本构建 ==========
def build_training_samples(
    queries, centroids, cluster_ids, top_clusters, groundtruth,
    k_overlap=64, nprobe=120, train_range=(5000, 10000), num_neg_per_query=100
):
    q_indices = range(*train_range)
    sample_query_clusters = []
    sample_base_clusters = []
    sample_labels = []
    for query_idx in tqdm(q_indices, desc="Building train samples"):
        query = queries[query_idx]
        dists = np.sum((query - centroids)**2, axis=1)
        nearest_clusters = np.argsort(dists)[:nprobe]
        probe_vector_ids = np.where(np.isin(cluster_ids, nearest_clusters))[0]
        gt_ids = set(groundtruth[query_idx])

        # 正样本: probe_vector 在GT里
        positive_ids = [vid for vid in probe_vector_ids if vid in gt_ids]
        for vid in positive_ids:
            sample_query_clusters.append(np.argsort(dists)[:k_overlap])
            sample_base_clusters.append(top_clusters[vid][:k_overlap])
            sample_labels.append(1.0)
        # 负样本: probe_vector 不在GT里，随机采样
        neg_ids = [vid for vid in probe_vector_ids if vid not in gt_ids]
        if len(neg_ids) > num_neg_per_query:
            neg_ids = np.random.choice(neg_ids, num_neg_per_query, replace=False)
        for vid in neg_ids:
            sample_query_clusters.append(np.argsort(dists)[:k_overlap])
            sample_base_clusters.append(top_clusters[vid][:k_overlap])
            sample_labels.append(0.0)
    return (
        torch.LongTensor(np.array(sample_query_clusters)),
        torch.LongTensor(np.array(sample_base_clusters)),
        torch.FloatTensor(np.array(sample_labels))
    )

# ========== 模型 ==========
class ClusterAttentionScore(nn.Module):
    def __init__(self, num_clusters, embed_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(num_clusters, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1)
        )

    def forward(self, query_clusters, base_clusters):
        q_emb = self.embedding(query_clusters)      # [B, k, D]
        b_emb = self.embedding(base_clusters)       # [B, k, D]
        attn_output, _ = self.attn(q_emb, b_emb, b_emb) # [B, k, D]
        pooled = attn_output.mean(dim=1)                # [B, D]
        score = self.mlp(pooled).squeeze(-1)            # [B]
        return score

# ========== 训练 ==========
def train_model(model, optimizer, criterion, q_clusters, b_clusters, labels, epochs=3, batch_size=1024):
    model.train()
    dataset_size = len(labels)
    for epoch in range(epochs):
        perm = torch.randperm(dataset_size)
        total_loss = 0
        for i in tqdm(range(0, dataset_size, batch_size), desc=f"Epoch {epoch+1}"):
            idx = perm[i:i+batch_size]
            batch_q = q_clusters[idx]
            batch_b = b_clusters[idx]
            batch_label = labels[idx]
            optimizer.zero_grad()
            pred = model(batch_q, batch_b)
            loss = criterion(pred, batch_label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: avg_loss={total_loss/(dataset_size//batch_size):.4f}")

# ========== 推理 ==========
def model_inference(model, query_clusters, base_clusters, batch_size=4096):
    model.eval()
    scores = []
    with torch.no_grad():
        for i in range(0, len(query_clusters), batch_size):
            q = query_clusters[i:i+batch_size]
            b = base_clusters[i:i+batch_size]
            pred = model(q, b).cpu().numpy()
            scores.append(pred)
    return np.concatenate(scores, axis=0)

def eval_recall(probe_vector_ids, pred_scores, gt_ids, top_x):
    sorted_indices = np.argsort(pred_scores)[::-1][:top_x]
    pred_top_ids = set([probe_vector_ids[i] for i in sorted_indices])
    overlap = len(pred_top_ids.intersection(gt_ids))
    recall = overlap / len(gt_ids)
    return recall

# ========== 主流程 ==========
def main():
    # Step 1. 数据加载
    queries, centroids, cluster_ids, top_clusters, groundtruth = fake_data_loader()
    num_clusters = centroids.shape[0]
    k_overlap = 64
    nprobe = 120

    # Step 2. 构建训练样本
    q_clusters, b_clusters, labels = build_training_samples(
        queries, centroids, cluster_ids, top_clusters, groundtruth,
        k_overlap=k_overlap, nprobe=nprobe, train_range=(9000, 10000), num_neg_per_query=100
    )

    # Step 3. 初始化模型与训练
    model = ClusterAttentionScore(num_clusters, embed_dim=64)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    train_model(model, optimizer, criterion, q_clusters, b_clusters, labels, epochs=3, batch_size=1024)

    # Step 4. 推理&评估前1000个query
    recalls = []
    for query_idx in tqdm(range(0, 1000), desc="Inference/Eval"):  
        query = queries[query_idx]
        dists = np.sum((query - centroids)**2, axis=1)
        nearest_clusters = np.argsort(dists)[:nprobe]
        probe_vector_ids = np.where(np.isin(cluster_ids, nearest_clusters))[0]
        gt_ids = set(groundtruth[query_idx])

        q_input = [np.argsort(dists)[:k_overlap] for _ in probe_vector_ids]
        b_input = [top_clusters[vid][:k_overlap] for vid in probe_vector_ids]
        q_input = torch.LongTensor(q_input)
        b_input = torch.LongTensor(b_input)

        pred_scores = model_inference(model, q_input, b_input)
        recall = eval_recall(probe_vector_ids, pred_scores, gt_ids, top_x=40000)
        recalls.append(recall)
        print(f"Query {query_idx}: recall@40K = {recall:.4f}")

    avg_recall = np.mean(recalls)
    print(f"\nAverage recall@40K over {len(recalls)} queries = {avg_recall:.4f}")

if __name__ == '__main__':
    main()
