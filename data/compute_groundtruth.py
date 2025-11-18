#!/usr/bin/env python3
"""
计算数据集的groundtruth文件
使用暴力搜索（brute force）计算每个查询向量的k个最近邻
"""

import numpy as np
import faiss
import struct
import os
import argparse
from tqdm import tqdm

def read_vecs_fast(filename, show_progress=True):
    """
    快速读取 .vecs 格式的向量文件（优化版本）
    一次性读取所有数据并重新组织，比逐个向量读取快很多
    
    参数:
        filename: 文件路径
        show_progress: 是否显示读取进度
    """
    # 根据文件扩展名推断数据类型
    if filename.endswith(".fvecs"):
        dtype = np.float32
        dtype_size = 4
    elif filename.endswith(".ivecs"):
        dtype = np.int32
        dtype_size = 4
    elif filename.endswith(".bvecs"):
        dtype = np.uint8
        dtype_size = 1
    else:
        raise ValueError(f"未知的 vecs 文件类型: {filename}")
    
    if show_progress:
        print("  📊 分析文件结构...")
    
    # 获取文件大小
    file_size = os.path.getsize(filename)
    
    with open(filename, "rb") as f:
        # 读取第一个向量的维度
        dim = struct.unpack('i', f.read(4))[0]
        
        # 计算每个向量占用的字节数：4字节(维度) + dim * dtype_size
        vec_size = 4 + dim * dtype_size
        
        # 计算总向量数
        n = file_size // vec_size
        
        if show_progress:
            print(f"  📏 检测到 {n:,} 个向量，每个维度 {dim}")
            print(f"  💾 文件大小: {file_size / (1024**3):.2f} GB")
            print(f"  🚀 开始快速读取...")
        
        # 回到文件开头
        f.seek(0)
        
        # 一次性读取所有数据
        all_data = np.fromfile(f, dtype=np.uint8, count=file_size)
    
    if show_progress:
        print(f"  🔄 重组数据结构...")
    
    # 高效方法：使用numpy的视图和切片操作，避免Python循环
    # 将字节数据重新解释为结构化数组
    all_data = all_data.reshape(n, vec_size)
    
    # 跳过每个向量前4字节的维度信息，提取向量数据
    # all_data[:, 4:] 跳过前4列（维度信息）
    vec_data = all_data[:, 4:].copy()  # copy()确保数据连续
    
    # 将字节数据重新解释为目标数据类型
    vectors = np.frombuffer(vec_data.tobytes(), dtype=dtype).reshape(n, dim)
    
    if show_progress:
        print(f"  ✅ 完成！读取了 {n:,} 个 {dim} 维向量")
    
    return vectors

def write_ivecs(filename, data):
    """写入 ivecs 格式文件（整数向量）"""
    print(f"写入文件 - {filename}")
    with open(filename, 'wb') as fp:
        for row in data:
            # 写入维度
            d = struct.pack('I', len(row))
            fp.write(d)
            # 写入每个元素
            for x in row:
                a = struct.pack('i', int(x))
                fp.write(a)

def compute_groundtruth(base_vectors, query_vectors, k=100, metric='L2', use_gpu=False):
    """
    计算groundtruth
    
    参数:
        base_vectors: 基础数据集向量 (N x D)
        query_vectors: 查询向量 (Q x D)
        k: 返回的最近邻数量
        metric: 距离度量，'L2' 或 'IP' (inner product)
        use_gpu: 是否使用GPU加速
    
    返回:
        groundtruth: 每个查询的k个最近邻的索引 (Q x k)
    """
    n, d = base_vectors.shape
    nq = query_vectors.shape[0]
    
    print(f"数据集大小: {n} 个向量, 维度: {d}")
    print(f"查询数量: {nq}")
    print(f"计算 top-{k} 最近邻")
    print(f"距离度量: {metric}")
    
    # 创建 FAISS 索引
    if metric == 'L2':
        index = faiss.IndexFlatL2(d)
    elif metric == 'IP':
        index = faiss.IndexFlatIP(d)
    else:
        raise ValueError(f"不支持的距离度量: {metric}")
    
    # 如果使用GPU
    if use_gpu and faiss.get_num_gpus() > 0:
        print("使用GPU加速")
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
    
    # 添加基础向量到索引
    print("添加向量到索引...")
    index.add(base_vectors)
    
    # 执行搜索
    print("执行搜索...")
    distances, indices = index.search(query_vectors, k)
    
    return indices

def main():
    parser = argparse.ArgumentParser(description='计算数据集的groundtruth')
    parser.add_argument('--dataset', type=str, required=True,
                        help='数据集名称（例如: deep10m, gist, sift1M）')
    parser.add_argument('--data_dir', type=str, default='/data/vector_datasets',
                        help='数据集根目录')
    parser.add_argument('--k', type=int, default=100,
                        help='计算的最近邻数量（默认: 100）')
    parser.add_argument('--metric', type=str, default='L2', choices=['L2', 'IP'],
                        help='距离度量方式：L2（欧氏距离）或 IP（内积）')
    parser.add_argument('--gpu', action='store_true',
                        help='使用GPU加速')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='批处理大小（用于大规模查询，默认一次性处理所有查询）')
    
    args = parser.parse_args()
    
    # 构建文件路径
    dataset_path = os.path.join(args.data_dir, args.dataset)
    base_file = os.path.join(dataset_path, f'{args.dataset}_base.fvecs')
    query_file = os.path.join(dataset_path, f'{args.dataset}_query.fvecs')
    output_file = os.path.join(dataset_path, f'{args.dataset}_groundtruth.ivecs')
    
    # 检查文件是否存在
    if not os.path.exists(base_file):
        print(f"错误: 基础数据集文件不存在: {base_file}")
        return
    
    if not os.path.exists(query_file):
        print(f"错误: 查询文件不存在: {query_file}")
        return
    
    print(f"正在处理数据集: {args.dataset}")
    print(f"基础数据集: {base_file}")
    print(f"查询文件: {query_file}")
    print(f"输出文件: {output_file}")
    print("-" * 60)
    
    # 读取数据
    print("读取基础数据集...")
    base_vectors = read_vecs_fast(base_file)
    print(f"基础数据集形状: {base_vectors.shape}")
    
    print("读取查询向量...")
    query_vectors = read_vecs_fast(query_file)
    print(f"查询向量形状: {query_vectors.shape}")
    print("-" * 60)
    
    # 计算groundtruth
    if args.batch_size is None:
        # 一次性处理所有查询
        groundtruth = compute_groundtruth(
            base_vectors, 
            query_vectors, 
            k=args.k,
            metric=args.metric,
            use_gpu=args.gpu
        )
    else:
        # 批处理模式
        print(f"使用批处理模式，批大小: {args.batch_size}")
        nq = query_vectors.shape[0]
        groundtruth = np.zeros((nq, args.k), dtype=np.int32)
        
        for i in tqdm(range(0, nq, args.batch_size)):
            end_idx = min(i + args.batch_size, nq)
            batch_queries = query_vectors[i:end_idx]
            groundtruth[i:end_idx] = compute_groundtruth(
                base_vectors,
                batch_queries,
                k=args.k,
                metric=args.metric,
                use_gpu=args.gpu
            )
    
    # 保存结果
    print("-" * 60)
    print("保存groundtruth...")
    write_ivecs(output_file, groundtruth)
    
    # 验证结果
    print("-" * 60)
    print("验证结果...")
    loaded_gt = read_vecs_fast(output_file)
    assert loaded_gt.shape == groundtruth.shape, "保存的文件与原始数据形状不匹配"
    assert np.all(loaded_gt == groundtruth), "保存的文件与原始数据内容不匹配"
    print(f"验证通过！Groundtruth形状: {loaded_gt.shape}")
    
    # 显示示例
    print("\n前3个查询的top-10最近邻:")
    for i in range(min(3, loaded_gt.shape[0])):
        print(f"查询 {i}: {loaded_gt[i, :10]}")
    
    print("\n完成！")

if __name__ == '__main__':
    main()

