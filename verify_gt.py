#!/usr/bin/env python3
import numpy as np
import sys

def read_ivecs(filename):
    """读取 ivecs 格式文件"""
    with open(filename, 'rb') as f:
        data = []
        while True:
            # 读取维度
            dim_bytes = f.read(4)
            if len(dim_bytes) != 4:
                break
            dim = np.frombuffer(dim_bytes, dtype=np.int32)[0]
            
            # 读取向量
            vec = np.frombuffer(f.read(4 * dim), dtype=np.int32)
            data.append(vec)
        
        return np.array(data)

def read_fvecs(filename):
    """读取 fvecs 格式文件"""
    with open(filename, 'rb') as f:
        data = []
        while True:
            # 读取维度
            dim_bytes = f.read(4)
            if len(dim_bytes) != 4:
                break
            dim = np.frombuffer(dim_bytes, dtype=np.int32)[0]
            
            # 读取向量
            vec = np.frombuffer(f.read(4 * dim), dtype=np.float32)
            data.append(vec)
        
        return np.array(data)

def verify_groundtruth(gt_file):
    """验证 ground truth 文件"""
    print(f"正在读取 ground truth 文件: {gt_file}")
    gt = read_ivecs(gt_file)
    
    print(f"Ground truth 形状: {gt.shape}")
    print(f"查询数量: {gt.shape[0]}, k值: {gt.shape[1]}")
    
    # 检查前几个查询
    num_check = min(5, gt.shape[0])
    print(f"\n检查前 {num_check} 个查询的结果:")
    
    for i in range(num_check):
        neighbors = gt[i]
        print(f"\n查询 {i}:")
        print(f"  前10个邻居 ID: {neighbors[:10]}")
        print(f"  最小 ID: {neighbors.min()}, 最大 ID: {neighbors.max()}")
        print(f"  ID 范围: [{neighbors.min()}, {neighbors.max()}]")
        
        # 检查是否都是小于100的数字
        small_ids = np.sum(neighbors < 100)
        print(f"  小于100的 ID 数量: {small_ids}/{len(neighbors)}")
        
        # 检查唯一值数量
        unique_ids = len(np.unique(neighbors))
        print(f"  唯一 ID 数量: {unique_ids}/{len(neighbors)}")
    
    # 统计所有查询
    print(f"\n总体统计:")
    all_ids = gt.flatten()
    print(f"  所有 ID 的最小值: {all_ids.min()}")
    print(f"  所有 ID 的最大值: {all_ids.max()}")
    print(f"  小于100的 ID 比例: {np.sum(all_ids < 100) / len(all_ids) * 100:.2f}%")
    
    # 检查是否存在异常
    for i in range(gt.shape[0]):
        neighbors = gt[i]
        if np.sum(neighbors < 100) > gt.shape[1] * 0.9:  # 超过90%都小于100
            print(f"\n警告: 查询 {i} 有 {np.sum(neighbors < 100)} 个邻居 ID 小于 100!")
            if i < 3:
                print(f"  该查询的所有邻居: {neighbors}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("用法: python verify_gt.py <groundtruth.ivecs>")
        sys.exit(1)
    
    verify_groundtruth(sys.argv[1])

