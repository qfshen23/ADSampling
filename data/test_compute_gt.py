#!/usr/bin/env python3
"""
测试groundtruth计算功能
在小规模合成数据上验证功能是否正常
"""

import numpy as np
import tempfile
import os
import sys

# 导入主脚本中的函数
from compute_groundtruth import read_fvecs, write_ivecs, read_ivecs, compute_groundtruth

def write_fvecs(filename, data):
    """写入 fvecs 格式文件（浮点向量）"""
    import struct
    with open(filename, 'wb') as fp:
        for row in data:
            # 写入维度
            d = struct.pack('I', len(row))
            fp.write(d)
            # 写入每个元素
            for x in row:
                a = struct.pack('f', float(x))
                fp.write(a)

def test_compute_groundtruth():
    """测试groundtruth计算功能"""
    print("=" * 60)
    print("测试 Groundtruth 计算功能")
    print("=" * 60)
    
    # 创建合成数据
    np.random.seed(42)
    n_base = 1000  # 基础数据集大小
    n_query = 10   # 查询数量
    dim = 128      # 向量维度
    k = 10         # 返回的最近邻数量
    
    print(f"\n生成测试数据...")
    print(f"基础数据集: {n_base} 个 {dim} 维向量")
    print(f"查询数据: {n_query} 个 {dim} 维向量")
    print(f"计算 top-{k} 最近邻")
    
    # 生成随机向量
    base_vectors = np.random.randn(n_base, dim).astype(np.float32)
    query_vectors = np.random.randn(n_query, dim).astype(np.float32)
    
    # 创建临时文件
    with tempfile.TemporaryDirectory() as tmpdir:
        base_file = os.path.join(tmpdir, "test_base.fvecs")
        query_file = os.path.join(tmpdir, "test_query.fvecs")
        gt_file = os.path.join(tmpdir, "test_groundtruth.ivecs")
        
        # 写入测试数据
        print(f"\n写入测试文件...")
        write_fvecs(base_file, base_vectors)
        write_fvecs(query_file, query_vectors)
        print(f"  基础数据集: {base_file}")
        print(f"  查询数据: {query_file}")
        
        # 读取测试数据
        print(f"\n读取测试数据...")
        base_loaded = read_fvecs(base_file)
        query_loaded = read_fvecs(query_file)
        
        # 验证读取的数据
        assert np.allclose(base_loaded, base_vectors), "基础数据集读取错误"
        assert np.allclose(query_loaded, query_vectors), "查询数据读取错误"
        print("✓ 数据读取验证通过")
        
        # 计算groundtruth (L2距离)
        print(f"\n计算 groundtruth (L2距离)...")
        groundtruth_l2 = compute_groundtruth(
            base_loaded, 
            query_loaded, 
            k=k, 
            metric='L2',
            use_gpu=False
        )
        
        # 验证结果形状
        assert groundtruth_l2.shape == (n_query, k), f"Groundtruth形状错误: {groundtruth_l2.shape}"
        print(f"✓ Groundtruth形状正确: {groundtruth_l2.shape}")
        
        # 验证索引范围
        assert np.all(groundtruth_l2 >= 0) and np.all(groundtruth_l2 < n_base), "索引超出范围"
        print(f"✓ 索引范围正确: [0, {n_base})")
        
        # 手动验证第一个查询的结果
        print(f"\n手动验证第一个查询...")
        query_0 = query_loaded[0]
        distances = np.linalg.norm(base_loaded - query_0, axis=1)
        expected_indices = np.argsort(distances)[:k]
        
        assert np.array_equal(groundtruth_l2[0], expected_indices), "第一个查询的结果不正确"
        print(f"✓ 手动验证通过")
        print(f"  预期前5个索引: {expected_indices[:5]}")
        print(f"  实际前5个索引: {groundtruth_l2[0, :5]}")
        
        # 测试写入和读取
        print(f"\n测试 ivecs 文件读写...")
        write_ivecs(gt_file, groundtruth_l2)
        groundtruth_loaded = read_ivecs(gt_file)
        
        assert groundtruth_loaded.shape == groundtruth_l2.shape, "读取的形状不匹配"
        assert np.array_equal(groundtruth_loaded, groundtruth_l2), "读取的内容不匹配"
        print(f"✓ ivecs 文件读写验证通过")
        
        # 测试内积距离
        print(f"\n计算 groundtruth (内积距离)...")
        groundtruth_ip = compute_groundtruth(
            base_loaded, 
            query_loaded, 
            k=k, 
            metric='IP',
            use_gpu=False
        )
        
        # 验证结果
        assert groundtruth_ip.shape == (n_query, k), f"IP Groundtruth形状错误"
        print(f"✓ IP Groundtruth形状正确: {groundtruth_ip.shape}")
        
        # IP和L2的结果应该不同（除非数据特殊）
        if not np.array_equal(groundtruth_ip, groundtruth_l2):
            print(f"✓ L2和IP的结果不同（符合预期）")
        else:
            print(f"⚠ L2和IP的结果相同（罕见情况）")
    
    print("\n" + "=" * 60)
    print("✓ 所有测试通过！")
    print("=" * 60)

def test_edge_cases():
    """测试边界情况"""
    print("\n" + "=" * 60)
    print("测试边界情况")
    print("=" * 60)
    
    # 测试k=1的情况
    print("\n测试 k=1...")
    base = np.random.randn(100, 32).astype(np.float32)
    query = np.random.randn(5, 32).astype(np.float32)
    gt = compute_groundtruth(base, query, k=1, metric='L2', use_gpu=False)
    assert gt.shape == (5, 1), "k=1时形状错误"
    print("✓ k=1 测试通过")
    
    # 测试k等于数据集大小
    print("\n测试 k=数据集大小...")
    n = 50
    base = np.random.randn(n, 16).astype(np.float32)
    query = np.random.randn(3, 16).astype(np.float32)
    gt = compute_groundtruth(base, query, k=n, metric='L2', use_gpu=False)
    assert gt.shape == (3, n), f"k=n时形状错误: {gt.shape}"
    print(f"✓ k=n={n} 测试通过")
    
    # 验证所有索引都被使用且不重复
    for i in range(3):
        indices = gt[i]
        assert len(np.unique(indices)) == n, "存在重复索引"
        assert set(indices) == set(range(n)), "索引集合不完整"
    print("✓ 索引完整性验证通过")
    
    print("\n" + "=" * 60)
    print("✓ 边界情况测试通过！")
    print("=" * 60)

if __name__ == '__main__':
    try:
        test_compute_groundtruth()
        test_edge_cases()
        print("\n" + "🎉 " * 20)
        print("所有测试成功完成！脚本功能正常。")
        print("🎉 " * 20)
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

