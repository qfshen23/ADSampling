#!/usr/bin/env python3
"""
数据集归一化工具
对向量数据集进行L2归一化，并保存到新的文件夹中
"""

import numpy as np
import struct
import os
import argparse
from tqdm import tqdm

source = '/data/vector_datasets/'

def read_fvecs(filename, c_contiguous=True):
    """读取 fvecs 格式文件（浮点向量）"""
    print(f"  读取文件: {filename}")
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
    print(f"  ✓ 读取了 {fv.shape[0]:,} 个 {fv.shape[1]} 维向量")
    return fv

def to_fvecs(filename, data):
    """写入 fvecs 格式文件"""
    print(f"  写入文件: {filename}")
    with open(filename, 'wb') as fp:
        for y in tqdm(data, desc="  保存进度", leave=False):
            d = struct.pack('I', len(y))
            fp.write(d)
            for x in y:
                a = struct.pack('f', x)
                fp.write(a)
    print(f"  ✓ 已保存 {len(data):,} 个向量")

def normalize_vectors(vectors):
    """
    对向量进行L2归一化
    
    参数:
        vectors: numpy数组 (N x D)
    
    返回:
        归一化后的向量 (N x D)，每个向量的L2范数为1
    """
    print(f"  归一化 {vectors.shape[0]:,} 个向量...")
    
    # 计算每个向量的L2范数
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    
    # 避免除以零
    norms = np.maximum(norms, 1e-12)
    
    # 归一化
    normalized = vectors / norms
    
    # 验证归一化结果
    check_norms = np.linalg.norm(normalized[:100], axis=1)
    avg_norm = np.mean(check_norms)
    print(f"  ✓ 归一化完成，验证范数平均值: {avg_norm:.6f} (应该≈1.0)")
    
    return normalized.astype(np.float32)

def normalize_dataset(dataset_name, source_dir, file_types=['base', 'query']):
    """
    归一化单个数据集
    
    参数:
        dataset_name: 数据集名称
        source_dir: 数据集根目录
        file_types: 要归一化的文件类型列表 (例如: ['base', 'query', 'learn'])
    """
    print("=" * 70)
    print(f"处理数据集: {dataset_name}")
    print("=" * 70)
    
    # 原始数据集路径
    original_path = os.path.join(source_dir, dataset_name)
    
    # 检查原始数据集是否存在
    if not os.path.exists(original_path):
        print(f"❌ 错误: 数据集目录不存在: {original_path}")
        return False
    
    # 创建归一化后的数据集目录
    normalized_name = f"{dataset_name}_normalized"
    normalized_path = os.path.join(source_dir, normalized_name)
    
    if os.path.exists(normalized_path):
        print(f"⚠️  警告: 目录已存在: {normalized_path}")
        response = input("是否覆盖? (y/n): ")
        if response.lower() != 'y':
            print("跳过该数据集")
            return False
    else:
        os.makedirs(normalized_path)
        print(f"✓ 创建目录: {normalized_path}")
    
    print()
    
    # 处理每种类型的文件
    for file_type in file_types:
        print(f"📁 处理 {file_type} 文件:")
        print("-" * 70)
        
        # 构建文件路径
        original_file = os.path.join(original_path, f"{dataset_name}_{file_type}.fvecs")
        normalized_file = os.path.join(normalized_path, f"{normalized_name}_{file_type}.fvecs")
        
        # 检查原始文件是否存在
        if not os.path.exists(original_file):
            print(f"  ⚠️  跳过: 文件不存在 - {original_file}")
            print()
            continue
        
        # 读取原始向量
        vectors = read_fvecs(original_file)
        
        # 显示原始向量的统计信息
        sample_norms = np.linalg.norm(vectors[:1000], axis=1)
        print(f"  原始向量范数 - 均值: {np.mean(sample_norms):.4f}, "
              f"标准差: {np.std(sample_norms):.4f}, "
              f"最小: {np.min(sample_norms):.4f}, "
              f"最大: {np.max(sample_norms):.4f}")
        
        # 归一化
        normalized_vectors = normalize_vectors(vectors)
        
        # 保存归一化后的向量
        to_fvecs(normalized_file, normalized_vectors)
        
        print()
    
    print("=" * 70)
    print(f"✅ 数据集 {dataset_name} 归一化完成!")
    print(f"   归一化后的数据保存在: {normalized_path}")
    print("=" * 70)
    print()
    
    return True

def main():
    parser = argparse.ArgumentParser(description='归一化向量数据集')
    parser.add_argument('--datasets', type=str, nargs='+', 
                        help='要归一化的数据集名称列表（例如: glove2m deep1m）')
    parser.add_argument('--source', type=str, default='/data/vector_datasets',
                        help='数据集根目录（默认: /data/vector_datasets）')
    parser.add_argument('--files', type=str, nargs='+', default=['base', 'query'],
                        help='要归一化的文件类型（默认: base query）')
    parser.add_argument('--all', action='store_true',
                        help='归一化source目录下的所有数据集')
    
    args = parser.parse_args()
    
    # 确定要处理的数据集列表
    if args.all:
        # 获取所有数据集
        datasets = [d for d in os.listdir(args.source) 
                   if os.path.isdir(os.path.join(args.source, d)) 
                   and not d.endswith('_normalized')]
        print(f"找到 {len(datasets)} 个数据集")
        print(f"数据集列表: {datasets}")
        print()
    elif args.datasets:
        datasets = args.datasets
    else:
        # 使用默认列表（可以在这里修改）
        datasets = ['glove2m', 'deep1m', 'gist']
        print("使用默认数据集列表（可以使用 --datasets 参数指定）:")
        print(f"  {datasets}")
        print()
    
    # 处理每个数据集
    success_count = 0
    fail_count = 0
    
    for dataset in datasets:
        try:
            if normalize_dataset(dataset, args.source, args.files):
                success_count += 1
            else:
                fail_count += 1
        except Exception as e:
            print(f"❌ 处理 {dataset} 时出错: {e}")
            import traceback
            traceback.print_exc()
            fail_count += 1
            print()
    
    # 总结
    print("\n" + "=" * 70)
    print("📊 归一化任务完成!")
    print("=" * 70)
    print(f"✅ 成功: {success_count} 个数据集")
    print(f"❌ 失败: {fail_count} 个数据集")
    print("=" * 70)

if __name__ == '__main__':
    # 如果直接运行脚本，使用默认配置
    import sys
    
    if len(sys.argv) == 1:
        # 没有命令行参数，使用交互式模式
        print("=" * 70)
        print("向量数据集归一化工具")
        print("=" * 70)
        print()
        print("请输入要归一化的数据集名称（用空格分隔）")
        print("示例: glove2m deep1m gist")
        print("或直接回车使用默认列表: ['glove2m']")
        print()
        
        user_input = input("数据集列表: ").strip()
        
        if user_input:
            datasets = user_input.split()
        else:
            datasets = ['glove2m']
        
        print()
        print(f"将归一化以下数据集: {datasets}")
        print(f"数据源目录: {source}")
        print()
        
        response = input("确认开始? (y/n): ")
        if response.lower() != 'y':
            print("已取消")
            sys.exit(0)
        
        print()
        
        # 处理数据集
        success_count = 0
        fail_count = 0
        
        for dataset in datasets:
            try:
                if normalize_dataset(dataset, source, ['base', 'query']):
                    success_count += 1
                else:
                    fail_count += 1
            except Exception as e:
                print(f"❌ 处理 {dataset} 时出错: {e}")
                import traceback
                traceback.print_exc()
                fail_count += 1
                print()
        
        # 总结
        print("\n" + "=" * 70)
        print("📊 归一化任务完成!")
        print("=" * 70)
        print(f"✅ 成功: {success_count} 个数据集")
        print(f"❌ 失败: {fail_count} 个数据集")
        print("=" * 70)
    else:
        # 有命令行参数，使用argparse
        main()

