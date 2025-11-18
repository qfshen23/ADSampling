#!/usr/bin/env python3
"""
检查 fvecs/bvecs 文件的完整性和格式
"""

import struct
import os
import sys

def check_fvecs_file(filename):
    """检查 fvecs 文件"""
    print(f"检查文件: {filename}")
    print("=" * 70)
    
    if not os.path.exists(filename):
        print(f"❌ 文件不存在: {filename}")
        return False
    
    file_size = os.path.getsize(filename)
    print(f"文件大小: {file_size:,} 字节 ({file_size / (1024**3):.2f} GB)")
    
    try:
        with open(filename, 'rb') as f:
            # 读取第一个维度
            dim_bytes = f.read(4)
            if len(dim_bytes) < 4:
                print(f"❌ 文件太小，无法读取维度")
                return False
            
            dim = struct.unpack('i', dim_bytes)[0]
            print(f"读取的维度: {dim}")
            
            # 验证维度合理性
            if dim <= 0:
                print(f"❌ 维度异常: {dim} (应该 > 0)")
                return False
            
            if dim > 100000:
                print(f"❌ 维度过大: {dim} (应该 < 100000)")
                print("   文件可能损坏或格式不正确")
                return False
            
            print(f"✓ 维度合理: {dim}")
            
            # 计算向量数量
            if filename.endswith('.fvecs'):
                dtype_size = 4  # float32
                dtype_name = "float32"
            elif filename.endswith('.bvecs'):
                dtype_size = 1  # uint8
                dtype_name = "uint8"
            else:
                print("⚠️  未知文件类型（期望 .fvecs 或 .bvecs）")
                dtype_size = 4
                dtype_name = "unknown"
            
            bytes_per_vector = 4 + dim * dtype_size
            
            if file_size % bytes_per_vector != 0:
                print(f"❌ 文件大小不匹配:")
                print(f"   每个向量大小: {bytes_per_vector} 字节")
                print(f"   文件大小: {file_size} 字节")
                print(f"   余数: {file_size % bytes_per_vector} 字节")
                print(f"   文件可能损坏")
                return False
            
            n_vectors = file_size // bytes_per_vector
            print(f"✓ 向量数量: {n_vectors:,}")
            print(f"✓ 数据类型: {dtype_name}")
            
            # 计算所需内存
            memory_needed = n_vectors * dim * 4  # 转换为 float32
            print(f"✓ 需要内存: {memory_needed / (1024**3):.2f} GB")
            
            if memory_needed > 50 * (1024**3):
                print(f"⚠️  警告: 需要超过 50GB 内存!")
                print(f"   建议使用分块处理或增加内存限制")
            
            # 验证几个向量的维度
            print("\n验证向量维度...")
            f.seek(0)
            sample_positions = [0, n_vectors // 2, n_vectors - 1]
            
            for pos in sample_positions:
                if pos >= n_vectors:
                    continue
                
                f.seek(pos * bytes_per_vector)
                vec_dim_bytes = f.read(4)
                if len(vec_dim_bytes) < 4:
                    print(f"❌ 无法读取向量 {pos} 的维度")
                    return False
                
                vec_dim = struct.unpack('i', vec_dim_bytes)[0]
                if vec_dim != dim:
                    print(f"❌ 向量 {pos} 维度不一致: {vec_dim} (期望 {dim})")
                    return False
            
            print(f"✓ 抽样验证通过 (位置: {sample_positions})")
            
            print("\n" + "=" * 70)
            print("✅ 文件格式检查通过!")
            print("=" * 70)
            return True
            
    except Exception as e:
        print(f"❌ 检查过程出错: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    if len(sys.argv) < 2:
        print("用法: python check_fvecs_file.py <文件路径>")
        print("示例: python check_fvecs_file.py /data/vector_datasets/deep1m/deep1m_base.fvecs")
        sys.exit(1)
    
    filename = sys.argv[1]
    success = check_fvecs_file(filename)
    
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()

