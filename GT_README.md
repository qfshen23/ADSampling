# 高性能Groundtruth计算工具

这是一个用C++编写的高性能groundtruth计算工具，相比Python版本有显著的性能提升。

## 🚀 主要特性

- **SIMD加速**: 支持AVX2/SSE指令集，大幅提升距离计算速度
- **多线程并行**: 使用OpenMP实现高效的多线程并行计算
- **内存优化**: 多种算法策略，适应不同的内存和数据规模
- **高度优化**: 使用-O3编译优化和CPU特定指令
- **易于使用**: 简单的命令行接口

## 📊 性能对比

相比Python版本的预期性能提升：
- **距离计算**: 5-10x 提升（SIMD加速）
- **整体性能**: 10-50x 提升（多线程+优化）
- **内存使用**: 更高效的内存管理

## 🛠️ 编译

### 方法1: 使用构建脚本（推荐）
```bash
./build_gt.sh
```

### 方法2: 使用Makefile
```bash
make -f Makefile.gt
```

### 方法3: 手动编译
```bash
g++ -std=c++17 -O3 -march=native -mtune=native -fopenmp \
    -mavx2 -mfma -Wall -Wextra -DNDEBUG \
    -o compute_gt src/compute_gt.cpp
```

## 📖 使用方法

### 基本用法
```bash
./compute_gt -b base.fvecs -q query.fvecs -o groundtruth.ivecs -k 100
```

### 完整参数
```bash
./compute_gt \
    --base /data/vector_datasets/sift/sift_base.fvecs \
    --query /data/vector_datasets/sift/sift_query.fvecs \
    --output sift_groundtruth.ivecs \
    --topk 100 \
    --threads 8
```

### 参数说明
- `-b, --base`: 基础向量文件（fvecs格式）
- `-q, --query`: 查询向量文件（fvecs格式）
- `-o, --output`: 输出groundtruth文件（ivecs格式）
- `-k, --topk`: 近邻数量（默认100）
- `-t, --threads`: 线程数（默认自动检测）
- `-h, --help`: 显示帮助信息

## 🎯 批量处理示例

使用提供的脚本批量处理多个数据集：

```bash
# 处理单个数据集
./run_gt_examples.sh sift

# 处理所有数据集
./run_gt_examples.sh all

# 自定义参数
./run_gt_examples.sh sift 50 16  # k=50, 16线程
```

## 🔧 算法优化

### SIMD加速
- **AVX2**: 8个float并行计算，支持FMA指令
- **SSE**: 4个float并行计算（兼容性更好）
- **自动检测**: 根据CPU能力自动选择最佳SIMD指令

### 多种Top-K算法
1. **堆算法**: 适用于小k值（k < nb/100）
2. **部分排序**: 适用于大k值
3. **分块处理**: 内存友好版本

### 内存优化
- 缓存友好的数据访问模式
- 动态选择最优算法
- 减少内存分配和拷贝

## 📈 性能测试

### 系统信息检查
```bash
make -f Makefile.gt info
```

### 性能基准测试
```bash
make -f Makefile.gt benchmark
```

### 单次测试
```bash
make -f Makefile.gt test
```

## 🔍 支持的数据格式

### 输入格式（fvecs）
- 每个向量前4字节为维度（int32）
- 后续为向量数据（float32）

### 输出格式（ivecs）
- 每行前4字节为k值（int32）
- 后续为k个最近邻索引（int32）

## ⚡ 性能调优建议

1. **编译优化**:
   - 使用`-march=native`针对当前CPU优化
   - 确保启用AVX2支持

2. **线程数设置**:
   - 通常设为CPU核心数
   - 对于大数据集可以设为核心数的1.5-2倍

3. **内存考虑**:
   - 确保有足够内存加载完整数据集
   - 对于超大数据集考虑使用分块处理

## 🐛 故障排除

### 编译错误
- 确保g++版本支持C++17
- 检查是否安装了OpenMP
- 验证CPU是否支持所需的SIMD指令

### 运行时错误
- 检查输入文件格式是否正确
- 确保有足够的内存
- 验证文件路径是否正确

### 性能问题
- 检查是否启用了编译优化
- 调整线程数
- 监控内存使用情况

## 📝 示例输出

```
=== High-Performance Groundtruth Computation ===
Base file: /data/vector_datasets/sift/sift_base.fvecs
Query file: /data/vector_datasets/sift/sift_query.fvecs
Output file: sift_groundtruth.ivecs
k: 100
Threads: 8

Loading base vectors...
Reading 1000000 vectors of dimension 128 from /data/vector_datasets/sift/sift_base.fvecs
Loading query vectors...
Reading 10000 vectors of dimension 128 from /data/vector_datasets/sift/sift_query.fvecs
Data loaded successfully!
Base: 1000000 vectors, Query: 10000 vectors, Dim: 128

Computing groundtruth with k=100 using 8 threads...
Using AVX2 SIMD acceleration
Using heap-based algorithm
Processed 0/10000 queries
Processed 100/10000 queries
...
Computation completed in 15420 ms
Average time per query: 1.542 ms
Writing results to sift_groundtruth.ivecs...
Groundtruth computation completed successfully!
```

这个高性能版本应该比Python版本快10-50倍，特别是对于大规模数据集！
