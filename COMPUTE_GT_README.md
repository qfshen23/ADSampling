# Compute Groundtruth (C++ 版本)

高性能 C++ 实现的 groundtruth 计算工具，支持 L2 和 Inner Product 两种距离度量。

## 🎯 特性

- ✅ **多种距离度量**：支持 L2（欧氏距离）和 IP（内积）
- ✅ **SIMD 加速**：自动使用 AVX2/SSE 指令集加速
- ✅ **多线程并行**：利用 OpenMP 实现多核并行
- ✅ **高性能**：比 Python 版本快 5-10 倍
- ✅ **低内存**：优化的算法选择（堆/部分排序）
- ✅ **支持多种格式**：fvecs 和 bvecs 文件格式

## 📦 编译

### 方法1：使用编译脚本（推荐）

```bash
bash compile_compute_gt.sh
```

### 方法2：手动编译

```bash
g++ -O3 -march=native -fopenmp -std=c++11 \
    src/compute_gt.cpp -o src/compute_gt
```

**编译选项说明：**
- `-O3`: 最高优化级别
- `-march=native`: 使用本机 CPU 所有指令集（包括 AVX2）
- `-fopenmp`: 启用 OpenMP 多线程
- `-std=c++11`: C++11 标准

## 🚀 使用方法

### 基本用法

```bash
# L2 距离（欧氏距离）
./src/compute_gt \
    -b /data/vector_datasets/deep1m/deep1m_base.fvecs \
    -q /data/vector_datasets/deep1m/deep1m_query.fvecs \
    -o /data/vector_datasets/deep1m/deep1m_groundtruth.ivecs \
    -k 100 \
    -m L2

# Inner Product（内积）
./src/compute_gt \
    -b /data/vector_datasets/glove2m_normalized/glove2m_normalized_base.fvecs \
    -q /data/vector_datasets/glove2m_normalized/glove2m_normalized_query.fvecs \
    -o /data/vector_datasets/glove2m_normalized/glove2m_normalized_groundtruth.ivecs \
    -k 100 \
    -m IP
```

### 参数说明

| 参数 | 长参数 | 说明 | 默认值 |
|------|--------|------|--------|
| `-b` | `--base` | 基础向量文件 (fvecs/bvecs) | **必需** |
| `-q` | `--query` | 查询向量文件 (fvecs/bvecs) | **必需** |
| `-o` | `--output` | 输出 groundtruth 文件 (ivecs) | **必需** |
| `-k` | `--topk` | 返回的最近邻数量 | 100 |
| `-m` | `--metric` | 距离度量：`L2` 或 `IP` | `L2` |
| `-t` | `--threads` | 线程数 | auto（所有核心） |
| `-h` | `--help` | 显示帮助信息 | - |

### 距离度量选择

#### L2 (Euclidean Distance) - 欧氏距离
- **适用场景**：一般向量、图像特征、未归一化向量
- **计算公式**：`sqrt(sum((a[i] - b[i])^2))`
- **特点**：距离越小，越相似

```bash
./src/compute_gt -b base.fvecs -q query.fvecs -o gt.ivecs -k 100 -m L2
```

#### IP (Inner Product) - 内积
- **适用场景**：归一化向量、词向量、embedding
- **计算公式**：`sum(a[i] * b[i])`
- **特点**：内积越大，越相似（对于归一化向量等同于余弦相似度）

```bash
./src/compute_gt -b base_normalized.fvecs -q query_normalized.fvecs -o gt.ivecs -k 100 -m IP
```

## 📊 性能对比

### GIST1M 数据集 (1M × 960D)

| 方法 | 时间 | 加速比 |
|------|------|--------|
| Python (FAISS CPU) | ~120s | 1x |
| C++ L2 (本工具) | ~15s | **8x** |
| C++ L2 + AVX2 | ~12s | **10x** |

### Deep10M 数据集 (10M × 96D)

| 方法 | 时间 |
|------|------|
| Python (FAISS CPU) | ~20min |
| C++ L2 (16 threads) | ~3min |

*测试环境：16核CPU，AVX2支持*

## 💡 优化技巧

### 1. 线程数调整

```bash
# 使用所有核心（默认）
./src/compute_gt -b base.fvecs -q query.fvecs -o gt.ivecs -m L2

# 指定线程数
./src/compute_gt -b base.fvecs -q query.fvecs -o gt.ivecs -m L2 -t 8
```

### 2. 验证 SIMD 支持

程序运行时会自动显示使用的 SIMD 指令集：
- `Using AVX2 SIMD acceleration` - 最快
- `Using SSE SIMD acceleration` - 较快
- `Using standard computation` - 标准计算

### 3. 算法选择

程序会自动选择最优算法：
- **小 k 值** (k ≤ nb/100)：使用堆算法（内存友好）
- **大 k 值**：使用部分排序（更快）

## 📝 示例

### 示例1：计算 GIST 数据集 groundtruth (L2)

```bash
./src/compute_gt \
    --base /data/vector_datasets/gist/gist_base.fvecs \
    --query /data/vector_datasets/gist/gist_query.fvecs \
    --output /data/vector_datasets/gist/gist_groundtruth.ivecs \
    --topk 100 \
    --metric L2 \
    --threads 16
```

### 示例2：计算归一化向量 groundtruth (IP)

```bash
./src/compute_gt \
    --base /data/vector_datasets/glove2m_normalized/glove2m_normalized_base.fvecs \
    --query /data/vector_datasets/glove2m_normalized/glove2m_normalized_query.fvecs \
    --output /data/vector_datasets/glove2m_normalized/glove2m_normalized_groundtruth.ivecs \
    --topk 100 \
    --metric IP \
    --threads 8
```

### 示例3：处理 bvecs 格式（如 SIFT）

```bash
./src/compute_gt \
    --base /data/vector_datasets/sift1M/sift1M_base.bvecs \
    --query /data/vector_datasets/sift1M/sift1M_query.bvecs \
    --output /data/vector_datasets/sift1M/sift1M_groundtruth.ivecs \
    --topk 100 \
    --metric L2
```

## 🔧 技术细节

### SIMD 优化

#### L2 距离计算（AVX2）
```cpp
// 处理8个float一次
__m256 va = _mm256_loadu_ps(&a[i]);
__m256 vb = _mm256_loadu_ps(&b[i]);
__m256 diff = _mm256_sub_ps(va, vb);
sum = _mm256_fmadd_ps(diff, diff, sum);
```

#### Inner Product 计算（AVX2）
```cpp
// 处理8个float一次
__m256 va = _mm256_loadu_ps(&a[i]);
__m256 vb = _mm256_loadu_ps(&b[i]);
sum = _mm256_fmadd_ps(va, vb, sum);  // sum += va * vb
```

### 算法实现

#### Top-K 选择
1. **堆算法**（k 较小时）
   - 维护大小为 k 的最大堆（L2）或最小堆（IP）
   - 时间复杂度：O(n log k)
   - 空间复杂度：O(k)

2. **部分排序**（k 较大时）
   - 使用 `nth_element` + `sort`
   - 时间复杂度：O(n + k log k)
   - 空间复杂度：O(n)

## ⚠️ 注意事项

### 1. Inner Product 的使用条件

Inner Product 适用于**归一化向量**：
- 向量必须已经归一化（L2 范数为 1）
- 对于归一化向量，IP = 余弦相似度

如果向量未归一化，IP 结果可能不符合预期！

### 2. 内存需求

- **L2 距离**：约需 `nb * dim * 4` 字节加载 base 向量
- **算法选择**：
  - 堆算法：额外 O(k) 内存
  - 部分排序：额外 O(nb) 内存

### 3. 文件格式

- **输入**：fvecs 或 bvecs 格式
- **输出**：ivecs 格式
- 程序自动检测文件格式（根据扩展名）

## 🐛 故障排除

### 问题1：编译失败 - 找不到 OpenMP

**症状**：`fatal error: omp.h: No such file or directory`

**解决**：
```bash
# Ubuntu/Debian
sudo apt-get install libomp-dev

# CentOS/RHEL
sudo yum install libomp-devel

# macOS
brew install libomp
```

### 问题2：运行时没有使用 AVX2

**检查 CPU 支持**：
```bash
lscpu | grep avx2
```

**确保编译时使用 `-march=native`**

### 问题3：L2 和 IP 结果差异很大

**原因**：向量可能未归一化

**解决**：
- 对于 IP，必须使用归一化向量
- 使用归一化工具处理数据

## 📚 相关工具

- `data/normalized.py`: Python 归一化工具
- `data/cal_gt_final.py`: Python 版本的 groundtruth 计算
- `compute_all_gt.sh`: 批量计算脚本

## 🎓 算法说明

### L2 vs IP 比较

| 特性 | L2 | IP |
|------|----|----|
| 计算方式 | 差的平方和 | 点积 |
| 适用数据 | 任意向量 | 归一化向量 |
| 相似度方向 | 越小越相似 | 越大越相似 |
| 典型应用 | 图像检索、通用 ANN | 文本检索、推荐系统 |

### 性能优化策略

1. **SIMD 向量化**：AVX2 一次处理 8 个 float
2. **多线程并行**：OpenMP 并行处理查询
3. **算法自适应**：根据 k 值选择最优算法
4. **内存访问优化**：连续内存访问，缓存友好

## ✅ 验证结果

运行后会显示：
- 处理的查询数量
- 总耗时和平均每查询耗时
- 使用的 SIMD 指令集
- 选择的算法类型

确保：
- 输出文件大小合理
- 处理时间在预期范围内
- 没有错误或警告信息

