# IVFPQ 索引与搜索说明

## 文件说明

已创建以下文件用于 IVFPQ 索引构建和搜索：

### 源代码文件
1. `src/index_ivfpq.cpp` - IVFPQ 索引构建的 C++ 代码
2. `src/search_ivfpq.cpp` - IVFPQ 搜索的 C++ 代码

### 执行脚本
3. `script/index_ivfpq.sh` - 索引构建脚本
4. `script/search_ivfpq.sh` - 搜索脚本

## 主要特性

### 单线程控制
- 代码中使用 `omp_set_num_threads(1)` 强制单线程执行
- 在 `search_ivfpq.cpp` 中定义了 `EIGEN_DONT_PARALLELIZE` 和 `EIGEN_DONT_VECTORIZE`

### 禁用 SIMD
- 使用 `index.use_precomputed_table = 0` 禁用预计算表（避免 SIMD 优化）
- 编译选项使用 `-O3` 但不包含 AVX 等 SIMD 指令集

## 使用方法

### 1. 构建索引

```bash
cd script
./index_ivfpq.sh
```

**可配置参数**（在脚本中修改）：
- `NLIST`: IVF 聚类数量（默认 1024）
- `M`: PQ 子空间数量（默认 16）
- `NBITS`: 每个子空间的编码位数（默认 8）
- `TRAIN_SIZE`: 训练数据量，0 表示使用全部数据
- `datasets`: 数据集列表

**命令行参数**（直接运行 index_ivfpq）：
```bash
./src/index_ivfpq -d <data_path> -i <index_path> [-n nlist] [-m M] [-b nbits] [-t train_size]
```

### 2. 执行搜索

```bash
cd script
./search_ivfpq.sh
```

**可配置参数**（在脚本中修改）：
- `K`: 返回的 top-K 结果（默认 10）
- `NPROBE_LIST`: 探测的聚类数量列表（默认 5, 10, 15, ..., 50）
- `datasets`: 数据集列表

**命令行参数**（直接运行 search_ivfpq）：
```bash
./src/search_ivfpq -i <index_path> -q <query_path> -g <groundtruth_path> [-k K] [-p nprobe] [-r result_path]
```

## 参数说明

### 索引参数
- `-n, --nlist`: IVF 聚类中心数量
- `-m, --M`: PQ 子空间数量（维度必须能被 M 整除）
- `-b, --nbits`: 每个子空间的编码位数（通常为 8）
- `-d, --data_path`: 基础向量数据路径（.fvecs 格式）
- `-i, --index_path`: 索引输出路径
- `-t, --train_size`: 训练数据量（0 表示使用全部）

### 搜索参数
- `-k, --K`: 返回的近邻数量
- `-p, --nprobe`: 探测的聚类数量（越大召回率越高但速度越慢）
- `-i, --index_path`: 索引文件路径
- `-q, --query_path`: 查询向量路径（.fvecs 格式）
- `-g, --groundtruth_path`: 真值路径（.ivecs 格式）
- `-r, --result_path`: 结果输出路径

## 输出结果

搜索结果将包括：
- **Recall@K**: 召回率（百分比）
- **Time per query**: 每个查询的平均时间（微秒）
- **QPS**: 每秒查询数

结果会保存到 `results/` 目录下的日志文件中。

## 依赖项

需要安装以下库：
- Faiss (C++ 版本)
- OpenMP
- Eigen3

## 编译说明

脚本会自动编译代码。如需手动编译：

```bash
# 编译索引程序
g++ -O3 ./src/index_ivfpq.cpp -o ./src/index_ivfpq \
    -I ./src/ -I /usr/include/eigen3 -lfaiss -fopenmp

# 编译搜索程序
g++ -O3 ./src/search_ivfpq.cpp -o ./src/search_ivfpq \
    -I ./src/ -I /usr/include/eigen3 -lfaiss -fopenmp
```

## 与 IVF 代码的一致性

本代码参考了现有的 IVF 代码结构：
- 使用相同的命令行参数解析风格
- 使用相同的 Matrix 类读取数据
- 使用相同的时间统计方式
- 输出格式与 IVF 搜索保持一致

## 注意事项

1. 确保维度 `d` 能被 `M` 整除
2. `nlist` 的选择影响索引质量，通常设置为 `sqrt(N)` 到 `4*sqrt(N)` 之间，其中 N 是数据集大小
3. `nprobe` 应该远小于 `nlist`，通常在 1-100 之间
4. 单线程模式下性能会比多线程慢，但便于公平对比

