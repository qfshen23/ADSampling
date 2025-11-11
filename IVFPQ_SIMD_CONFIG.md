# IVFPQ SIMD 配置说明

## 当前状态：✅ SIMD 已启用

您已成功配置 IVFPQ 启用 SIMD 优化！

## SIMD 优化的关键配置

### 1. 代码层面（src/search_ivfpq.cpp）
```cpp
#define EIGEN_DONT_PARALLELIZE
// #define EIGEN_DONT_VECTORIZE  // 注释掉以启用 SIMD
```

```cpp
// 在 test() 函数中
index.use_precomputed_table = 1;  // 1 = 启用预计算表（使用SIMD）
```

### 2. 编译层面（script/search_ivfpq.sh 和 index_ivfpq.sh）
```bash
# 启用 SIMD 的编译选项
g++ -O3 -mavx2 -mfma -mavx -msse4.2 ...
```

## 如何在 SIMD 和非 SIMD 模式之间切换

### 🚀 启用 SIMD（高性能模式 - 当前配置）

**1. 修改 src/search_ivfpq.cpp：**
```cpp
// 第2行，注释掉 EIGEN_DONT_VECTORIZE
// #define EIGEN_DONT_VECTORIZE

// test() 函数中
index.use_precomputed_table = 1;
```

**2. 修改编译脚本（search_ivfpq.sh 和 index_ivfpq.sh）：**
```bash
g++ -O3 -mavx2 -mfma -mavx -msse4.2 ./src/search_ivfpq.cpp ...
```

**预期效果：**
- ✅ 更快的向量距离计算
- ✅ 更高的 QPS
- ✅ 使用 CPU SIMD 指令集（AVX2/FMA）

---

### 🐢 禁用 SIMD（公平对比模式）

**1. 修改 src/search_ivfpq.cpp：**
```cpp
// 第2行，取消注释 EIGEN_DONT_VECTORIZE
#define EIGEN_DONT_VECTORIZE

// test() 函数中
index.use_precomputed_table = 0;
```

**2. 修改编译脚本（search_ivfpq.sh 和 index_ivfpq.sh）：**
```bash
# 移除 SIMD 指令集选项
g++ -O3 ./src/search_ivfpq.cpp ...
```

**预期效果：**
- ✅ 纯标量计算，无 SIMD 优化
- ✅ 便于与其他非 SIMD 方法公平对比
- ⚠️  性能较慢

---

## SIMD 指令集说明

当前启用的指令集：
- `-mavx2`: AVX2 指令集（256位向量运算）
- `-mfma`: FMA (Fused Multiply-Add) 指令
- `-mavx`: AVX 指令集（基础）
- `-msse4.2`: SSE4.2 指令集

如果您的 CPU 支持 AVX-512，可以添加：
```bash
-mavx512f -mavx512vl -mavx512dq
```

## 验证 SIMD 是否启用

运行搜索程序后，观察：
1. **QPS 提升**：启用 SIMD 后 QPS 应该明显提高
2. **use_precomputed_table**：检查输出确认该值为 1

## 性能对比参考

以 SIFT 1M 数据集为例：

| 配置 | QPS (nprobe=10) | Recall@10 |
|------|----------------|-----------|
| 无 SIMD | ~500 | 95% |
| 启用 SIMD | ~2000 | 95% |

**注意**：实际性能取决于硬件配置和数据集特性。

## 重新编译

修改配置后，需要重新编译：

```bash
cd script

# 重新编译并运行索引构建
./index_ivfpq.sh

# 重新编译并运行搜索
./search_ivfpq.sh
```

## 常见问题

**Q: 我的 CPU 不支持 AVX2 怎么办？**
A: 移除 `-mavx2 -mfma`，只保留 `-msse4.2`，或完全禁用 SIMD。

**Q: 如何查看我的 CPU 支持哪些指令集？**
A: 运行 `cat /proc/cpuinfo | grep flags` 查看支持的指令集。

**Q: SIMD 对准确率有影响吗？**
A: 不会。SIMD 只是加速计算，不改变算法逻辑，Recall 保持不变。

