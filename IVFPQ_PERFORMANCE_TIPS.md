# IVFPQ 性能问题诊断与优化

## 🔍 已发现的问题

### 问题 1：测试方式不一致 ✅ 已修复

**原因：**
- IVF：逐查询搜索和计时（每个查询单独调用 `search()`）
- IVFPQ（旧版）：批量搜索所有查询（一次调用 `search(nq, ...)`）

**批量搜索的问题：**
- 包含额外的内存分配和管理开销
- 初始化成本被平摊到每个查询
- 无法准确反映单查询性能

**修复方案：** ✅ 已改为逐查询搜索

---

## 🚀 IVFPQ 性能优化建议

### 1. Polysemous 编码（可选优化）

IVFPQ 支持 Polysemous 优化，可以加速搜索：

```cpp
// 在索引构建后或加载后设置
index.polysemous_ht = 54;  // 推荐值：54 或 64
```

**添加位置：** `src/search_ivfpq.cpp` 中加载索引后

```cpp
faiss::IndexIVFPQ* ivfpq_index = dynamic_cast<faiss::IndexIVFPQ*>(base_index);

// 添加这一行
ivfpq_index->polysemous_ht = 54;
```

### 2. 扫描表类型（scan_table_threshold）

```cpp
// 设置扫描阈值
index.scan_table_threshold = 0;  // 0 = 总是使用扫描表
```

### 3. 预计算表策略

当前设置：`use_precomputed_table = 1`

不同值的影响：
- `0`：不使用预计算表（省内存，较慢）
- `1`：使用预计算表（默认，平衡）
- `2`：使用完整预计算表（最快，但内存大）

**小 nprobe 时建议：**
```cpp
if(nprobe <= 10) {
    index.use_precomputed_table = 2;  // 小 nprobe 用完整表
} else {
    index.use_precomputed_table = 1;  // 大 nprobe 用默认
}
```

### 4. 并行扫描（如果允许多线程）

```cpp
index.parallel_mode = 0;  // 单线程模式
// index.parallel_mode = 1;  // 多线程模式（需要允许多线程时）
```

---

## 🔧 推荐的优化代码

在 `src/search_ivfpq.cpp` 的 `test()` 函数中：

```cpp
void test(const Matrix<float> &Q, const Matrix<unsigned> &G, faiss::IndexIVFPQ &index, int k, int nprobe){
    // ... existing code ...
    
    // 设置 nprobe
    index.nprobe = nprobe;
    
    // 优化设置
    if(nprobe <= 10) {
        index.use_precomputed_table = 2;  // 小 nprobe 使用完整预计算表
    } else {
        index.use_precomputed_table = 1;  // 大 nprobe 使用默认
    }
    
    // 可选：启用 Polysemous 优化
    index.polysemous_ht = 54;
    
    // 并行模式（单线程）
    index.parallel_mode = 0;
    
    // ... rest of the code ...
}
```

---

## 📊 性能对比预期

以 SIFT 1M 为例，优化后预期结果：

| Method | nprobe=5 | nprobe=10 | nprobe=20 |
|--------|----------|-----------|-----------|
| IVF (精确距离) | ~1000 QPS | ~800 QPS | ~500 QPS |
| IVFPQ (优化前) | ~800 QPS ❌ | ~600 QPS | ~400 QPS |
| IVFPQ (优化后) | ~2000 QPS ✅ | ~1500 QPS | ~1000 QPS |

**IVFPQ 应该比 IVF 快的原因：**
1. PQ 编码后的向量更小（通常 8-32 字节 vs 128-512 字节）
2. 距离计算使用查表而非浮点运算
3. 更好的缓存局部性

---

## 🐛 如果 IVFPQ 仍然慢的原因

### 1. 索引质量问题
```bash
# 检查索引统计
# 确保：
# - M 和 dimension 匹配（dim % M == 0）
# - nlist 合理（通常 sqrt(N) 到 4*sqrt(N)）
# - 训练数据充足（至少 30*nlist 个向量）
```

### 2. PQ 编码参数不当
```bash
# 当前配置（script/index_ivfpq.sh）：
NLIST=2048
M=20
NBITS=8

# 建议检查：
# - M=20 对于某些维度可能不是最优
# - 尝试 M=16 或 M=32
```

### 3. 数据集特性
某些数据集 PQ 效果不佳：
- 高维稀疏数据
- 非均匀分布的数据

---

## 🔬 调试步骤

### 1. 添加详细计时
在 `test()` 函数中添加：

```cpp
cout << "Precomputed table: " << index.use_precomputed_table << endl;
cout << "Polysemous: " << index.polysemous_ht << endl;
cout << "PQ M: " << index.pq.M << endl;
cout << "PQ nbits: " << index.pq.nbits << endl;
```

### 2. 对比批量 vs 逐查询
测试两种模式的性能差异

### 3. 检查 Recall
确保 IVFPQ 的 Recall 合理（通常略低于 IVF）

---

## ✅ 已修复清单

- ✅ 改为逐查询搜索（与 IVF 一致）
- ✅ 逐查询计时（更准确）
- ✅ 预热改为逐查询
- ⏳ 待测试：Polysemous 优化
- ⏳ 待测试：动态预计算表策略

---

## 🎯 下一步

1. **重新编译并测试**
   ```bash
   cd script
   ./search_ivfpq.sh
   ```

2. **对比结果**
   - 检查 IVFPQ 是否比 IVF 快
   - 特别关注小 nprobe 的场景

3. **如果仍然慢**，尝试添加 Polysemous 优化或调整 `use_precomputed_table`

4. **收集数据**
   - 记录不同 nprobe 下的 QPS 和 Recall
   - 对比 IVF vs IVFPQ 的性能曲线

