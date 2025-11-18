# Compute_GT 故障排除指南

## 错误: std::length_error - vector::_M_default_append

### 错误描述
```
terminate called after throwing an instance of 'std::length_error'
  what():  vector::_M_default_append
Aborted (core dumped)
```

### 原因分析

这个错误通常是因为程序尝试分配过大的内存导致的。可能的原因：

1. **文件格式错误**：文件不是标准的 fvecs/bvecs 格式
2. **维度值异常**：读取到的维度值不正确（例如负数或超大值）
3. **文件损坏**：文件内容损坏
4. **内存不足**：系统内存不足以加载整个数据集

### 解决步骤

#### 步骤1: 检查文件格式

使用提供的检查脚本：

```bash
python check_fvecs_file.py /data/vector_datasets/msmarco20m/msmarco20m_base.fvecs
```

**正常输出示例**：
```
检查文件: /data/vector_datasets/deep1m/deep1m_base.fvecs
======================================================================
文件大小: 513,515,520 字节 (0.48 GB)
读取的维度: 96
✓ 维度合理: 96
✓ 向量数量: 1,000,000
✓ 数据类型: float32
✓ 需要内存: 0.36 GB
✓ 抽样验证通过 (位置: [0, 500000, 999999])
======================================================================
✅ 文件格式检查通过!
```

**异常输出示例**：
```
❌ 维度异常: 1234567890 (应该 > 0)
   文件可能损坏或格式不正确
```

#### 步骤2: 重新编译程序

修复后的代码包含更严格的验证，重新编译：

```bash
bash compile_compute_gt.sh
```

或手动编译：
```bash
g++ -O3 -march=native -fopenmp -std=c++11 \
    src/compute_gt.cpp -o src/compute_gt
```

#### 步骤3: 运行修复后的程序

```bash
./src/compute_gt \
    -b /data/vector_datasets/msmarco20m/msmarco20m_base.fvecs \
    -q /data/vector_datasets/msmarco20m/msmarco20m_query.fvecs \
    -o /data/vector_datasets/msmarco20m/msmarco20m_groundtruth.ivecs \
    -k 100 \
    -m L2 \
    -t 32
```

#### 步骤4: 查看错误信息

修复后的程序会显示详细的错误信息：

**维度错误**：
```
Invalid dimension: 1234567890 (expected 1-100000)
File may be corrupted or in wrong format: xxx.fvecs
```

**文件大小不匹配**：
```
File size mismatch. File size: 12345678, bytes per vector: 400
File may be corrupted: xxx.fvecs
```

**内存不足**：
```
Error: Would allocate too much memory: 75.5 GB
Maximum allowed: 50.0 GB
```

### 常见问题及解决方案

#### 问题1: 文件损坏

**症状**：
- 维度值异常大或为负数
- 文件大小与计算不匹配

**解决方案**：
1. 重新下载或生成数据文件
2. 检查文件传输过程是否完整
3. 使用 `md5sum` 验证文件完整性

```bash
md5sum /data/vector_datasets/xxx/xxx_base.fvecs
```

#### 问题2: 内存限制

**症状**：
```
Error: Would allocate too much memory: 75.5 GB
Maximum allowed: 50.0 GB
```

**解决方案**：

修改 `compute_gt.cpp` 中的内存限制：

找到这行：
```cpp
size_t max_size = 50ULL * 1024 * 1024 * 1024;  // 50 GB
```

修改为更大的值（例如 100GB）：
```cpp
size_t max_size = 100ULL * 1024 * 1024 * 1024;  // 100 GB
```

然后重新编译。

**或者**，使用内存映射版本（如果可用）。

#### 问题3: 文件格式不是标准 fvecs

**症状**：
- 第一个整数不是维度
- 文件结构与标准不符

**解决方案**：

检查文件实际格式：

```bash
# 查看文件前 20 字节（十六进制）
hexdump -C /path/to/file.fvecs | head -n 5

# 标准 fvecs 格式应该显示：
# 00000000  60 00 00 00  xx xx xx xx  ...
#           ^^^^^^^^^
#           这是维度 (96 = 0x60)
```

如果格式不对，需要转换文件格式。

#### 问题4: 系统内存不足

**症状**：
- 即使文件正确，仍然 Abort
- `dmesg` 显示 OOM (Out of Memory)

**检查系统内存**：
```bash
free -h
```

**解决方案**：
1. 增加系统内存
2. 使用分块处理版本（如果可用）
3. 减少并行线程数：`-t 8` 而不是 `-t 32`

### 诊断命令速查表

```bash
# 1. 检查文件格式
python check_fvecs_file.py <文件路径>

# 2. 查看文件十六进制内容
hexdump -C <文件路径> | head -n 20

# 3. 查看系统内存
free -h

# 4. 查看系统日志（OOM 错误）
dmesg | tail -50

# 5. 检查文件大小
ls -lh <文件路径>

# 6. 验证文件完整性
md5sum <文件路径>
```

### 修复后的改进

新版本的 `compute_gt.cpp` 包含以下改进：

✅ **维度验证**：检查维度范围 (1-100000)  
✅ **文件大小验证**：确保文件大小与向量数量匹配  
✅ **内存限制**：防止分配超过 50GB 内存  
✅ **异常捕获**：捕获 `std::bad_alloc` 异常  
✅ **详细错误信息**：显示具体的错误原因  
✅ **进度显示**：显示文件加载进度  

### 预防措施

#### 1. 使用文件验证脚本

在运行 `compute_gt` 之前，先验证文件：

```bash
#!/bin/bash
for file in base query; do
    python check_fvecs_file.py "/path/to/${file}.fvecs"
    if [ $? -ne 0 ]; then
        echo "文件验证失败: ${file}.fvecs"
        exit 1
    fi
done
```

#### 2. 监控内存使用

使用 `htop` 或 `watch` 监控内存：

```bash
# 终端1：运行程序
./src/compute_gt ...

# 终端2：监控内存
watch -n 1 free -h
```

#### 3. 使用测试数据

先在小数据集上测试：

```bash
# 测试 SIFT1M (较小)
./src/compute_gt \
    -b /data/vector_datasets/sift1M/sift1M_base.fvecs \
    -q /data/vector_datasets/sift1M/sift1M_query.fvecs \
    -o test_output.ivecs \
    -k 100 -m L2
```

如果小数据集正常，再处理大数据集。

### 联系支持

如果问题仍未解决，请提供以下信息：

1. 文件检查脚本的完整输出
2. `compute_gt` 的完整错误信息
3. 系统信息：
   ```bash
   uname -a
   free -h
   g++ --version
   ```
4. 文件的前 100 字节：
   ```bash
   hexdump -C file.fvecs | head -n 10
   ```

