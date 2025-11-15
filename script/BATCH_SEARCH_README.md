# 批量测试脚本使用说明

本目录包含用于批量测试 `search_ivf` 程序的脚本。

## 修改内容

### search_ivf.cpp 修改

`search_ivf.cpp` 已修改为支持通过命令行参数传递测试参数，而不是硬编码在代码中。

**新增参数:**
- `-z` 或 `--test_params`: 传递测试参数，格式为 `"nprobe1:refine_num1,nprobe2:refine_num2,..."`

**示例:**
```bash
./src/search_ivf -z "10:400,15:1500,20:2800,25:3500"
```

如果不提供 `-z` 参数，程序将使用默认的测试参数。

## 脚本说明

### 1. batch_search_ivf_new.sh
直接在脚本中定义测试配置的批量测试脚本。

**特点:**
- 配置直接写在脚本中
- 适合快速测试固定的几个配置
- 包含多个数据集的预定义配置

**使用方法:**
```bash
cd script
./batch_search_ivf_new.sh
```

### 2. batch_search_ivf_config.sh (推荐)
从配置文件读取测试参数的批量测试脚本。

**特点:**
- 配置从外部文件读取
- 更灵活，易于修改
- 支持注释
- 提供详细的测试统计

**使用方法:**
```bash
cd script
# 使用默认配置文件
./batch_search_ivf_config.sh

# 或指定自定义配置文件
./batch_search_ivf_config.sh my_config.txt
```

### 3. test_params_config.txt
测试参数配置文件。

**格式:**
```
数据集名称 C值 CC值 ACTUAL_C值 test_params
```

**示例:**
```
sift 1024 256 256 10:400,15:1500,20:2800,25:3500,30:5000
sift10m 2048 512 512 10:400,15:1500,20:2800,25:3500
gist 1024 256 256 10:500,15:2000,20:3500,25:4500
```

**说明:**
- 以 `#` 开头的行是注释
- 空行会被忽略
- 每行代表一个测试配置

### 4. search_ivf_single_test.sh
单数据集测试脚本示例。

**特点:**
- 适合测试单个数据集
- 可以直接修改脚本中的参数
- 代码简单，易于理解和修改

**使用方法:**
1. 编辑脚本，修改参数：
   ```bash
   dataset="sift10m"
   C=2048
   test_params="10:400,15:1500,20:2800"
   ```
2. 运行脚本：
   ```bash
   cd script
   ./search_ivf_single_test.sh
   ```

## 参数说明

### 数据集参数
- `dataset`: 数据集名称 (如 sift, sift10m, gist, tiny5m)
- `C`: 聚类数量
- `CC`: 每个向量在搜索时使用的聚类数量
- `ACTUAL_C`: 实际存储在文件中的聚类数量
- `K`: 返回的最近邻数量
- `k_overlap`: 重叠参数
- `randomize`: 是否使用随机化 (0: IVF, 1: IVF++, 2: IVF+)

### 测试参数 (test_params)
格式: `nprobe1:refine_num1,nprobe2:refine_num2,...`

- `nprobe`: 搜索时探测的聚类数量
- `refine_num`: 精化阶段使用的向量数量

**不同数据集的建议参数:**

#### SIFT (1M vectors, 128d)
- C=1024: `10:400,15:1500,20:2800,25:3500,30:5000,35:7500,40:9000,45:10000,50:12000,60:14000,80:15000`
- C=4096: `10:200,15:800,20:1400,25:1800,30:2500,35:3800,40:4500,45:5000,50:6000,60:7000,80:7500`

#### SIFT10M (10M vectors, 128d)
- C=2048: `10:400,15:1500,20:2800,25:3500,30:5000,35:7500,40:9000,45:10000,50:12000,60:14000,80:15000`
- C=4096: `10:200,15:800,20:1400,25:1800,30:2500,35:3800,40:4500,45:5000,50:6000,60:7000,80:7500`

#### GIST (1M vectors, 960d)
- C=1024: `10:500,15:2000,20:3500,25:4500,30:6000,35:9000,40:11000,45:12000,50:14000,60:17000,80:20000`

#### TINY5M (5M vectors)
- C=2048: `10:400,15:1500,20:2800,25:3500,30:5000,35:7500,40:9000,45:10000,50:12000,60:14000,80:15000`

## 工作流程

### 方式一: 使用配置文件 (推荐)

1. 编辑 `test_params_config.txt`，添加或修改测试配置
2. 运行批量测试脚本:
   ```bash
   cd script
   ./batch_search_ivf_config.sh
   ```
3. 查看结果目录中的日志文件

### 方式二: 直接运行脚本

1. 编辑 `batch_search_ivf_new.sh`，修改配置数组
2. 运行脚本:
   ```bash
   cd script
   ./batch_search_ivf_new.sh
   ```

### 方式三: 单数据集测试

1. 编辑 `search_ivf_single_test.sh`，设置参数
2. 运行脚本:
   ```bash
   cd script
   ./search_ivf_single_test.sh
   ```

## 输出结果

测试结果保存在 `results/` 目录下，文件名格式为:
```
{dataset}_IVF{C}_{randomize}.log
```

每个日志文件包含:
- 每组参数的 Recall 值
- 每次查询的平均时间
- QPS (Queries Per Second)
- 距离计算次数
- 其他性能指标

## 故障排除

### 编译失败
- 检查是否安装了 Eigen3: `sudo apt-get install libeigen3-dev`
- 检查编译器版本: `g++ --version` (需要支持 C++11)

### 文件不存在
- 检查 `path`, `index_path` 变量是否正确
- 确保索引文件已经生成
- 检查数据集文件路径是否正确

### 测试失败
- 查看具体的错误信息
- 检查参数是否合理 (例如 nprobe 不能大于 C)
- 查看日志文件中的详细错误信息

## 注意事项

1. 确保在运行脚本前已经生成了索引文件
2. 测试参数的选择应该根据数据集特性调整
3. nprobe 值不应该超过聚类数量 C
4. refine_num 值会影响精度和性能的权衡
5. 批量测试可能需要较长时间，建议使用 `nohup` 或 `screen` 在后台运行

## 示例

### 快速测试单个数据集
```bash
./src/search_ivf -d 0 -n sift -i /data/tmp/ivf/sift/sift_ivf_1024_0.index \
    -q /data/vector_datasets/sift/sift_query.fvecs \
    -g /data/vector_datasets/sift/sift_groundtruth.ivecs \
    -r ./results/sift_test.log \
    -k 1 -o 64 -f 256 \
    -b /data/vector_datasets/sift/sift_top_clusters_256_of_1024.ivecs \
    -h /data/vector_datasets/sift/sift_centroid_1024.fvecs \
    -x 256 \
    -z "10:400,20:2800,30:5000"
```

### 批量测试所有配置
```bash
cd script
./batch_search_ivf_config.sh test_params_config.txt > batch_test.log 2>&1
```

