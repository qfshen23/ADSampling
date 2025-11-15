# 快速开始指南

## 快速开始

### 1. 编译程序
```bash
cd /home/qfshen/workspace/vdb/adsampling
g++ ./src/search_ivf.cpp -O3 -mavx -mavx512vpopcntdq -g -o ./src/search_ivf -I ./src/ -I /usr/include/eigen3 -fopenmp
```

### 2. 单次测试
```bash
./src/search_ivf \
    -d 0 \
    -n sift10m \
    -i /data/tmp/ivf/sift10m/sift10m_ivf_2048_0.index \
    -q /data/vector_datasets/sift10m/sift10m_query.fvecs \
    -g /data/vector_datasets/sift10m/sift10m_groundtruth.ivecs \
    -r ./results/sift10m_test.log \
    -k 1 \
    -o 64 \
    -f 512 \
    -b /data/vector_datasets/sift10m/sift10m_top_clusters_512_of_2048.ivecs \
    -h /data/vector_datasets/sift10m/sift10m_centroid_2048.fvecs \
    -x 512 \
    -z "10:400,15:1500,20:2800,25:3500,30:5000"
```

### 3. 批量测试（推荐）

#### 方法一：使用配置文件
```bash
cd script
chmod +x batch_search_ivf_config.sh
./batch_search_ivf_config.sh test_params_config.txt
```

#### 方法二：使用内置配置
```bash
cd script
chmod +x batch_search_ivf_new.sh
./batch_search_ivf_new.sh
```

#### 方法三：单数据集测试
```bash
cd script
# 编辑 search_ivf_single_test.sh，修改数据集和参数
chmod +x search_ivf_single_test.sh
./search_ivf_single_test.sh
```

## 参数说明

### 新增参数
- `-z` 或 `--test_params`: 测试参数，格式为 `"nprobe1:refine_num1,nprobe2:refine_num2,..."`

### 示例
```bash
# 测试 3 组参数
-z "10:400,20:2800,30:5000"

# 测试 5 组参数
-z "10:400,15:1500,20:2800,25:3500,30:5000"

# 如果不指定 -z 参数，使用默认的 11 组参数
```

## 配置文件格式

`test_params_config.txt` 格式:
```
# 数据集名称 C值 CC值 ACTUAL_C值 test_params
sift 1024 256 256 10:400,15:1500,20:2800,25:3500,30:5000
sift10m 2048 512 512 10:400,15:1500,20:2800,25:3500
```

## 常见用例

### 用例1: 快速测试几组参数
```bash
./src/search_ivf -d 0 -n sift -i /path/to/index ... -z "10:400,20:2800,30:5000"
```

### 用例2: 不同数据集批量测试
编辑 `test_params_config.txt`:
```
sift 1024 256 256 10:400,20:2800,30:5000
sift10m 2048 512 512 10:400,20:2800,30:5000
gist 1024 256 256 10:500,20:3500,30:6000
```

然后运行:
```bash
cd script
./batch_search_ivf_config.sh
```

### 用例3: 同一数据集不同 C 值
```
sift 1024 256 256 10:400,20:2800,30:5000
sift 2048 512 512 10:400,20:2800,30:5000
sift 4096 1024 1024 10:200,20:1400,30:2500
```

## 查看结果

结果保存在 `results/` 目录:
```bash
# 查看最新的结果
tail -n 50 results/sift10m_IVF2048_0.log

# 查看所有测试的 Recall 和 QPS
grep -E "Recall|QPS" results/sift10m_IVF2048_0.log
```

## 注意事项

1. **test_params 格式**: 必须是 `nprobe:refine_num` 的格式，用逗号分隔
2. **文件路径**: 确保所有输入文件存在
3. **nprobe 范围**: nprobe 值应该在 1 到 C 之间
4. **批量测试时间**: 批量测试可能需要较长时间，建议在后台运行
5. **默认参数**: 如果不提供 `-z` 参数，程序会使用默认的 11 组测试参数

## 后台运行

长时间批量测试建议使用 `nohup`:
```bash
cd script
nohup ./batch_search_ivf_config.sh > batch_test.log 2>&1 &
```

查看进度:
```bash
tail -f batch_test.log
```

