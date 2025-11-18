#!/bin/bash

# 编译 compute_gt.cpp
# 支持 L2 和 Inner Product 两种距离度量

echo "编译 compute_gt.cpp..."

# 检测编译器
if command -v g++ &> /dev/null; then
    CXX=g++
elif command -v clang++ &> /dev/null; then
    CXX=clang++
else
    echo "错误: 找不到 C++ 编译器 (g++ 或 clang++)"
    exit 1
fi

echo "使用编译器: $CXX"

# 编译选项
# -O3: 最高优化级别
# -march=native: 使用本机CPU的所有指令集（包括AVX2）
# -fopenmp: 启用OpenMP多线程支持
# -std=c++11: 使用C++11标准

$CXX -O3 -march=native -fopenmp -std=c++11 \
    src/compute_gt.cpp -o src/compute_gt

if [ $? -eq 0 ]; then
    echo "✓ 编译成功!"
    echo "可执行文件: src/compute_gt"
    echo ""
    echo "使用方法:"
    echo "  # L2 距离（欧氏距离）"
    echo "  ./src/compute_gt -b base.fvecs -q query.fvecs -o gt.ivecs -k 100 -m L2"
    echo ""
    echo "  # Inner Product（内积，适合归一化向量）"
    echo "  ./src/compute_gt -b base.fvecs -q query.fvecs -o gt.ivecs -k 100 -m IP"
    echo ""
    echo "查看帮助:"
    echo "  ./src/compute_gt -h"
else
    echo "✗ 编译失败"
    exit 1
fi

