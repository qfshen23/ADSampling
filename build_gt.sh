#!/bin/bash

# High-performance groundtruth computation build script

echo "=== Building High-Performance Groundtruth Computer ==="

# Check if compiler supports required features
echo "Checking compiler and CPU features..."

# Detect available SIMD instructions
SIMD_FLAGS=""
if grep -q avx2 /proc/cpuinfo; then
    echo "✓ AVX2 support detected"
    SIMD_FLAGS="-mavx2 -mfma"
elif grep -q avx /proc/cpuinfo; then
    echo "✓ AVX support detected"
    SIMD_FLAGS="-mavx"
elif grep -q sse4_2 /proc/cpuinfo; then
    echo "✓ SSE4.2 support detected"
    SIMD_FLAGS="-msse4.2"
else
    echo "⚠ No advanced SIMD support detected, using basic optimizations"
    SIMD_FLAGS=""
fi

# Compiler flags
CXX="g++"
CXXFLAGS="-std=c++17 -O3 -march=native -mtune=native -fopenmp"
CXXFLAGS="$CXXFLAGS -Wall -Wextra -DNDEBUG"
CXXFLAGS="$CXXFLAGS $SIMD_FLAGS"

echo "Compiler: $CXX"
echo "Flags: $CXXFLAGS"

# Build
echo "Building..."
$CXX $CXXFLAGS -o compute_gt src/compute_gt.cpp

if [ $? -eq 0 ]; then
    echo "✓ Build successful!"
    echo ""
    echo "Usage examples:"
    echo "  ./compute_gt -b base.fvecs -q query.fvecs -o groundtruth.ivecs -k 100"
    echo "  ./compute_gt -b /data/vector_datasets/sift/sift_base.fvecs \\"
    echo "               -q /data/vector_datasets/sift/sift_query.fvecs \\"
    echo "               -o sift_groundtruth.ivecs -k 100 -t 8"
    echo ""
    echo "For help: ./compute_gt --help"
else
    echo "✗ Build failed!"
    exit 1
fi
