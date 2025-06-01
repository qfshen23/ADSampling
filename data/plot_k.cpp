#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/index_io.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <filesystem>
#include <numeric>
#include <chrono>

namespace fs = std::filesystem;

// Function to read .fvecs file format
std::vector<float> read_fvecs(const std::string& filename, size_t& dim, size_t& num) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) throw std::runtime_error("Cannot open file " + filename);

    std::vector<float> data;
    dim = 0; num = 0;

    while (in) {
        int32_t d;
        in.read(reinterpret_cast<char*>(&d), 4);
        if (!in) break;

        if (dim == 0) dim = d;
        else if (dim != static_cast<size_t>(d)) throw std::runtime_error("Inconsistent dimensions");

        std::vector<float> vec(d);
        in.read(reinterpret_cast<char*>(vec.data()), d * sizeof(float));
        if (!in) break;

        data.insert(data.end(), vec.begin(), vec.end());
        ++num;
    }
    return data;
}

// Compute kNN indegree using GPU
std::vector<int> knn_indegree_faiss_gpu(const float* data, size_t N, size_t D, int K, size_t batch_size = 10000) {
    faiss::gpu::StandardGpuResources res;
    faiss::gpu::GpuIndexFlatL2 index(&res, D);
    index.add(N, data);

    std::vector<int> indegree(N, 0);
    std::vector<int64_t> neighbors((K + 1) * batch_size);
    std::vector<float> distances((K + 1) * batch_size);

    for (size_t start = 0; start < N; start += batch_size) {
        size_t end = std::min(start + batch_size, N);
        size_t bs = end - start;

        index.search(bs, data + start * D, K + 1, distances.data(), neighbors.data());

        for (size_t i = 0; i < bs; ++i) {
            size_t offset = i * (K + 1);
            for (int j = 0, cnt = 0; j < K + 1 && cnt < K; ++j) {
                int64_t id = neighbors[offset + j];
                if (id != static_cast<int64_t>(start + i)) {
                    indegree[id]++;
                    cnt++;
                }
            }
        }
    }
    return indegree;
}

// Save to .npy-like format (binary integers)
void save_indegree(const std::string& filename, const std::vector<int>& indegree) {
    std::ofstream out(filename, std::ios::binary);
    if (!out) throw std::runtime_error("Failed to open " + filename);
    out.write(reinterpret_cast<const char*>(indegree.data()), indegree.size() * sizeof(int));
}

int main() {
    std::vector<std::string> datasets = {"tiny5m", "sift10m"};
    std::string source = "/data/vector_datasets/";
    int K = 100;

    for (const auto& dataset : datasets) {
        std::cout << "Processing dataset - " << dataset << std::endl;

        std::string base_path = source + dataset + "/" + dataset + "_base.fvecs";
        size_t dim = 0, num = 0;
        std::vector<float> base = read_fvecs(base_path, dim, num);

        std::cout << "Loaded data with " << num << " vectors of dimension " << dim << std::endl;

        auto indegree = knn_indegree_faiss_gpu(base.data(), num, dim, K, 100000);
        save_indegree(dataset + "_faiss_knn_indegree.bin", indegree);
    }

    return 0;
}