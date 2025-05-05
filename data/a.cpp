#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <thread>
#include <mutex>
#include <algorithm>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

using MatrixXfRow = Matrix<float, Dynamic, Dynamic, RowMajor>;

MatrixXfRow read_fvecs(const string& filename) {
    ifstream input(filename, ios::binary);
    if (!input) throw runtime_error("Cannot open file");

    int dim;
    input.read(reinterpret_cast<char*>(&dim), 4);
    input.seekg(0, ios::end);
    size_t file_size = input.tellg();
    input.seekg(0, ios::beg);

    size_t vec_size = 4 + 4 * dim;
    size_t num_vecs = file_size / vec_size;

    MatrixXfRow data(num_vecs, dim);

    for (size_t i = 0; i < num_vecs; ++i) {
        int d;
        input.read(reinterpret_cast<char*>(&d), 4);
        if (d != dim) throw runtime_error("Dimension mismatch");
        input.read(reinterpret_cast<char*>(data.data() + i * dim), sizeof(float) * dim);
    }
    return data;
}

MatrixXfRow compute_query_to_centroids(const MatrixXfRow& queries, const MatrixXfRow& centroids) {
    VectorXf q_norm = queries.rowwise().squaredNorm();
    VectorXf c_norm = centroids.rowwise().squaredNorm();
    MatrixXfRow dot = queries * centroids.transpose();
    return (q_norm.replicate(1, centroids.rows()) + c_norm.transpose().replicate(queries.rows(), 1) - 2 * dot).cwiseMax(0).cwiseSqrt();
}

void estimate_lower_bounds_thread(const MatrixXfRow& q2c, const MatrixXfRow& b2c,
                                   MatrixXfRow& result, int start, int end) {
    int nb = b2c.rows();
    for (int i = start; i < end; ++i) {
        for (int j = 0; j < nb; ++j) {
            result(i, j) = (q2c.row(i) - b2c.row(j)).cwiseAbs().maxCoeff();
        }
    }
}

MatrixXfRow estimate_lower_bounds(const MatrixXfRow& q2c, const MatrixXfRow& b2c, int num_threads = 24) {
    int nq = q2c.rows(), nb = b2c.rows();
    MatrixXfRow result(nq, nb);
    vector<thread> threads;
    int block = (nq + num_threads - 1) / num_threads;

    for (int t = 0; t < num_threads; ++t) {
        int start = t * block;
        int end = min(nq, start + block);
        threads.emplace_back(estimate_lower_bounds_thread, ref(q2c), ref(b2c), ref(result), start, end);
    }
    for (auto& th : threads) th.join();
    return result;
}

void compute_l2_thread(const MatrixXfRow& queries, const MatrixXfRow& base, MatrixXfRow& result, int start, int end) {
    int nb = base.rows();
    for (int i = start; i < end; ++i) {
        for (int j = 0; j < nb; ++j) {
            result(i, j) = (queries.row(i) - base.row(j)).norm();
        }
    }
}

MatrixXfRow compute_l2(const MatrixXfRow& queries, const MatrixXfRow& base, int num_threads = 24) {
    int nq = queries.rows(), nb = base.rows();
    MatrixXfRow result(nq, nb);
    vector<thread> threads;
    int block = (nq + num_threads - 1) / num_threads;

    for (int t = 0; t < num_threads; ++t) {
        int start = t * block;
        int end = min(nq, start + block);
        threads.emplace_back(compute_l2_thread, ref(queries), ref(base), ref(result), start, end);
    }
    for (auto& th : threads) th.join();
    return result;
}

int main() {
    string root = "/data/vector_datasets/";
    string dataset = "sift";
    int K = 1024;
    string path = root + dataset + "/";

    auto base = read_fvecs(path + dataset + "_base.fvecs");
    auto query = read_fvecs(path + dataset + "_query.fvecs");
    auto centroids = read_fvecs(path + dataset + "_centroid_" + to_string(K) + ".fvecs");
    auto base2centroids = read_fvecs(path + dataset + "_distances_" + to_string(K) + ".fvecs");
    // base2centroids = base2centroids.cwiseSqrt();

    auto q2c = compute_query_to_centroids(query, centroids);
    cout << "Phase 1" << endl;
    auto estimated = estimate_lower_bounds(q2c, base2centroids);
    cout << "Phase 2" << endl;
    auto true_dist = compute_l2(query, base);
    cout << "Phase 3" << endl;

    // Calculate average ratio for each base vector
    const int rows = estimated.rows();
    const int cols = estimated.cols();
    
    vector<float> avg_ratios(cols, 0.0f);
    #pragma omp parallel for
    for (int j = 0; j < cols; ++j) {
        float sum_ratio = 0.0f;
        for (int i = 0; i < rows; ++i) {
            float ratio = estimated(i, j) / (true_dist(i, j) + 1e-6f);
            sum_ratio += ratio;
        }
        avg_ratios[j] = sum_ratio / rows;
    }

    // Write average ratios to file
    string output_file = path + dataset + "_avg_ratios_" + to_string(K) + ".txt";
    FILE* out = fopen(output_file.c_str(), "w");
    for (int j = 0; j < cols; ++j) {
        fprintf(out, "%f\n", avg_ratios[j]);
    }
    fclose(out);
}
