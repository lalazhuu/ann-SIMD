#ifndef PQ_TRAINING_IP_H
#define PQ_TRAINING_IP_H

#include <vector>
#include <cmath>
#include <limits>
#include <random>
#include <numeric> // 用于 std::iota
#include <algorithm>
#include <cstdint>
#include <cstddef>
#include <iostream>
#include <omp.h> 
#include <new>
#include <arm_neon.h>

namespace InMemoryPQ_IP {

// --- 参数定义 ---
const size_t M = 4;  // 子空间数量
const size_t Ks = 256; // 每个子空间的中心点数量
const int kmeans_max_iterations = 50; // K-Means 的最大迭代次数

// --- 内积计算 (SIMD 版本) ---
inline float calculate_inner_product_simd(const float* vec1, const float* vec2, size_t sub_vecdim) {
    const int simd_width = 4;
    float inner_product = 0.0f;
    float32x4_t sum_vec = vdupq_n_f32(0.0f);
    size_t d = 0;
    for (; d + simd_width <= sub_vecdim; d += simd_width) {
        float32x4_t v1_vec = vld1q_f32(vec1 + d);
        float32x4_t v2_vec = vld1q_f32(vec2 + d);
        sum_vec = vfmaq_f32(sum_vec, v1_vec, v2_vec); // FMA: sum = sum + (v1 * v2)
    }
    inner_product = vaddvq_f32(sum_vec);
    for (; d < sub_vecdim; ++d) {
        inner_product += vec1[d] * vec2[d];
    }
    return inner_product;
}


// --- K-Means 实现 (基于内积最大化分配，均值更新) ---
inline std::vector<float> kmeans_ip_heuristic(const float* data, size_t num_points, size_t dim, size_t k, int max_iter) {
    if (num_points < k) {
        std::cerr << "错误 (kmeans_ip): 数据点数量 (" << num_points << ") 小于 k (" << k << ")" << std::endl;
        return {};
    }

    std::vector<float> centroids(k * dim);
    std::vector<size_t> assignments(num_points);
    std::vector<size_t> cluster_counts(k);
    std::vector<float> new_centroids(k * dim);

    // 初始化: 随机抽样
    std::vector<size_t> indices(num_points);
    std::iota(indices.begin(), indices.end(), 0);
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);
    for (size_t i = 0; i < k; ++i) {
        size_t point_idx = indices[i];
        if (point_idx >= num_points) { std::cerr << "错误 (kmeans_ip init): 无效索引。" << std::endl; return {}; }
        std::copy(data + point_idx * dim, data + point_idx * dim + dim, centroids.begin() + i * dim);
    }

    // K-Means 迭代
    for (int iter = 0; iter < max_iter; ++iter) {
        bool changed = false;

        // *** 分配步骤：基于最大内积 ***
        #pragma omp parallel for reduction(||:changed)
        for (size_t i = 0; i < num_points; ++i) {
            float max_inner_product = -std::numeric_limits<float>::max(); // 寻找最大内积
            size_t best_cluster = 0;
            const float* current_point = data + i * dim;

            for (size_t c = 0; c < k; ++c) {
                float ip = calculate_inner_product_simd(current_point, centroids.data() + c * dim, dim);
                if (ip > max_inner_product) { // 比较内积值
                    max_inner_product = ip;
                    best_cluster = c;
                }
            }
            if (iter > 0 && assignments[i] != best_cluster) {
                changed = true;
            }
            assignments[i] = best_cluster;
        } // 分配结束

        // *** 更新步骤：仍然使用均值 ***
        std::fill(new_centroids.begin(), new_centroids.end(), 0.0f);
        std::fill(cluster_counts.begin(), cluster_counts.end(), 0);
        for (size_t i = 0; i < num_points; ++i) {
            size_t c = assignments[i]; cluster_counts[c]++;
            const float* p = data + i * dim; float* cent = new_centroids.data() + c * dim;
            for (size_t d = 0; d < dim; ++d) cent[d] += p[d];
        }
        #pragma omp parallel for
        for (size_t c = 0; c < k; ++c) {
            if (cluster_counts[c] > 0) {
                float* cent = new_centroids.data() + c * dim; float count_inv = 1.0f / cluster_counts[c];
                for (size_t d = 0; d < dim; ++d) cent[d] *= count_inv;
            } else {
                std::copy(centroids.begin() + c * dim, centroids.begin() + c * dim + dim, new_centroids.begin() + c * dim);
                 #pragma omp critical
                 { std::cerr << "  警告: KMeans_IP 簇 " << c << " 在迭代 " << iter+1 << " 中变为空" << std::endl; }
            }
        }
        centroids = new_centroids; // 更新中心点

        // 检查收敛
        if (iter > 0 && !changed) {
             #pragma omp critical
             { std::cerr << "  KMeans_IP 在 " << iter + 1 << " 次迭代后收敛。" << std::endl; }
            break;
        }
         if (iter == max_iter - 1) {
             #pragma omp critical
             { std::cerr << "  KMeans_IP 达到最大迭代次数 (" << max_iter << ")." << std::endl; }
         }
    } // K-Means 迭代结束
    return centroids;
}

// --- 在内存中训练基于 IP 的 PQ 码本 ---
inline std::vector<float> train_pq_ip_in_memory(const float* base_data, size_t base_number, size_t vecdim)
{
    if (vecdim % M != 0) {
        std::cerr << "错误 (train_pq_ip): vecdim " << vecdim << " 不能被 M " << M << " 整除" << std::endl;
        return {};
    }
    const size_t Ds = vecdim / M;
    std::vector<float> all_codebooks_vec(M * Ks * Ds);

    std::cerr << "\n--- 开始内存中 PQ 训练 (基于 IP Heuristic, M=" << M << ", Ks=" << Ks << ", Ds=" << Ds << ") ---" << std::endl;
    bool success = true;

    #pragma omp parallel for schedule(dynamic)
    for (size_t m = 0; m < M; ++m) {
         if (!success) continue;
         #pragma omp critical
         std::cerr << "训练子空间 " << m << " (IP)... (线程 " << omp_get_thread_num() << ")" << std::endl;

        // 提取子向量
        std::vector<float> sub_vectors(base_number * Ds);
        #pragma omp parallel for
        for (size_t i = 0; i < base_number; ++i) {
            std::copy(base_data + i * vecdim + m * Ds, base_data + i * vecdim + m * Ds + Ds, sub_vectors.data() + i * Ds);
        }

        // *** 调用 IP 版本的 K-Means ***
        std::vector<float> current_centroids = kmeans_ip_heuristic(sub_vectors.data(), base_number, Ds, Ks, kmeans_max_iterations);

        if (current_centroids.empty()) {
            #pragma omp critical
            { std::cerr << "错误: KMeans_IP 失败于子空间 " << m << std::endl; success = false; }
            continue;
        }

        // 存储中心点
        std::copy(current_centroids.begin(), current_centroids.end(), all_codebooks_vec.data() + m * Ks * Ds);
         #pragma omp critical
         std::cerr << "完成训练子空间 " << m << " (IP) (线程 " << omp_get_thread_num() << ")" << std::endl;
    }

    if (!success) { return {}; }
    std::cerr << "--- 完成内存中 PQ 训练 (IP Heuristic) ---" << std::endl;
    return all_codebooks_vec;
}


// --- 使用基于 IP 的内存码本进行编码 ---
inline std::vector<uint8_t> encode_pq_ip_in_memory(
    const float* base_data,
    size_t base_number,
    size_t vecdim,
    const std::vector<float>& all_codebooks_vec // 传入用 IP K-Means 训练的码本
) {
    if (vecdim % M != 0 || all_codebooks_vec.empty() || all_codebooks_vec.size() != M*Ks*(vecdim/M)) {
         std::cerr << "错误 (encode_pq_ip): 维度或码本无效。" << std::endl;
         return {};
    }
    const size_t Ds = vecdim / M;
    const float* all_codebooks_ptr = all_codebooks_vec.data();
    std::vector<uint8_t> base_codes_vec(base_number * M);

    std::cerr << "--- 开始内存中 PQ 编码 (基于 IP) ---" << std::endl;

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < base_number; ++i) {
        uint8_t* current_code_output_ptr = base_codes_vec.data() + i * M;
        const float* current_base_vec_ptr = base_data + i * vecdim;

        for (size_t m = 0; m < M; ++m) {
            const float* sub_vector = current_base_vec_ptr + m * Ds;
            const float* current_codebook_ptr = all_codebooks_ptr + m * Ks * Ds;
            // *** 寻找内积最大的中心点 ***
            float max_inner_product = -std::numeric_limits<float>::max();
            uint8_t best_code = 0;

            for (size_t c = 0; c < Ks; ++c) {
                float ip = calculate_inner_product_simd(sub_vector, current_codebook_ptr + c * Ds, Ds);
                if (ip > max_inner_product) { // 找最大内积
                    max_inner_product = ip;
                    best_code = static_cast<uint8_t>(c);
                }
            }
            current_code_output_ptr[m] = best_code; // 存储最佳码字
        }
    } // 基向量循环结束

    std::cerr << "--- 完成内存中 PQ 编码 (基于 IP) ---" << std::endl;
    return base_codes_vec;
}

} 

#endif