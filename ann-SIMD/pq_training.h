#ifndef PQ_TRAINING_H
#define PQ_TRAINING_H

#include <vector>
#include <cmath>
#include <limits>
#include <random>
#include <numeric> 
#include <algorithm>
#include <cstdint>
#include <cstddef>
#include <iostream>
#include <omp.h>
#include <new> 

namespace InMemoryPQ {// InMemoryPQ::函数名调用


const size_t M = 4;  // 子空间数量
const size_t Ks = 256; // 每个子空间的中心点数量
const int kmeans_max_iterations = 50; // K-Means 的最大迭代次数

// 平方L2距离 (K-Means的辅助函数)
inline float calculate_sq_l2_distance(const float* vec1, const float* vec2, size_t dim) {
    float dist = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        float diff = vec1[i] - vec2[i];
        dist += diff * diff;
    }
    return dist;
}

// 基础K-Means实现
// 返回中心点向量 (k * dim)，如果失败则返回空向量。
inline std::vector<float> kmeans(const float* data, size_t num_points, size_t dim, size_t k, int max_iter) {
    // 参数：data - 指向输入数据的指针 (num_points * dim 的一维数组)；
    //       num_points - 数据点的数量； dim - 每个数据点的维度；
    //       k - 需要聚成的簇的数量 (等于 Ks)； max_iter - 最大迭代次数。
    // 返回：一个 std::vector<float>，包含了学习到的 k 个中心点的坐标 (k * dim 的一维数组)，
    if (num_points < k) {
        std::cerr << "错误 (kmeans): 数据点数量 (" << num_points << ") 小于 k (" << k << ")" << std::endl;
        return {};
    }

    std::vector<float> centroids(k * dim);// 存储k个中心点，每个中心点dim维
    std::vector<size_t> assignments(num_points); // 存储每个点的归属簇索引(0到k-1)
    std::vector<size_t> cluster_counts(k);// 存储每个簇当前包含多少个数据点
    std::vector<float> new_centroids(k * dim);// 存储迭代中计算出的新中心点

    // 初始化: 随机抽样
    std::vector<size_t> indices(num_points);// 创建一个大小为num_points的索引 vector
    std::iota(indices.begin(), indices.end(), 0); // 使用 std::iota 填充 vector，使其包含 0, 1, 2, ..., num_points-1
    std::random_device rd;// 创建一个随机设备，用于生成随机数种子
    std::mt19937 g(rd());// 使用随机设备初始化一个梅森旋转引擎，高质量的随机数生成器
    std::shuffle(indices.begin(), indices.end(), g); // 使用随机数生成器g打乱indices vector中元素的顺序

    // 从打乱后的索引中选取前k个，对应的数据点作为初始中心点
    for (size_t i = 0; i < k; ++i) {
        size_t point_idx = indices[i]; // 获取选中的随机索引
         if (point_idx >= num_points) { // 安全性检查，防止索引越界
             std::cerr << "错误 (kmeans init): 无效的随机索引。" << std::endl; return {};
         }
        // 使用 std::copy 将选中的数据点 (data + point_idx * dim) 复制到centroids向量的对应位置。
        std::copy(data + point_idx * dim, data + point_idx * dim + dim, centroids.begin() + i * dim);
    }

    // K-Means迭代主循环
    for (int iter = 0; iter < max_iter; ++iter) { // 循环执行，直到达到最大迭代次数或收敛
        bool changed = false; // 标记在本次迭代中是否有数据点的分配发生了改变


        //分配
        #pragma omp parallel for reduction(||:changed)
        for (size_t i = 0; i < num_points; ++i) { // 遍历每一个数据点
            float min_dist = std::numeric_limits<float>::max(); // 初始化当前点到所有中心点的最小距离为 float 的最大值。
            size_t best_cluster = 0; // 初始化分配给当前点的最佳簇索引为 0
            const float* current_point = data + i * dim; // 获取当前数据点的指针

            // 遍历所有 k 个中心点
            for (size_t c = 0; c < k; ++c) {
                // 计算当前点与中心点c之间的平方L2距离
                float dist = calculate_sq_l2_distance(current_point, centroids.data() + c * dim, dim);
                // 如果找到更小的距离
                if (dist < min_dist) {
                    min_dist = dist; // 更新最小距离
                    best_cluster = c; // 更新最佳簇索引
                }
            }
            // 检查分配是否改变 (从第二次迭代开始检查，iter > 0)
            if (iter > 0 && assignments[i] != best_cluster) {
                changed = true; // 标记发生了改变
            }
            assignments[i] = best_cluster; // 将当前点分配给最佳簇
        } 

        //更新步骤
        // 将new_centroids（用于累加）和cluster_counts（用于计数）清零。
        std::fill(new_centroids.begin(), new_centroids.end(), 0.0f);
        std::fill(cluster_counts.begin(), cluster_counts.end(), 0);

        // 聚合簇内所有点。遍历所有数据点。
        for (size_t i = 0; i < num_points; ++i) {
            size_t cluster_idx = assignments[i]; // 获取当前点所属的簇索引。
            cluster_counts[cluster_idx]++; // 将对应簇的计数加 1。
            const float* current_point = data + i * dim; // 当前点的指针。
            float* centroid_to_update = new_centroids.data() + cluster_idx * dim; // 指向需要更新的那个中心点（累加器）。
            // 将当前点的坐标累加到对应簇的累加器中。
            for (size_t d = 0; d < dim; ++d) {
                centroid_to_update[d] += current_point[d];
            }
        }

        // 计算新的中心点（取均值）
        #pragma omp parallel for // 并行处理k个中心点的计算
        for (size_t c = 0; c < k; ++c) { // 遍历每一个簇/中心点
            if (cluster_counts[c] > 0) { // 如果这个簇不是空的
                float* centroid_to_update = new_centroids.data() + c * dim; // 获取指向新中心点的指针
                float count_inv = 1.0f / cluster_counts[c]; // 计算计数的倒数，避免在循环内重复做除法
                // 将累加的坐标除以簇内点的数量，得到均值
                for (size_t d = 0; d < dim; ++d) {
                    centroid_to_update[d] *= count_inv;
                }
            } else { // 如果这个簇是空的
                // 处理空簇：简单的策略是保持上一次迭代的中心点不变。
                // 复制旧中心点 (centroids) 的数据到新中心点 (new_centroids) 的对应位置。
                std::copy(centroids.begin() + c * dim, centroids.begin() + c * dim + dim, new_centroids.begin() + c * dim);
                #pragma omp critical
                {
                    std::cerr << "  警告: KMeans 簇 " << c << " 在迭代 " << iter + 1 << " 中变为空" << std::endl;
                }
            }
        }

        centroids = new_centroids; // 将计算得到的新中心点更新为当前中心点，用于下一次迭代。

        //检查收敛
        // 如果在一次迭代中 (iter > 0)，没有任何点的分配发生改变 (!changed)，则认为算法收敛。
        if (iter > 0 && !changed) {
             #pragma omp critical // 保证只有一个线程打印收敛信息
             {
                std::cerr << "  KMeans 在 " << iter + 1 << " 次迭代后收敛。" << std::endl;
             }
            break; // 提前退出迭代循环
        }
         // 如果达到了最大迭代次数
         if (iter == max_iter - 1) {
             #pragma omp critical // 保证只有一个线程打印达到最大次数的信息
             {
                 std::cerr << "  KMeans 达到最大迭代次数 (" << max_iter << ")." << std::endl;
             }
         }
    }

    return centroids; // 返回最终计算得到的中心点
}


// 在内存中训练PQ码本
// 返回一个包含所有连接起来的码本的 vector (M * Ks * Ds 个 float)
inline std::vector<float> train_pq_in_memory(const float* base_data, size_t base_number, size_t vecdim)
{
    // 函数功能：对整个基数据集 base_data 进行 PQ 训练，学习所有 M 个子空间的码本。
    // 参数：base_data - 指向原始基数据的指针； base_number - 基向量数量； vecdim - 原始向量维度。
    // 返回：一个 std::vector<float>，包含了所有 M*Ks 个中心点的数据 (M*Ks*Ds)，如果失败则为空。

    // 检查原始维度是否能被子空间数量 M 整除
    if (vecdim % M != 0) {
        std::cerr << "错误 (train_pq_in_memory): 向量维度 " << vecdim << " 不能被 M " << M << " 整除" << std::endl;
        return {}; // 返回空 vector
    }
    const size_t Ds = vecdim / M; // 计算每个子向量的维度。

    // 预先分配足够存储所有码本中心点的内存。
    // M 个子空间，每个空间 Ks 个中心点，每个中心点 Ds 维。
    std::vector<float> all_codebooks_vec(M * Ks * Ds);

    // 打印开始训练的信息
    std::cerr << "\n--- 开始内存中 PQ 训练 (M=" << M << ", Ks=" << Ks << ", Ds=" << Ds << ") ---" << std::endl;

    bool success = true; // 用于在并行环境中标记是否有任何子空间训练失败。

    // --- 并行训练每个子空间的码本 ---
    // 使用 OpenMP 并行化外层循环，每个线程负责训练一个或多个子空间。
    // schedule(dynamic): 让 OpenMP 运行时动态地给空闲线程分配下一个子空间 (m) 的任务。
    // 这在 K-Means 执行时间不均匀时可能比静态分配更好。
    #pragma omp parallel for schedule(dynamic)
    for (size_t m = 0; m < M; ++m) { // 遍历 M 个子空间
         if (!success) continue; // 如果已经有其他线程报告失败，当前线程就不再开始新的任务。

        #pragma omp critical
        std::cerr << "训练子空间 " << m << "/" << M - 1 << " (线程 " << omp_get_thread_num() << ")" << std::endl;

        // --- 提取当前子空间 m 的所有子向量 ---
        // !! 这是一个内存开销很大的操作，需要 base_number * Ds * sizeof(float) 的额外内存。
        std::vector<float> sub_vectors(base_number * Ds);
        for (size_t i = 0; i < base_number; ++i) { // 遍历所有基向量
            const float* src = base_data + i * vecdim + m * Ds; // 计算原始数据中子向量的起始地址。
            float* dest = sub_vectors.data() + i * Ds; // 计算在临时 vector 中的存储地址。
            std::copy(src, src + Ds, dest);
        } 

        // --- 对当前子空间的子向量运行 K-Means ---
        std::vector<float> current_centroids = kmeans(sub_vectors.data(), base_number, Ds, Ks, kmeans_max_iterations);

        // --- 检查 K-Means 结果 ---
        // 如果返回的 vector 为空，或者大小不等于预期的 Ks * Ds，说明 K-Means 失败。
        if (current_centroids.empty() || current_centroids.size() != Ks * Ds) {
             #pragma omp critical
             {
                std::cerr << "错误: 子空间 " << m << " 的 K-Means 失败" << std::endl;
                success = false; // 设置全局失败标志。
             }
             continue;
        }

        // --- 存储学习到的中心点 ---
        // 计算在最终 all_codebooks_vec 中存储当前子空间中心点的起始地址。
        float* dest_codebook_ptr = all_codebooks_vec.data() + m * Ks * Ds;
        // 将当前计算得到的中心点数据复制到最终的 vector 中。
        std::copy(current_centroids.begin(), current_centroids.end(), dest_codebook_ptr);

        #pragma omp critical
        std::cerr << "完成训练子空间 " << m << " (线程 " << omp_get_thread_num() << ")" << std::endl;
    } 

    // 检查是否有任何线程报告了失败
    if (!success) {
         std::cerr << "PQ 训练过程中发生错误。" << std::endl;
         return {}; // 返回空 vector
    }

    std::cerr << "--- 完成内存中 PQ 训练 ---" << std::endl;
    // 返回包含所有码本数据的 vector。
    return all_codebooks_vec;
}

// --- 使用内存中的码本对基数据进行编码 ---
// 返回一个包含所有连接起来的编码的 vector (base_number * M 个 uint8_t)
inline std::vector<uint8_t> encode_pq_in_memory(
    const float* base_data,
    size_t base_number,
    size_t vecdim,
    const std::vector<float>& all_codebooks_vec // 传入包含所有训练好的码本的 vector
) {
    // 函数功能：使用训练好的码本，将原始基数据量化为 PQ 编码。
    // 返回：一个 std::vector<uint8_t>，包含了所有基向量的 PQ 编码 (base_number * M)，失败时为空。

    if (vecdim % M != 0) { // 检查维度是否匹配
        std::cerr << "错误 (encode_pq_in_memory): 向量维度 " << vecdim << " 不能被 M " << M << " 整除" << std::endl;
        return {};
    }
    if (all_codebooks_vec.empty() || all_codebooks_vec.size() != M * Ks * (vecdim / M)) {
         std::cerr << "错误 (encode_pq_in_memory): 提供了无效或空的码本 vector。" << std::endl;
         return {};
    }

    const size_t Ds = vecdim / M; // 计算子空间维度。
    const float* all_codebooks_ptr = all_codebooks_vec.data(); // 获取指向码本数据的底层指针，提高访问效率。
    std::vector<uint8_t> base_codes_vec(base_number * M); // 预分配存储所有 PQ 编码的 vector。

    std::cerr << "--- 开始内存中 PQ 编码 ---" << std::endl;

    // --- 并行编码所有基向量 ---
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < base_number; ++i) {
        // 计算当前基向量编码在 base_codes_vec 中的起始存储地址。
        uint8_t* current_code_output_ptr = base_codes_vec.data() + i * M;
        // 获取当前原始基向量的起始地址。
        const float* current_base_vec_ptr = base_data + i * vecdim;

        // --- 对当前基向量的 M 个子空间进行编码 ---
        for (size_t m = 0; m < M; ++m) { // 遍历 M 个子空间
            const float* sub_vector = current_base_vec_ptr + m * Ds; // 获取当前子向量的指针。
            const float* current_codebook_ptr = all_codebooks_ptr + m * Ks * Ds; // 获取当前子空间对应码本的指针。
            float min_dist = std::numeric_limits<float>::max(); // 初始化最小距离为最大值。
            uint8_t best_code = 0; // 初始化最佳码字索引为 0。

            // --- 寻找距离当前子向量最近的中心点 ---
            for (size_t c = 0; c < Ks; ++c) { // 遍历当前子码本中的 Ks 个中心点
                // 计算子向量与中心点之间的平方 L2 距离
                float dist = calculate_sq_l2_distance(sub_vector, current_codebook_ptr + c * Ds, Ds);
                // 如果找到更近的中心点
                if (dist < min_dist) {
                    min_dist = dist; // 更新最小距离
                    best_code = static_cast<uint8_t>(c); // 将中心点的索引 c 存储为最佳码字 (转换为 uint8_t)
                }
            } 
            current_code_output_ptr[m] = best_code; // 将找到的最佳码字存入结果 vector 的对应位置
        } 
    } 

    std::cerr << "--- 完成内存中 PQ 编码 ---" << std::endl;
    // 返回包含所有 PQ 编码的 vector。
    return base_codes_vec;
}

} 

#endif