#include <vector>
#include <cstring>
#include <string>
#include <iostream>
#include <fstream>
#include <set>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <sys/time.h>
#include <omp.h>
#include "hnswlib/hnswlib/hnswlib.h"
#include "flat_scan.h"
// 可以自行添加需要的头文件
#include <queue>
#include <vector>
#include <utility>
#include <arm_neon.h> //NEON指令集头文件
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <limits>
#include <algorithm>
#include <new>


// 朴素SIMD
std::priority_queue<std::pair<float, uint32_t> > flat_search_simd_neon(float* base, float* query, size_t base_number, size_t vecdim, size_t k) {
    std::priority_queue<std::pair<float, uint32_t> > q;

    // 一次处理4个float (128位向量)
    const int simd_width = 4;

    for(uint32_t i = 0; i < base_number; ++i) {
        float inner_product = 0.0f;
        float* current_base = base + (size_t)i * vecdim; // 当前基向量的指针

        // 初始化NEON向量累加器为0
        float32x4_t sum_vec = vdupq_n_f32(0.0f);

        size_t d = 0;
        for (; d + simd_width <= vecdim; d += simd_width) {
            float32x4_t base_vec = vld1q_f32(current_base + d);
            float32x4_t query_vec = vld1q_f32(query + d);

            // 使用FMA乘加
            sum_vec = vfmaq_f32(sum_vec, base_vec, query_vec); // sum_vec = sum_vec + (base_vec * query_vec)
        }

        // 水平求和
        inner_product = vaddvq_f32(sum_vec); 

        // 处理剩余的维度(如果vecdim不是4的倍数)，串行处理
        for (; d < vecdim; ++d) {
            inner_product += current_base[d] * query[d];
        }

        // 最终的内积距离
        float dis = 1.0f - inner_product;

        // 维护Top K优先队列
        if(q.size() < k) {
            q.push({dis, i});
        } else {
            if(dis < q.top().first) {
                q.pop(); 
                q.push({dis, i});
            }
        }
    }
    return q;
}





// 使用标量量化 (uint8_t) 的基数据和 NEON SIMD 进行暴力搜索
std::priority_queue<std::pair<float, uint32_t> > flat_search_sq_simd_neon(
    const uint8_t* base_quantized,
    const float* query_float,  
    float data_min_val,     
    float data_scale,      
    size_t base_number, 
    size_t vecdim,        
    size_t k)  
{
    std::priority_queue<std::pair<float, uint32_t> > q;

    // --- 首先量化浮点查询向量 ---
    std::vector<uint8_t> query_quantized(vecdim);
    for (size_t d = 0; d < vecdim; ++d) {
        query_quantized[d] = quantize_float_to_uint8(query_float[d], data_min_val, data_scale);
    } // 注意这里的范围用的是base的
    const uint8_t* query_q_ptr = query_quantized.data();

    // NEON处理uint8_t时，一次处理16个元素
    const int simd_width = 16;

    // 遍历所有基向量
    for(uint32_t i = 0; i < base_number; ++i) {
        uint32_t dot_product_u32 = 0;
        const uint8_t* current_base_q = base_quantized + (size_t)i * vecdim;

        uint32x4_t sum_vec = vdupq_n_u32(0);

        // 主循环 (步长为16)
        size_t d = 0;
        for (; d + simd_width <= vecdim; d += simd_width) {
            uint8x16_t base_vec = vld1q_u8(current_base_q + d);
            uint8x16_t query_vec = vld1q_u8(query_q_ptr + d);

            // --- 点积计算 ---
            // 扩展乘法：uint8 * uint8 -> uint16 (分高低8字节进行)
            uint16x8_t prod_low = vmull_u8(vget_low_u8(base_vec), vget_low_u8(query_vec));
            uint16x8_t prod_high = vmull_u8(vget_high_u8(base_vec), vget_high_u8(query_vec));

            // 两两相加并扩展：uint16 -> uint32
            // 将prod_low的8个uint16加为4个uint32
            uint32x4_t sum32_low = vpaddlq_u16(prod_low);
            // 将prod_high的8个uint16加为4个uint32
            uint32x4_t sum32_high = vpaddlq_u16(prod_high);

            // 将高低部分的和加起来，得到这16个元素对的点积和 (存在uint32x4_t中)
            uint32x4_t current_dot_sum_vec = vaddq_u32(sum32_low, sum32_high);

            // 累加到总和向量中
            sum_vec = vaddq_u32(sum_vec, current_dot_sum_vec);
            // --- 点积计算结束 ---
        }

        // 对sum_vec中的4个uint32结果进行水平求和
        dot_product_u32 = vaddvq_u32(sum_vec);

        // 处理剩余的维度-串行处理
        for (; d < vecdim; ++d) {
            dot_product_u32 += (uint32_t)current_base_q[d] * (uint32_t)query_q_ptr[d];
        }

        // 使用uint32点积的负数作为点积距离
        float pseudo_distance = -static_cast<float>(dot_product_u32);

        // 维护Top K优先队列
        if (q.size() < k) {
            q.push({pseudo_distance, i});
        } else {
            if (pseudo_distance < q.top().first) {
                q.pop();
                q.push({pseudo_distance, i});
            }
        }
    }
    return q;
}

inline float calculate_inner_product_simd(const float* vec1, const float* vec2, size_t sub_vecdim) {
    const int simd_width = 4; // NEON float 一次处理 4 个
    float inner_product = 0.0f;

    // 初始化 NEON 向量累加器为 0
    float32x4_t sum_vec = vdupq_n_f32(0.0f);

    // --- 使用 NEON FMA 计算内积 ---
    size_t d = 0;
    for (; d + simd_width <= sub_vecdim; d += simd_width) {
        float32x4_t v1_vec = vld1q_f32(vec1 + d);
        float32x4_t v2_vec = vld1q_f32(vec2 + d);
        // FMA: sum = sum + (v1 * v2)
        sum_vec = vfmaq_f32(sum_vec, v1_vec, v2_vec);
    }
    // 水平求和
    inner_product = vaddvq_f32(sum_vec);
    // 处理尾部
    for (; d < sub_vecdim; ++d) {
        inner_product += vec1[d] * vec2[d];
    }
    return inner_product;
}


std::priority_queue<std::pair<float, uint32_t>> flat_search_pq_ip_simd_neon(
    const uint8_t* base_codes_ptr, 
    const float* query_float, 
    const float* codebooks_ptr,  
    size_t base_number,  
    size_t vecdim, 
    size_t M,
    size_t Ks, 
    size_t k  
) {
    if (M == 0 || vecdim % M != 0) {
        std::cerr << "错误 (flat_search_pq_ip): M 为零或 vecdim 不能被 M 整除。" << std::endl;
        return {};
    }
    const size_t Ds = vecdim / M; // 子空间维度

    std::priority_queue<std::pair<float, uint32_t>> q;

    // 在线计算内积查找表 (LUTs)
    std::vector<std::vector<float>> luts(M, std::vector<float>(Ks));
    #pragma omp parallel for
    for (size_t m = 0; m < M; ++m) {
        const float* query_subvector = query_float + m * Ds;
        const float* current_codebook_ptr = codebooks_ptr + m * Ks * Ds;
        for (size_t c = 0; c < Ks; ++c) {
            const float* centroid = current_codebook_ptr + c * Ds;

            luts[m][c] = calculate_inner_product_simd(query_subvector, centroid, Ds);
        }
    }



    // ADC (近似距离计算，使用内积 LUT)
    const uint8_t* current_base_code_ptr = base_codes_ptr;
    for (uint32_t i = 0; i < base_number; ++i) {
        float approx_dot_product = 0.0f;
        for (size_t m = 0; m < M; ++m) {
            uint8_t code = current_base_code_ptr[m];
            approx_dot_product += luts[m][code];
        }

        if (approx_dot_product == -std::numeric_limits<float>::max()) {
             current_base_code_ptr += M;
             continue;
        }

        float approx_ip_distance = 1.0f - approx_dot_product; 


        if (q.size() < k) {
            q.push({approx_ip_distance, i});
        } else {
            if (approx_ip_distance < q.top().first) { 
                q.pop();
                q.push({approx_ip_distance, i});
            }
        }
        current_base_code_ptr += M;
    }
    return q; 
}


// 精确重排函数：对给定的候选索引列表，使用Float NEON SIMD计算精确距离并返回Top-K
std::priority_queue<std::pair<float, uint32_t>> rerank_float_simd_neon(
    const float* base,
    const float* query,
    const std::vector<uint32_t>& candidate_indices, // 从SQ阶段获取的p个候选索引
    size_t vecdim,
    size_t k 
) {
    std::priority_queue<std::pair<float, uint32_t>> q; // 结果优先队列 (大小为 k)

    const int simd_width = 4;

    // 遍历p个候选索引
    // #pragma omp parallel for
    for (uint32_t candidate_idx : candidate_indices) {
        float inner_product = 0.0f;
        // 获取当前候选向量的指针
        const float* current_base = base + (size_t)candidate_idx * vecdim;

        float32x4_t sum_vec = vdupq_n_f32(0.0f);

        // 使用与flat_search_simd_neon完全相同的逻辑计算精确内积
        size_t d = 0;
        for (; d + simd_width <= vecdim; d += simd_width) {
            float32x4_t base_vec = vld1q_f32(current_base + d);
            float32x4_t query_vec = vld1q_f32(query + d);
            sum_vec = vfmaq_f32(sum_vec, base_vec, query_vec); // FMA
        }
        inner_product = vaddvq_f32(sum_vec); // 水平求和
        for (; d < vecdim; ++d) { // 处理尾部
            inner_product += current_base[d] * query[d];
        }

        // 计算精确的内积距离
        float dis = 1.0f - inner_product;

        if (q.size() < k) {
            q.push({dis, candidate_idx}); // 直接使用精确距离 dis
        } else {
            if (dis < q.top().first) { 
                q.pop(); 
                q.push({dis, candidate_idx}); 
            }
        }
    }

    return q;
}