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

//欧几里得距离串行算法
std::priority_queue<std::pair<float, uint32_t>> flat_search_l2_serial(
    const float* base, 
    const float* query,  
    size_t base_number, 
    size_t vecdim,    
    size_t k     
) {

    std::priority_queue<std::pair<float, uint32_t>> q;

    for (int i = 0; i < base_number; ++i) {
        float squared_l2_distance = 0.0f;
        const float* current_base = base + (size_t)i * vecdim; // 当前基向量的指针

        // 计算查询向量与当前基向量之间的平方欧几里得距离
        for (int d = 0; d < vecdim; ++d) {
            float diff = current_base[d] - query[d];
            squared_l2_distance += diff * diff; 
        }


        // 维护Top-K优先队列
        if (q.size() < k) {
            // 如果队列未满，直接将当前结果加入
            q.push({squared_l2_distance, i});
        } else {
            // 如果队列已满，比较当前距离与队列中最大距离 (q.top())
            if (squared_l2_distance < q.top().first) {
                // 如果当前距离更小，则移除队列顶部的最大距离元素
                q.pop();
                // 并将当前更小的距离元素加入队列
                q.push({squared_l2_distance, i});
            }
        }
    }
    // 返回包含Top-K最近邻结果的优先队列
    return q;
}

std::priority_queue<std::pair<float, uint32_t>> flat_search_l2_simd_neon(
    const float* base,
    const float* query,
    size_t base_number,
    size_t vecdim,
    size_t k
) {

    std::priority_queue<std::pair<float, uint32_t>> q;

    const int simd_width = 4;

    for (uint32_t i = 0; i < base_number; ++i) {
        float squared_l2_distance = 0.0f;
        const float* current_base = base + (size_t)i * vecdim; // 当前基向量指针

        float32x4_t sum_vec = vdupq_n_f32(0.0f);

        //主循环
        size_t d = 0;
        for (; d + simd_width <= vecdim; d += simd_width) {

            float32x4_t base_vec = vld1q_f32(current_base + d);
            float32x4_t query_vec = vld1q_f32(query + d);

            // 计算差值向量: diff_vec = base_vec - query_vec
            float32x4_t diff_vec = vsubq_f32(base_vec, query_vec);

            // 累加差值的平方: sum_vec = sum_vec + (diff_vec * diff_vec)
            // 使用FMA指令实现 vfmaq_f32(a, b, c) -> a + (b * c)
            sum_vec = vfmaq_f32(sum_vec, diff_vec, diff_vec);
        }

        // 水平求和
        // 将sum_vec(包含4个部分平方和) 中的值加到一个标量 squared_l2_distance 中
        squared_l2_distance = vaddvq_f32(sum_vec);

        // 处理剩余的维度 (如果vecdim不是4的倍数)，串行处理
        for (; d < vecdim; ++d) {
            float diff = current_base[d] - query[d];
            squared_l2_distance += diff * diff;
        }
 
        if (q.size() < k) {
            q.push({squared_l2_distance, i});
        } else {
            if (squared_l2_distance < q.top().first) {
                q.pop(); 
                q.push({squared_l2_distance, i}); 
            }
        }
    }
    return q;
}

// SQ量化float转uint8
inline uint8_t quantize_float_to_uint8(float val, float min_val, float scale) {
    if (scale == 0.0f) {
         return 0;
    }
    val = std::max(min_val, std::min(min_val + scale * 255.0f, val));
    float scaled_val = (val - min_val) / scale;
    return static_cast<uint8_t>(std::max(0.0f, std::min(255.0f, roundf(scaled_val))));
}


inline float calculate_sq_l2_distance_simd(const float* vec1, const float* vec2, size_t sub_vecdim) {
    const int simd_width = 4; // NEON float 一次处理 4 个
    float32x4_t sum_vec = vdupq_n_f32(0.0f); // 平方差累加器

    size_t d = 0;
    for (; d + simd_width <= sub_vecdim; d += simd_width) {
        float32x4_t v1 = vld1q_f32(vec1 + d);
        float32x4_t v2 = vld1q_f32(vec2 + d);
        float32x4_t diff = vsubq_f32(v1, v2);   // 计算差值
        // sum_vec = vfmaq_f32(sum_vec, diff, diff); // 计算平方并累加 (sum += diff*diff)
        sum_vec = vmlaq_f32(sum_vec, diff, diff); // 使用 VMLA 指令累加平方差
    }

    // 水平求和得到平方和
    float sum_sq_diff = vaddvq_f32(sum_vec);

    // 处理剩余维度
    for (; d < sub_vecdim; ++d) {
        float diff = vec1[d] - vec2[d];
        sum_sq_diff += diff * diff;
    }
    return sum_sq_diff;
}

//SQ+SIMD
std::priority_queue<std::pair<float, uint32_t>> flat_search_sq_l2_simd_neon(
    const uint8_t* base_quantized,
    const float* query_float,
    float data_min_val,
    float data_scale,
    size_t base_number,
    size_t vecdim,
    size_t k 
) {
    std::priority_queue<std::pair<float, uint32_t>> q;

    // 量化查询向量
    std::vector<uint8_t> query_quantized(vecdim);
    for (size_t d = 0; d < vecdim; ++d) {
        query_quantized[d] = quantize_float_to_uint8(query_float[d], data_min_val, data_scale);
    }
    const uint8_t* query_q_ptr = query_quantized.data();

    // NEON处理uint8_t时，一次处理16个元素
    const int simd_width = 16;

    // 遍历所有基向量
    for (uint32_t i = 0; i < base_number; ++i) {
        uint32_t total_sq_l2_u32 = 0; // 最终的近似平方L2距离 (uint32_t)
        const uint8_t* current_base_q = base_quantized + (size_t)i * vecdim;

        uint32x4_t sum_vec = vdupq_n_u32(0);

        // 主循环(步长为16)
        size_t d = 0;
        for (; d + simd_width <= vecdim; d += simd_width) {
            uint8x16_t base_vec_u8 = vld1q_u8(current_base_q + d);
            uint8x16_t query_vec_u8 = vld1q_u8(query_q_ptr + d);

            // 扩展uint8->uint16 (分高低8字节)
            uint16x8_t base_low_u16 = vmovl_u8(vget_low_u8(base_vec_u8));
            uint16x8_t base_high_u16 = vmovl_u8(vget_high_u8(base_vec_u8));
            uint16x8_t query_low_u16 = vmovl_u8(vget_low_u8(query_vec_u8));
            uint16x8_t query_high_u16 = vmovl_u8(vget_high_u8(query_vec_u8));

            // 转换为有符号类型并计算差值 (int16)
            // 使用vreinterpretq将uint16视为int16进行减法
            int16x8_t diff_low_s16 = vsubq_s16(vreinterpretq_s16_u16(base_low_u16), vreinterpretq_s16_u16(query_low_u16));
            int16x8_t diff_high_s16 = vsubq_s16(vreinterpretq_s16_u16(base_high_u16), vreinterpretq_s16_u16(query_high_u16));

            // 计算平方差 (int16 * int16 -> int32，需要 widening multiply)
            // 计算低8个元素的平方差
            int32x4_t sq_diff_low1 = vmull_s16(vget_low_s16(diff_low_s16), vget_low_s16(diff_low_s16));
            int32x4_t sq_diff_low2 = vmull_s16(vget_high_s16(diff_low_s16), vget_high_s16(diff_low_s16));
            // 计算高8个元素的平方差
            int32x4_t sq_diff_high1 = vmull_s16(vget_low_s16(diff_high_s16), vget_low_s16(diff_high_s16));
            int32x4_t sq_diff_high2 = vmull_s16(vget_high_s16(diff_high_s16), vget_high_s16(diff_high_s16));

            // 累加平方差到uint32x4_t累加器
            // 因为平方总是非负，可以直接将int32结果视为uint32进行累加
            sum_vec = vaddq_u32(sum_vec, vreinterpretq_u32_s32(sq_diff_low1));
            sum_vec = vaddq_u32(sum_vec, vreinterpretq_u32_s32(sq_diff_low2));
            sum_vec = vaddq_u32(sum_vec, vreinterpretq_u32_s32(sq_diff_high1));
            sum_vec = vaddq_u32(sum_vec, vreinterpretq_u32_s32(sq_diff_high2));
        }

        // 水平求和
        total_sq_l2_u32 = vaddvq_u32(sum_vec);

        // 处理剩余的维度-串行处理
        for (; d < vecdim; ++d) {
            // 将uint8_t转换为int进行减法，避免溢出
            int diff = (int)current_base_q[d] - (int)query_q_ptr[d];
            total_sq_l2_u32 += (uint32_t)(diff * diff); // 累加平方差
        }
        // total_sq_l2_u32 现在存储了近似平方 L2 距离


        float distance_for_queue = static_cast<float>(total_sq_l2_u32);

        if (q.size() < k) {
            q.push({distance_for_queue, i});
        } else {
            if (distance_for_queue < q.top().first) {
                q.pop();
                q.push({distance_for_queue, i});
            }
        }
    }
    return q;
}

std::priority_queue<std::pair<float, uint32_t>> rerank_l2_simd_neon(
    const float* base,
    const float* query,
    const std::vector<uint32_t>& candidate_indices, // p个候选索引
    size_t vecdim,
    size_t k 
) {

    std::priority_queue<std::pair<float, uint32_t>> q;
    const int simd_width = 4;

    for (size_t i = 0; i < candidate_indices.size(); ++i) {
        uint32_t candidate_idx = candidate_indices[i];

        float exact_sq_l2_distance = 0.0f;

        const float* current_base = base + (size_t)candidate_idx * vecdim;

        float32x4_t sum_vec = vdupq_n_f32(0.0f);

        // NEON SIMD 计算精确平方L2距离
        size_t d = 0;
        for (; d + simd_width <= vecdim; d += simd_width) {
            float32x4_t base_vec = vld1q_f32(current_base + d);
            float32x4_t query_vec = vld1q_f32(query + d);
            // 计算差值
            float32x4_t diff_vec = vsubq_f32(base_vec, query_vec);
            // 累加平方差
            sum_vec = vfmaq_f32(sum_vec, diff_vec, diff_vec); // sum = sum + diff*diff
        }
        // 水平求和
        exact_sq_l2_distance = vaddvq_f32(sum_vec);
        // 处理尾部
        for (; d < vecdim; ++d) {
            float diff = current_base[d] - query[d];
            exact_sq_l2_distance += diff * diff;
        }


        if (q.size() < k) {
            q.push({exact_sq_l2_distance, candidate_idx});
        } else {
            if (exact_sq_l2_distance < q.top().first) {
                q.pop();
                q.push({exact_sq_l2_distance, candidate_idx});
            }
        }

    } 

    return q;
}


std::priority_queue<std::pair<float, uint32_t>> flat_search_pq_simd_neon(
    const uint8_t* base_codes_ptr,
    const float* query_float,
    const float* codebooks_ptr, // 指向内存中码本数据的指针
    size_t base_number,
    size_t vecdim,// 原始向量维度
    size_t M,// 子空间数量 
    size_t Ks,// 每个子空间的中心点数量
    size_t k 
) {

    if (M == 0 || vecdim % M != 0) {
        std::cerr << "错误 (flat_search_pq): M 为零或 vecdim 不能被 M 整除。" << std::endl;
        return {};
    }
    const size_t Ds = vecdim / M; // 子空间维度

    std::priority_queue<std::pair<float, uint32_t>> q; // 结果优先队列

    // 在线计算距离查找表 (LUT)
    std::vector<std::vector<float>> luts(M, std::vector<float>(Ks));
    #pragma omp parallel for // 可以并行计算不同子空间的LUT
    for (size_t m = 0; m < M; ++m) {
        const float* query_subvector = query_float + m * Ds;
        const float* current_codebook_ptr = codebooks_ptr + m * Ks * Ds;
        for (size_t c = 0; c < Ks; ++c) {
            const float* centroid = current_codebook_ptr + c * Ds;
            // 计算查询子向量和中心点的平方L2距离 (使用 SIMD)
            luts[m][c] = calculate_sq_l2_distance_simd(query_subvector, centroid, Ds);
        }
    }

    // ADC (近似距离计算，使用LUT和base_codes_ptr)
    const uint8_t* current_base_code_ptr = base_codes_ptr;
    for (uint32_t i = 0; i < base_number; ++i) {
        float approx_distance = 0.0f;
        // 遍历 M 个子空间的码字
        for (size_t m = 0; m < M; ++m) {
            uint8_t code = current_base_code_ptr[m]; // 获取第 m 个子空间的码字 (0-255)
            approx_distance += luts[m][code];// 从对应的LUT查表并累加
        }

         if (approx_distance == std::numeric_limits<float>::max()) {
             current_base_code_ptr += M; // 移动指针到下一个基向量
             continue; // 跳过这个有问题的向量
         }


        if (q.size() < k) {
            q.push({approx_distance, i});
        } else {
            if (approx_distance < q.top().first) {
                q.pop();
                q.push({approx_distance, i});
            }
        }
        current_base_code_ptr += M; 
    } 
    return q; 
}



