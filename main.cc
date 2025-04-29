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
#include "pq_training.h"
#include "pq_training_ip.h"
#include "l2.h"
#include "ip.h"
using namespace hnswlib;

template<typename T>
T *LoadData(std::string data_path, size_t& n, size_t& d)
{
    std::ifstream fin;
    fin.open(data_path, std::ios::in | std::ios::binary);
    fin.read((char*)&n,4);
    fin.read((char*)&d,4);
    T* data = new T[n*d];
    int sz = sizeof(T);
    for(int i = 0; i < n; ++i){
        fin.read(((char*)data + i*d*sz), d*sz);
    }
    fin.close();

    std::cerr<<"load data "<<data_path<<"\n";
    std::cerr<<"dimension: "<<d<<"  number:"<<n<<"  size_per_element:"<<sizeof(T)<<"\n";

    return data;
}

struct SearchResult
{
    float recall;
    int64_t latency; // 单位us
};

void build_index(float* base, size_t base_number, size_t vecdim)
{
    const int efConstruction = 150; // 为防止索引构建时间过长，efc建议设置200以下
    const int M = 16; // M建议设置为16以下

    HierarchicalNSW<float> *appr_alg;
    InnerProductSpace ipspace(vecdim);
    appr_alg = new HierarchicalNSW<float>(&ipspace, base_number, M, efConstruction);

    appr_alg->addPoint(base, 0);
    #pragma omp parallel for
    for(int i = 1; i < base_number; ++i) {
        appr_alg->addPoint(base + 1ll*vecdim*i, i);
    }

    char path_index[1024] = "files/hnsw.index";
    appr_alg->saveIndex(path_index);
}




int main(int argc, char *argv[])
{
    size_t test_number = 0, base_number = 0;
    size_t test_gt_d = 0, vecdim = 0;

    std::string data_path = "/anndata/"; 
    auto test_query = LoadData<float>(data_path + "DEEP100K.query.fbin", test_number, vecdim);
    auto test_gt = LoadData<int>(data_path + "DEEP100K.gt.query.100k.top100.bin", test_number, test_gt_d);
    auto base = LoadData<float>(data_path + "DEEP100K.base.100k.fbin", base_number, vecdim);
    // 只测试前2000条查询
    test_number = 2000;

    const size_t p = 600; // 重排 (Rerank) 候选集大小 P (p >= k)
    const size_t k = 10;

    // --- 执行内存中的 PQ 训练和编码(欧几里得距离) ---
    // 从头文件中获取 PQ 参数
    const size_t M = InMemoryPQ::M;
    const size_t Ks = InMemoryPQ::Ks;

     // 再次检查维度是否有效
    if (vecdim == 0 || vecdim % M != 0) {
         std::cerr << "错误: vecdim=" << vecdim << " 为 0 或不能被 M=" << M << " 整除" << std::endl;
         delete[] test_query; delete[] test_gt; delete[] base;
         return 1;
    }

    // 在内存中训练码本
    std::vector<float> all_codebooks_vec = InMemoryPQ::train_pq_in_memory(base, base_number, vecdim);
    // 检查训练是否成功
    if (all_codebooks_vec.empty()) {
         std::cerr << "错误: 内存中 PQ 训练失败。程序退出。" << std::endl;
         delete[] test_query; delete[] test_gt; delete[] base;
         return 1;
    }

    // 使用训练好的码本在内存中编码基数据
    std::vector<uint8_t> base_codes_vec = InMemoryPQ::encode_pq_in_memory(base, base_number, vecdim, all_codebooks_vec);
     // 检查编码是否成功
     if (base_codes_vec.empty()) {
         std::cerr << "错误: 内存中 PQ 编码失败。程序退出。" << std::endl;
         delete[] test_query; delete[] test_gt; delete[] base;
         // 注意: all_codebooks_vec 会自动清理
         return 1;
    }
    // //  --- 执行内存中的 PQ 训练和编码（内积距离） ---

    // const size_t M = InMemoryPQ_IP::M; // 从 pq_training_ip.h 获取
    // const size_t Ks = InMemoryPQ_IP::Ks; // 从 pq_training_ip.h 获取

    //  // 再次检查维度是否有效
    // if (vecdim == 0 || vecdim % M != 0) {
    //      std::cerr << "错误: vecdim=" << vecdim << " 为 0 或不能被 M=" << M << " 整除" << std::endl;
    //      delete[] test_query; delete[] test_gt; delete[] base;
    //      return 1;
    // }

    // std::cerr << "Starting IP-based PQ Training..." << std::endl; // 更新日志信息
    // std::vector<float> all_codebooks_vec = InMemoryPQ_IP::train_pq_ip_in_memory(base, base_number, vecdim);
    // // 检查训练是否成功 (逻辑不变)
    // if (all_codebooks_vec.empty()) {
    //      std::cerr << "错误: 内存中 IP PQ 训练失败。程序退出。" << std::endl; // 更新日志信息
    //      delete[] test_query; delete[] test_gt; delete[] base;
    //      return 1;
    // }

    // std::cerr << "Starting IP-based PQ Encoding..." << std::endl; // 更新日志信息
    // std::vector<uint8_t> base_codes_vec = InMemoryPQ_IP::encode_pq_ip_in_memory(base, base_number, vecdim, all_codebooks_vec);
    //  // 检查编码是否成功 (逻辑不变)
    //  if (base_codes_vec.empty()) {
    //      std::cerr << "错误: 内存中 IP PQ 编码失败。程序退出。" << std::endl; // 更新日志信息
    //      delete[] test_query; delete[] test_gt; delete[] base;
    //      // 注意: all_codebooks_vec 会自动清理
    //      return 1;
    // }

    std::vector<SearchResult> results;
    results.resize(test_number);

    // --- 对base数据进行量化（SQ） ---

    // std::cerr << "Quantizing base data...\n";
    // float data_min_val = std::numeric_limits<float>::max();
    // float data_max_val = std::numeric_limits<float>::lowest(); // 或者 -max()

    // // 找到全局min和max
    // for (size_t i = 0; i < base_number; ++i) {
    //     for (size_t d = 0; d < vecdim; ++d) {
    //         float val = base[i * vecdim + d];
    //         if (val < data_min_val) data_min_val = val;
    //         if (val > data_max_val) data_max_val = val;
    //     }
    // }
    // std::cerr << "Base data range: [" << data_min_val << ", " << data_max_val << "]\n";

    // // 计算scale
    // float data_scale = (data_max_val - data_min_val) / 255.0f;

    // // 创建量化后的base数组,类型uint8
    // std::vector<uint8_t> base_quantized_vec(base_number * vecdim);

    // // 量化 base 数据
    // #pragma omp parallel for // 并行
    // for (size_t i = 0; i < base_number; ++i) {
    //     for (size_t d = 0; d < vecdim; ++d) {
    //         size_t index = i * vecdim + d; // 当前基向量下标
    //         base_quantized_vec[index] = quantize_float_to_uint8(base[index], data_min_val, data_scale);
    //     }
    // }

    // // 获取指向量化数据的指针
    // const uint8_t* base_quantized = base_quantized_vec.data();

    // std::cerr << "Base data quantization finished.\n";

    // // --- 量化结束 ---


    
     // 获取指向内存数据的指针
     const float* codebooks_ptr = all_codebooks_vec.data();
     const uint8_t* base_codes_ptr = base_codes_vec.data();
    // 查询测试代码
    for(int i = 0; i < test_number; ++i) {
        const unsigned long Converter = 1000 * 1000;
        struct timeval val;
        int ret = gettimeofday(&val, NULL);

        // 该文件已有代码中你只能修改该函数的调用方式
        // 可以任意修改函数名，函数参数或者改为调用成员函数，但是不能修改函数返回值。

        //-----------------------------------------------------内积距离计算——串行----------------------------------------------------------

        //auto res = flat_search(base, test_query + i*vecdim, base_number, vecdim, k);

        // -----------------------------------------------------内积距离计算——SIMD(NEON)--------------------------------------------------

        //auto res = flat_search_simd_neon(base, test_query + i*vecdim, base_number, vecdim, k);

        //------------------------------------------------------内积距离计算——SQ+SIMD(NEON)--------------------------------------------------

        // auto res = flat_search_sq_simd_neon(
        //     base_quantized,       // 传入预量化的基数据
        //     test_query + i*vecdim, // 传入当前的浮点查询向量
        //     data_min_val,         // 传入量化最小值
        //     data_scale,           // 传入量化缩放因子
        //     base_number,
        //     vecdim,
        //     k
        // );

        //------------------------------------------------------内积距离计算——SQ+SIMD(NEON)+rerank---------------------------------------------

        // // --- 阶段一：使用 SQ+SIMD 获取 p 个候选者索引 ---
        // const float* current_query_float = test_query + (size_t)i * vecdim;

        // // 调用 SQ 搜索函数获取 Top-p 候选 (基于伪距离)
        // auto candidate_pq = flat_search_sq_simd_neon( // 假设使用无点积指令的版本
        //     base_quantized,
        //     current_query_float,
        //     data_min_val,
        //     data_scale,
        //     base_number,
        //     vecdim,
        //     p // *** 获取 p 个候选 ***
        // );

        // // 提取候选者的索引
        // std::vector<uint32_t> candidate_indices;
        // candidate_indices.reserve(p);
        // while (!candidate_pq.empty()) {
        //     candidate_indices.push_back(candidate_pq.top().second);
        //     candidate_pq.pop();
        // }
        // // 现在 candidate_indices 包含了 p 个最有可能的候选索引

        // // --- 阶段二：调用新的精确重排函数 ---
        // // 这个函数内部会遍历 candidate_indices，计算精确距离并返回最终 Top-K
        // auto res = rerank_float_simd_neon(
        //     base,                   // 传入原始 float 基数据
        //     current_query_float,    // 传入当前 float 查询
        //     candidate_indices,      // 传入 p 个候选索引
        //     vecdim,                 // 维度
        //     k                       // 最终需要的 Top-K 数量
        // );
        // // 'res' 现在直接就是基于精确距离重排后的 Top-K 优先队列

        //---------------------------------------------------内积距离计算——PQ+SIMD(NEON)---------------------------------------------------

        // const float* current_query_float = test_query + (size_t)i * vecdim;

        // // === 调用新的 PQ+SIMD 内积近似搜索函数 ===
        // auto res = flat_search_pq_ip_simd_neon( // <--- 调用 IP 版本函数
        //     base_codes_ptr,
        //     current_query_float,
        //     codebooks_ptr,
        //     base_number,
        //     vecdim,
        //     M,
        //     Ks,
        //     k
        // );
        // // ========================================
        







        //---------------------------------------------------内积距离计算——PQ+SIMD(NEON)+rerank-----------------------------------------------------

        // const float* current_query_float = test_query + (size_t)i * vecdim;

        // // --- 阶段一：使用 IP PQ+SIMD 获取 p 个候选者索引 ---
        // auto candidate_pq_ip = flat_search_pq_ip_simd_neon( // 调用 IP 近似函数
        //     base_codes_ptr, current_query_float, codebooks_ptr,
        //     base_number, vecdim, M, Ks,
        //     p // *** 获取 p 个候选 ***
        // );

        // // 提取候选索引
        // std::vector<uint32_t> candidate_indices;
        // candidate_indices.reserve(candidate_pq_ip.size());
        // while (!candidate_pq_ip.empty()) {
        //     candidate_indices.push_back(candidate_pq_ip.top().second);
        //     candidate_pq_ip.pop();
        // }
  


        // // --- 阶段二：调用内积精确重排函数 ---
        // auto res = rerank_float_simd_neon( // 调用内积 Rerank 函数
        //     base,                   // 原始 float 数据
        //     current_query_float,    // 当前查询
        //     candidate_indices,      // p 个候选索引
        //     vecdim,                 // 维度
        //     k                       // 最终 Top-K
        // );
        // // 'res' 是基于精确内积距离重排后的 Top-K



         //-----------------------best！！！！-------------------------内积距离计算——PQ+SIMD(NEON)(L2)+rerank（内积）-----------------------------------------------------

          const float* current_query_float = test_query + (size_t)i * vecdim; // 当前查询向量指针

          auto candidate_pq = flat_search_pq_simd_neon(
              base_codes_ptr,
              current_query_float,
              codebooks_ptr,
              base_number,
              vecdim,
              M,
              Ks,
              p // *** 获取 p 个候选 ***
          );
  
          // 提取候选者的索引到 vector 中
          std::vector<uint32_t> candidate_indices;
          candidate_indices.reserve(candidate_pq.size()); // 预分配空间
          while (!candidate_pq.empty()) {
              // PQ 返回的是近似距离，值越小越好，优先队列顶部是最大距离（最差的候选）
              // 但 rerank 函数不关心这个顺序，只需要索引列表
              candidate_indices.push_back(candidate_pq.top().second);
              candidate_pq.pop();
          }

          auto res = rerank_float_simd_neon(
              base,                   // 传入原始 float 基数据
              current_query_float,    // 传入当前 float 查询
              candidate_indices,      // 传入 p 个候选索引
              vecdim,                 // 维度
              k                       // 最终需要的 Top-K 数量
          );
          


         

        


        //--------------------------------------------------欧几里得距离计算——串行--------------------------------------------------------

        //  const float* current_query_l2 = test_query + (size_t)i * vecdim;
        //  auto res = flat_search_l2_serial(
        //     base,               // 原始 float 基数据
        //     current_query_l2,   // 当前查询向量
        //     base_number,      // 基向量数量
        //     vecdim,           // 维度
        //     k                 // Top-K
        // );

        //----------------------------------------------欧几里得距离计算——SIMD(NEON)---------------------------------------------------

        // 获取当前查询向量的指针
        // const float* current_query_l2 = test_query + (size_t)i * vecdim;

        // // === 调用欧几里得距离 SIMD 搜索函数 ===
        // auto res = flat_search_l2_simd_neon( // <--- 修改函数名为 SIMD 版本
        //     base,               // 原始 float 基数据
        //     current_query_l2,   // 当前查询向量
        //     base_number,      // 基向量数量
        //     vecdim,           // 维度
        //     k                 // Top-K
        // );
        // ====================================


        //-------------------------------------------欧几里得距离计算——SQ+SIMD(NEON)-----------------------------------------------------

        // const float* current_query_float = test_query + (size_t)i * vecdim;

        // // === 调用 SQ L2 SIMD 搜索函数 ===
        // auto res = flat_search_sq_l2_simd_neon( // <--- 调用 L2 版本函数
        //     base_quantized,       // 传入预量化的基数据
        //     current_query_float,  // 传入当前的浮点查询向量
        //     data_min_val,         // 传入量化最小值
        //     data_scale,           // 传入量化缩放因子
        //     base_number,
        //     vecdim,
        //     k
        // );

        //--------------------------------------------欧几里得距离计算——SQ+SIMD(NEON)+rerank--------------------------------------------

        // const float* current_query_float = test_query + (size_t)i * vecdim;

        // // --- 阶段一：使用 SQ+SIMD (L2) 获取 p 个候选者索引 ---
        // // 调用 L2 版本的 SQ 搜索，获取 Top-p 候选
        // auto candidate_pq_l2 = flat_search_sq_l2_simd_neon(
        //     base_quantized,
        //     current_query_float,
        //     data_min_val,
        //     data_scale,
        //     base_number,
        //     vecdim,
        //     p // *** 获取 p 个候选 ***
        // );

        // // 提取候选者的索引到 vector 中
        // std::vector<uint32_t> candidate_indices;
        // candidate_indices.reserve(candidate_pq_l2.size());
        // while (!candidate_pq_l2.empty()) {
        //     candidate_indices.push_back(candidate_pq_l2.top().second);
        //     candidate_pq_l2.pop();
        // }



        // // --- 阶段二：调用新的 L2 精确重排函数 ---
        // // 使用 rerank_l2_simd_neon 对这 p 个候选进行精确 L2 排序
        // auto res = rerank_l2_simd_neon( // <--- 调用新的 L2 rerank 函数
        //     base,                   // 传入原始 float 基数据
        //     current_query_float,    // 传入当前 float 查询
        //     candidate_indices,      // 传入 p 个候选索引
        //     vecdim,                 // 维度
        //     k                       // 最终需要的 Top-K 数量
        // );
        // // 'res' 现在是基于精确 L2 距离重排后的 Top-K 优先队列

        

        //-----------------------------------------------------欧几里得距离计算——PQ+SIMD(NEON)------------------------------------------------

        // const float* current_query_float = test_query + (size_t)i * vecdim;

        // auto res = flat_search_pq_simd_neon( // 调用 L2 近似函数
        //     base_codes_ptr, current_query_float, codebooks_ptr,
        //     base_number, vecdim, M, Ks,
        //     k // *** 获取 p 个候选 ***
        // );

        //-----------------------------------------欧几里得距离计算——PQ+SIMD(NEON)+rerank------------------------------------------------------

        // --- 阶段一：使用 PQ+SIMD (近似 L2) 获取 p 个候选者索引 ---
        //  const float* current_query_float = test_query + (size_t)i * vecdim;

        //  auto candidate_pq_l2 = flat_search_pq_simd_neon( // 调用 L2 近似函数
        //     base_codes_ptr, current_query_float, codebooks_ptr,
        //     base_number, vecdim, M, Ks,
        //     p // *** 获取 p 个候选 ***
        // );

        // // 提取候选索引
        // std::vector<uint32_t> candidate_indices;
        // candidate_indices.reserve(candidate_pq_l2.size());
        // while (!candidate_pq_l2.empty()) {
        //     candidate_indices.push_back(candidate_pq_l2.top().second);
        //     candidate_pq_l2.pop();
        // }

        // // --- 阶段二：调用 L2 精确重排函数 ---
        // auto res = rerank_l2_simd_neon( // *** 调用 L2 Rerank 函数 ***
        //     base,                   // 原始 float 数据
        //     current_query_float,    // 当前查询
        //     candidate_indices,      // p 个候选索引
        //     vecdim,                 // 维度
        //     k                       // 最终 Top-K
        // );
        // //'res' 是基于精确 L2 距离重排后的 Top-K

        
        struct timeval newVal;
        ret = gettimeofday(&newVal, NULL);
        int64_t diff = (newVal.tv_sec * Converter + newVal.tv_usec) - (val.tv_sec * Converter + val.tv_usec);

        std::set<uint32_t> gtset;
        for(int j = 0; j < k; ++j){
            int t = test_gt[j + i*test_gt_d];
            gtset.insert(t);
        }

        size_t acc = 0;
        while (res.size()) {   
            int x = res.top().second;
            if(gtset.find(x) != gtset.end()){
                ++acc;
            }
            res.pop();
        }
        float recall = (float)acc/k;

        results[i] = {recall, diff};
    }

    float avg_recall = 0, avg_latency = 0;
    for(int i = 0; i < test_number; ++i) {
        avg_recall += results[i].recall;
        avg_latency += results[i].latency;
    }

    // 浮点误差可能导致一些精确算法平均recall不是1
    std::cout << "average recall: "<<avg_recall / test_number<<"\n";
    std::cout << "average latency (us): "<<avg_latency / test_number<<"\n";
    return 0;
}