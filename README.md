# ann-SIMD
ip.h和l2.h中是内积距离和欧几里得距离的各种算法函数，pq_training_ip.h和pq_training.h分别是基于IP和L2的PQ训练编码函数，main中调用这些量化函数和优化算法函数进行测试，现已经调至最好状态，test.o中存储当前最好状态的测试结果，可以通过修改调用函数测试其他方法组合。ann-SIMD文件夹内是按照recall@10=0.95的的标准测试的，ann-SIMD文件夹外是更新的recall@10=0.9标准下的测试结果和main代码修改。
![image](https://github.com/user-attachments/assets/4ec3d679-7b08-486c-b252-27bf1ba1ab5f)
