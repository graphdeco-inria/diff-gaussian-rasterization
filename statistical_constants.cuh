#pragma once

// 最大样本数
#define MAX_STD_SAMPLES 1000

#ifdef __CUDACC__   // 只有 NVCC 编译 .cu 时才看到下面两行
__constant__ float BASE_SAMPLES_MAX[MAX_STD_SAMPLES * 3];
__constant__ int   NUM_STD_SAMPLES;
#endif
