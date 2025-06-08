/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 */

#include <torch/extension.h>
#include "rasterize_points.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cuda_runtime.h>
#include <vector>
#include "statistical_constants.cuh"

namespace py = pybind11;

// 上传 samples.txt 到 constant memory
void upload_samples_to_constant(
    py::array_t<float, py::array::c_style | py::array::forcecast> arr,
    int N)
{
    auto info = arr.request();
    float* ptr = static_cast<float*>(info.ptr);
    size_t bytes = size_t(N) * 3 * sizeof(float);

    // —— 用 C API，并把符号地址转成 const void* ——  
    // 拷贝实际样本数 N
    cudaMemcpyToSymbol(
        (const void*)&NUM_STD_SAMPLES,  // symbol address
        &N,                              // src
        sizeof(int),                     // count
        0,                               // offset
        cudaMemcpyHostToDevice           // kind
    );

    // 拷贝样本数据
    cudaMemcpyToSymbol(
        (const void*)BASE_SAMPLES_MAX,   // symbol address
        ptr,                             // src
        bytes,                           // count = N*3*sizeof(float)
        0,                               // offset
        cudaMemcpyHostToDevice
    );
}

// 从 constant memory 拷回到 Host，返回 (loadedN,3) numpy array
py::array_t<float> get_samples_from_constant() {
    // 1. 拷回实际 N
    int loadedN = 0;
    cudaMemcpyFromSymbol(
        &loadedN,                       // dst
        (const void*)&NUM_STD_SAMPLES,  // symbol address
        sizeof(int),
        0,
        cudaMemcpyDeviceToHost
    );

    // 2. 拷回样本数据
    std::vector<float> host_data(size_t(loadedN) * 3);
    cudaMemcpyFromSymbol(
        host_data.data(),               // dst
        (const void*)BASE_SAMPLES_MAX,  // symbol address
        size_t(loadedN) * 3 * sizeof(float),
        0,
        cudaMemcpyDeviceToHost
    );

    // 3. 构造并返回 numpy array
    return py::array_t<float>(
        { loadedN, 3 },               // shape
        { 3 * sizeof(float),          // strides
          sizeof(float) },
        host_data.data()
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rasterize_gaussians", &RasterizeGaussiansCUDA);
  m.def("rasterize_gaussians_backward", &RasterizeGaussiansBackwardCUDA);
  m.def("mark_visible", &markVisible);

  // 新增接口
  m.def("upload_samples_to_constant", &upload_samples_to_constant,
        "Upload standard normal samples to GPU constant memory");
  m.def("get_samples_from_constant", &get_samples_from_constant,
        "Download samples from GPU constant memory for testing");
  m.attr("MAX_STD_SAMPLES") = py::int_(MAX_STD_SAMPLES);
}
