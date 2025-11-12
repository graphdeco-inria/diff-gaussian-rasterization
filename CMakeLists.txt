#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

cmake_minimum_required(VERSION 3.20)

project(DiffRast LANGUAGES CXX)
find_package(HIP)
if(HIP_FOUND)
	set(HIP_ENABLED ON)
	project(DiffRast LANGUAGES HIP)
else()
	project(DiffRast LANGUAGES CUDA)
endif()


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)

if(HIP_ENABLED)
	set(CMAKE_HIP_STANDARD 17)
	message(STATUS "Building with HIP support")
else()
	set(CMAKE_CUDA_STANDARD 17)
	message(STATUS "Building with CUDA support")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

if(HIP_ENABLED)
	add_library(HipRasterizer
		hip_rasterizer/backward.h
		hip_rasterizer/backward.hip
		hip_rasterizer/forward.h
		hip_rasterizer/forward.hip
		hip_rasterizer/auxiliary.h
		hip_rasterizer/rasterizer_impl.hip
		hip_rasterizer/rasterizer_impl.h
		hip_rasterizer/rasterizer.h
	)
	set_target_properties(HipRasterizer PROPERTIES HIP_ARCHITECTURES "gfx908;gfx90a;gfx942")
	target_include_directories(HipRasterizer PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/hip_rasterizer)
    target_include_directories(HipRasterizer PRIVATE ${CMAKE_HIP_TOOLKIT_INCLUDE_DIRECTORIES})
	target_include_directories(HipRasterizer PRIVATE third_party/glm)
else()
	add_library(CudaRasterizer
		cuda_rasterizer/backward.h
		cuda_rasterizer/backward.cu
		cuda_rasterizer/forward.h
		cuda_rasterizer/forward.cu
		cuda_rasterizer/auxiliary.h
		cuda_rasterizer/rasterizer_impl.cu
		cuda_rasterizer/rasterizer_impl.h
		cuda_rasterizer/rasterizer.h
	)
	set_target_properties(CudaRasterizer PROPERTIES CUDA_ARCHITECTURES "70;75;86")
	target_include_directories(CudaRasterizer PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/cuda_rasterizer)
    target_include_directories(CudaRasterizer PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endif()