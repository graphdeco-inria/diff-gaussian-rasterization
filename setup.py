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
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension, ROCM_HOME
from torch.utils.hipify import hipify_python
import os
import torch

# Include this line immediately after the import statements
TORCH_MAJOR = int(torch.__version__.split('.')[0])
TORCH_MINOR = int(torch.__version__.split('.')[1])
is_rocm = False
if TORCH_MAJOR > 1 or (TORCH_MAJOR == 1 and TORCH_MINOR >= 5):
  is_rocm = True if ((torch.version.hip is not None) and (ROCM_HOME is not None)) else False

current_dir = os.path.dirname(os.path.abspath(__file__))
hipify_source_dir = os.path.join(current_dir, "cuda_rasterizer")
hipify_source_file = os.path.join(current_dir, "rasterize_points.cu")
hipify_dst_dir = os.path.join(current_dir, "hip_rasterizer")
hipify_in_files = [hipify_source_dir, hipify_source_file]
hipify_out_dirs = [hipify_dst_dir, current_dir]

if is_rocm:
  print("[INFO] ROCm detected: running hipify...")
  for i in range(len(hipify_in_files)):
    hipify_python.hipify(
      project_directory=hipify_in_files[i],
      output_directory=hipify_out_dirs[i],
      show_detailed=True,
      is_pytorch_extension=True
    )
  source_files = ["hip_rasterizer/rasterizer_impl.hip", "hip_rasterizer/forward.hip", "hip_rasterizer/backward.hip", "rasterize_points.hip", "ext.cpp"]
else:
  print("[INFO] CUDA detected: using CUDA source directly.")
  source_files = ["cuda_rasterizer/rasterizer_impl.cu", "cuda_rasterizer/forward.cu", "cuda_rasterizer/backward.cu", "rasterize_points.cu", "ext.cpp"]
  
glm_include_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/")
setup(
    name="diff_gaussian_rasterization",
    packages=['diff_gaussian_rasterization'],
    ext_modules=[
        CUDAExtension(
            name="diff_gaussian_rasterization._C",
            sources=source_files,
            extra_compile_args={"nvcc":["-I"+glm_include_dir], "cxx":["-I"+glm_include_dir]})
        ],
    cmdclass={
        'build_ext': BuildExtension
    },
    version='1.0.0'
)
