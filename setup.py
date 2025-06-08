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
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os

here = os.path.dirname(os.path.abspath(__file__))

setup(
    name="diff_gaussian_rasterization",
    packages=['diff_gaussian_rasterization'],
    ext_modules=[
        CUDAExtension(
            name="diff_gaussian_rasterization._C",
            sources=[
                "cuda_rasterizer/rasterizer_impl.cu",
                "cuda_rasterizer/forward.cu",
                "cuda_rasterizer/backward.cu",
                "rasterize_points.cu",
                "ext.cpp",
            ],
            # 关键就在这里，把两个目录都加进来
            include_dirs=[
                os.path.join(here, "cuda_rasterizer"),
                os.path.join(here, "third_party", "glm"),
            ],
            extra_compile_args={
                "nvcc": [
                    "-allow-unsupported-compiler",
                    # 如果需要指定其它 nvcc 参数，也放在这里
                ]
            },
        )
    ],
    cmdclass={'build_ext': BuildExtension},
)
