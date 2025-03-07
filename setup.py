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
import subprocess
os.path.dirname(os.path.abspath(__file__))

def check_and_install_libglm():
    try:
        print("Checking if libglm-dev is installed...")
        subprocess.run(["dpkg", "-s", "libglm-dev"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("libglm-dev is installed.")
    except subprocess.CalledProcessError:
        print("libglm-dev is not installed. Trying to install libglm-dev")
        subprocess.run(["sudo", "apt", "install", "libglm-dev"], check=True)
        print("libglm-dev is installed.")
    except Exception as e:
        print("libglm-dev is not installed. Installation also failed. Probable reason, missing root access. Try installing with root access or do the installtion manually.")
        print("Please share the whole error message with the developers. ref: https://github.com/maifeeulasad/diff-gaussian-rasterization, https://github.com/graphdeco-inria/diff-gaussian-rasterization")
        print("Error: ", e)
        raise RuntimeError("libglm-dev is not installed. Please install it using 'sudo apt install libglm-dev'")
    
check_and_install_libglm()

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
            "ext.cpp"],
            extra_compile_args={"nvcc": ["-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/")]})
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
