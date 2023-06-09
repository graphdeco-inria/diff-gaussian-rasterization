from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
os.path.dirname(os.path.abspath(__file__))

setup(
    name="diff_rasterizationCUDA",
    ext_modules=[
        CUDAExtension(
            name="diff_rasterizationCUDA._C",
            sources=[
            "rasterize_points.cu", 
            "cuda_rasterizer/rasterizer_impl.cu",
            "cuda_rasterizer/forward.cu",
            "cuda_rasterizer/backward.cu",
            "ext.cpp"],
            extra_compile_args={"nvcc": ["-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/")],
                                "cxx": ["/wd4624"]})
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
