#!/usr/bin/env python
"""
功能性单元测试：验证基于采样的三维高斯投影前向流程

运行方法::

    # 确保先编译并安装你的 CUDA 扩展 (python setup.py install)
    python test_forward_sampling.py --device cuda --N 10000

脚本流程：
1. 读取或生成 N 个标准三维正态样本 (𝒩(0,I)).
2. 使用 Cholesky 将其线性变换为目标高斯 (均值+协方差)。
3. 透视投影到二维，计算 CPU 端样本均值与协方差并与解析结果比对。
4. （可选）调用 diff_gaussian_rasterization 的 forward 函数，
   只要 forward 能跑通且不抛错，就算 GPU 端管线基本连通。

如需测试多高斯、不同相机等，可自行扩展 run_single_gaussian_test。
"""
import argparse
from pathlib import Path
import os

import numpy as np
import torch

# ---------------- 全局配置 ----------------
DEFAULT_STD_SAMPLE_PATH = "std3d_samples.npy"
DEFAULT_N_SAMPLES = 10_000
TOL_MEAN = 1e-3   # 二维均值容差
TOL_COV  = 1e-2   # 二维协方差 Frobenius 误差容差


# ---------------- 工具函数 ----------------

def load_or_create_std_samples(N: int, path: str = DEFAULT_STD_SAMPLE_PATH, seed: int = 2025) -> torch.Tensor:
    """生成或加载固定的标准三维正态样本 (N,3)。保存为 .npy 以便下次复用。"""
    rng = np.random.default_rng(seed)
    p = Path(path)
    if p.exists():
        z = np.load(p)
        if z.shape[0] != N:
            z = rng.standard_normal(size=(N, 3)).astype(np.float32)
            np.save(p, z)
    else:
        z = rng.standard_normal(size=(N, 3)).astype(np.float32)
        np.save(p, z)
    return torch.from_numpy(z)


def cholesky_transform(std_samples: torch.Tensor, mean: torch.Tensor, cov: torch.Tensor) -> torch.Tensor:
    """x = mean + L z, 其中 L 为 Cholesky 下三角矩阵。"""
    L = torch.linalg.cholesky(cov)
    return mean + std_samples @ L.T


def project_perspective(points: torch.Tensor, view: torch.Tensor, proj: torch.Tensor) -> torch.Tensor:
    """将 (N,3) 点做透视投影 (view+proj) 并透视除法 => (N,2)。"""
    N = points.shape[0]
    ones = torch.ones((N, 1), dtype=points.dtype, device=points.device)
    homo = torch.cat([points, ones], dim=1)  # (N,4)
    clip = (proj @ (view @ homo.T)).T        # (N,4)
    ndc  = clip[:, :2] / clip[:, 3:4]
    return ndc


def compute_2d_moments(points2d: torch.Tensor):
    mean = points2d.mean(dim=0)
    diff = points2d - mean
    cov  = diff.T @ diff / points2d.shape[0]
    return mean, cov


def make_perspective(fov_deg: float = 90.0, aspect: float = 1.0,
                     near: float = 1.0, far: float = 20.0, device="cpu") -> torch.Tensor:
    """生成右手坐标系的 4x4 透视投影矩阵。"""
    f = 1.0 / torch.tan(torch.deg2rad(torch.tensor(fov_deg) / 2))
    P = torch.zeros((4, 4), dtype=torch.float32, device=device)
    P[0, 0] = f / aspect
    P[1, 1] = f
    P[2, 2] = far / (far - near)
    P[2, 3] = (-far * near) / (far - near)
    P[3, 2] = 1.0
    return P


# ---------------- 主测试函数 ----------------

def run_single_gaussian_test(N: int = DEFAULT_N_SAMPLES, device: str = "cuda"):
    """验证单个各向同性高斯投影后的 2D 均值/协方差是否符合理论值。"""
    device = torch.device(device)

    # 1) 标准样本
    z = load_or_create_std_samples(N).to(device)

    # 2) 构造测试高斯 (μ, Σ)
    sigma = 0.1
    mean  = torch.tensor([0.0, 0.0, 4.0], device=device)
    cov   = torch.diag(torch.full((3,), sigma ** 2, device=device))

    # 3) 线性变换到目标分布
    x = cholesky_transform(z, mean, cov)

    # 4) 相机矩阵 (简化：identity view + 90° fov projection)
    view = torch.eye(4, dtype=torch.float32, device=device)
    proj = make_perspective(device=device)

    # 5) 透视投影到 2D
    pts2d = project_perspective(x, view, proj)
    mean2d_cpu, cov2d_cpu = compute_2d_moments(pts2d)

    # 6) 理论期望：var = (σ / z)²
    expected_var = (sigma / mean[2]) ** 2
    expected_cov = torch.diag(torch.full((2,), expected_var, device=device))

    mean_err = torch.norm(mean2d_cpu)  # 理论值为 (0,0)
    cov_err  = torch.norm(cov2d_cpu - expected_cov)
    print(f"mean err = {mean_err:.4e},  cov err = {cov_err:.4e}")
    assert mean_err < TOL_MEAN, "二维均值超出容差"
    assert cov_err  < TOL_COV,  "二维协方差超出容差"
    print("CPU 数学验证通过  ✓\n")

    # ------------- GPU Kernel 调用（可选）-------------
    try:
        # 根据实际模块名修改此处
        import diff_gaussian_rasterization as dgr

        colors    = torch.tensor([[1., 1., 1., 1.]], device=device)  # RGBA
        opacities = torch.tensor([1.0], device=device)

        with torch.no_grad():
            # 请将参数名改成你在 ext.cpp 暴露给 Python 的签名
            out = dgr.forward(
                means3D = mean.unsqueeze(0),      # (1,3)
                cov3D   = cov.reshape(1, 9),      # (1,9)
                colors  = colors,                 # (1,4)
                opacities = opacities,            # (1,)
                view = view.flatten(),            # (16,)
                proj = proj.flatten(),            # (16,)
                image_height = 64,
                image_width  = 64,
                bg_color = torch.tensor([0., 0., 0.], device=device)
            )
        print("GPU forward 执行完成（未抛错） ✓")
    except ModuleNotFoundError:
        print("⚠️  未找到 diff_gaussian_rasterization 模块，跳过 GPU 端测试。")
    except Exception as e:
        print("❌  GPU forward 抛出异常：", e)
        raise


# ---------------- CLI ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=DEFAULT_N_SAMPLES, help="采样点个数")
    parser.add_argument("--device", default="cuda", help="'cuda' 或 'cpu'")
    args = parser.parse_args()
    run_single_gaussian_test(N=args.N, device=args.device)
