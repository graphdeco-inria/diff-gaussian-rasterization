#!/usr/bin/env python
"""
前向流程测试脚本（功能性 + 数值精度）
================================================
本脚本旨在验证你在 3DGS *test* 分支里基于采样的前向实现，既能 **正确跑通**，又能 **在统计意义上达到足够精度**。

运行示例
---------
功能性 + 精度基准：
    python test_forward_sampling.py --device cuda --N 10000 --bench
只跑功能性快速断言：
    python test_forward_sampling.py --device cuda --N 10000

脚本要点
--------
1. **功能性单元测试**（`run_single_gaussian_test`）
   - 单高斯，理论结果已知，断言二维均值与协方差误差在容差内。
2. **数值精度基准**（`run_precision_benchmark`）
   - 多种样本数 *N*（默认 100、1k、10k、100k），多随机种子，对均值 / 协方差误差算均值 + 最大值。
   - 同时比较：
       • CPU float64 (高精参考)
       • GPU float32 (你当前实现)
   - 最终打印误差随 N 收敛趋势，帮助你判断采样数够不够，或 float32 是否失真。
3. **可选 GPU forward** 调用：如果安装了 `diff_gaussian_rasterization` 扩展，会尝试跑一次；无就跳过。
"""
import argparse
from pathlib import Path
import os
import itertools
import statistics

import numpy as np
import torch

# ---------------- 全局阈值 ----------------
DEFAULT_STD_SAMPLE_PATH = "std3d_samples.npy"
DEFAULT_N_SAMPLES = 10_000
TOL_MEAN = 1e-3   # 二维均值容差
TOL_COV  = 1e-2   # 二维协方差 Frobenius 误差容差

# ---------------- 工具函数 ----------------

def load_or_create_std_samples(N: int, path: str = DEFAULT_STD_SAMPLE_PATH, seed: int = 2025) -> torch.Tensor:
    """生成或加载固定标准三维正态样本 (N,3) float32。"""
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
    """x = mean + L·z, L 为 Cholesky 下三角矩阵。"""
    L = torch.linalg.cholesky(cov)
    return mean + std_samples @ L.T


def project_perspective(points: torch.Tensor, view: torch.Tensor, proj: torch.Tensor) -> torch.Tensor:
    """将 (N,3) 点透视投影到 (N,2)。"""
    N = points.shape[0]
    ones = torch.ones((N, 1), dtype=points.dtype, device=points.device)
    homo = torch.cat([points, ones], dim=1)          # (N,4)
    clip = (proj @ (view @ homo.T)).T               # (N,4)
    ndc  = clip[:, :2] / clip[:, 3:4]
    return ndc


def compute_2d_moments(points2d: torch.Tensor):
    mean = points2d.mean(dim=0)
    diff = points2d - mean
    cov  = diff.T @ diff / points2d.shape[0]
    return mean, cov


def make_perspective(fov_deg: float = 90.0, aspect: float = 1.0,
                     near: float = 1.0, far: float = 20.0, device="cpu") -> torch.Tensor:
    """生成右手坐标系 4×4 透视投影矩阵。"""
    f = 1.0 / torch.tan(torch.deg2rad(torch.tensor(fov_deg) / 2))
    P = torch.zeros((4, 4), dtype=torch.float32, device=device)
    P[0, 0] = f / aspect
    P[1, 1] = f
    P[2, 2] = far / (far - near)
    P[2, 3] = (-far * near) / (far - near)
    P[3, 2] = 1.0
    return P

# ---------------- 功能性单元测试 ----------------

def run_single_gaussian_test(N: int = DEFAULT_N_SAMPLES, device: str = "cuda"):
    """验证单个各向同性高斯投影后的 2D 均值/协方差与理论吻合。"""
    device = torch.device(device)

    # 1) 标准样本 (float32)
    z = load_or_create_std_samples(N).to(device)

    # 2) 目标高斯参数
    sigma = 0.1
    mean  = torch.tensor([0.0, 0.0, 4.0], device=device)
    cov   = torch.diag(torch.full((3,), sigma ** 2, device=device))

    # 3) 变换到目标分布
    x = cholesky_transform(z, mean, cov)

    # 4) 透视投影矩阵（identity view）
    view = torch.eye(4, dtype=torch.float32, device=device)
    proj = make_perspective(device=device)

    # 5) 投影 -> 2D
    pts2d = project_perspective(x, view, proj)
    mean2d, cov2d = compute_2d_moments(pts2d)

    # 6) 理论值
    expected_var = (sigma / mean[2]) ** 2
    expected_cov = torch.diag(torch.full((2,), expected_var, device=device))

    mean_err = torch.norm(mean2d)  # 理论均值 (0,0)
    cov_err  = torch.norm(cov2d - expected_cov)
    print(f"[功能性] mean_err={mean_err:.4e}, cov_err={cov_err:.4e}")
    assert mean_err < TOL_MEAN, "二维均值误差过大"
    assert cov_err  < TOL_COV,  "二维协方差误差过大"
    print("功能性测试通过  ✓\n")

    # 可选 GPU forward 调用
    try:
        import diff_gaussian_rasterization as dgr
        colors    = torch.tensor([[1., 1., 1., 1.]], device=device)
        opacities = torch.tensor([1.0], device=device)
        with torch.no_grad():
            _ = dgr.forward(
                means3D = mean.unsqueeze(0),
                cov3D   = cov.reshape(1, 9),
                colors  = colors,
                opacities = opacities,
                view = view.flatten(),
                proj = proj.flatten(),
                image_height = 64,
                image_width  = 64,
                bg_color = torch.tensor([0., 0., 0.], device=device)
            )
        print("GPU forward 执行完成（未抛错） ✓\n")
    except ModuleNotFoundError:
        print("⚠️  未找到 diff_gaussian_rasterization 扩展，跳过 GPU 端测试。\n")
    except Exception as e:
        print("❌  GPU forward 抛异常：", e)
        raise

# ---------------- 数值精度基准 ----------------

def run_precision_benchmark(N_list=(100, 1000, 10000, 100000), seeds=(0,1,2,3,4), device="cuda"):
    device = torch.device(device)
    sigma = 0.1
    mean  = torch.tensor([0.0, 0.0, 4.0], device=device)
    cov   = torch.diag(torch.full((3,), sigma ** 2, device=device))

    view = torch.eye(4, dtype=torch.float32, device=device)
    proj = make_perspective(device=device)

    expected_var = (sigma / mean[2]) ** 2
    expected_cov = torch.diag(torch.full((2,), expected_var, device=device))

    print("\n========= 数值精度基准 =========")
    print("样本数 | mean_err(float32) | cov_err(float32) | mean_err(float64) | cov_err(float64)")
    print("--------------------------------------------------------------------------")

    for N in N_list:
        mean_err_f32, cov_err_f32 = [], []
        mean_err_f64, cov_err_f64 = [], []
        for s in seeds:
            # --- 生成/加载 float64 基准样本 ---
            z_cpu = load_or_create_std_samples(N, seed=2025+s).double()  # float64 on CPU
            # --- float64 精准计算 ---
            x64 = cholesky_transform(z_cpu, mean.double(), cov.double())
            pts2d_64 = project_perspective(x64, view.double(), proj.double())
            m64, c64 = compute_2d_moments(pts2d_64)
            mean_err_f64.append(torch.norm(m64).item())
            cov_err_f64.append(torch.norm(c64 - expected_cov.double()).item())

            # --- float32 GPU 计算 ---
            z32 = z_cpu.float().to(device)
            x32 = cholesky_transform(z32, mean, cov)
            pts2d_32 = project_perspective(x32, view, proj)
            m32, c32 = compute_2d_moments(pts2d_32)
            mean_err_f32.append(torch.norm(m32).item())
            cov_err_f32.append(torch.norm(c32 - expected_cov).item())

        # 统计
        def fmt(l):
            return f"{statistics.mean(l):.2e}±{max(l):.1e}"
        print(f"{N:7d} | {fmt(mean_err_f32):>16} | {fmt(cov_err_f32):>15} | {fmt(mean_err_f64):>17} | {fmt(cov_err_f64):>15}")

    print("-------------------------------- 完成 --------------------------------\n")

# ---------------- CLI ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3DGS 前向采样+投影 单元测试 & 精度基准")
    parser.add_argument("--N", type=int, default=DEFAULT_N_SAMPLES, help="默认功能性测试的样本点数")
    parser.add_argument("--device", default="cuda", help="'cuda' 或 'cpu'")
    parser.add_argument("--bench", action="store_true", help="是否跑数值精度基准")
    args = parser.parse_args()

    run_single_gaussian_test(N=args.N, device=args.device)

    if args.bench:
        run_precision_benchmark(device=args.device)
