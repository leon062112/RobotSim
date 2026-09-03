"""
torch-kf baseline（领域批量卡尔曼滤波库）— 对标 Trident 的 batch 轴。

torch-kf 是 batched LINEAR Gaussian filter/smoother（PyTorch 实现）。这里用它跑
d=15 状态、3 维观测、B 条独立轨迹的线性滤波，测其批量吞吐，与 Trident v5（fused
whole-scan + 非线性 SINS/EKF）对比 batch 轴。

关键 caveat（论文须如实声明）：
  - torch-kf 只能表达 LINEAR filter（固定 F/H/Q/R 或逐步传入），无法表达目标负载的
    非线性 SINS 机械编排、四元数姿态、闭环反馈、解析 3×3 逆与里程计异常判别。
  - 因此这里是「同维线性滤波核心」的吞吐对比：Torch-kf 用逐步 Python 循环 + batched
    cuBLAS，Trident v5 用单 kernel 融合整条串行 scan。

用法（仓库根目录）：
  python code/optimization/benchmark_torchkf.py                 # B 扫描（小 N）+ 全量 N 一个点
  python code/optimization/benchmark_torchkf.py --quick         # 冒烟
"""
import os
import sys
import json
import time
import argparse
import numpy as np
import torch
from torch_kf import KalmanFilter, GaussianState

_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)
os.chdir(os.path.abspath(os.path.join(_here, '..', '..')))

D = 15          # 状态维
M = 3           # 观测维
DT = 0.01
N_FULL = 166667


def build_model(B, dev='cuda', dtype=torch.float64):
    """构造 d=15/m=3 的固定线性模型（与 Trident 同维、同 dt），batched over B。"""
    torch.manual_seed(0)
    F = torch.eye(D, dtype=dtype, device=dev).repeat(B, 1, 1).contiguous()
    # 加一个小的时不变扰动使线性模型具备一般性（逐位值不影响吞吐，只影响数值）
    F = F + 0.001 * torch.randn(B, D, D, dtype=dtype, device=dev)
    H = torch.zeros(B, M, D, dtype=dtype, device=dev)
    H[:, 0, 3] = 1.0   # 观测速度 x 误差
    H[:, 1, 0] = 1.0   # 横向速度约束
    H[:, 2, 2] = 1.0   # 径向速度约束
    Q = torch.eye(D, dtype=dtype, device=dev).repeat(B, 1, 1) * 1e-6
    R = torch.eye(M, dtype=dtype, device=dev).repeat(B, 1, 1) * 1e-3
    return F, H, Q, R


def run_torchkf(B, N, dev='cuda', dtype=torch.float64, verbose=False):
    F, H, Q, R = build_model(B, dev, dtype)
    kf = KalmanFilter(F, H, Q, R)
    mean = torch.zeros(B, D, 1, dtype=dtype, device=dev)
    cov = torch.eye(D, dtype=dtype, device=dev).repeat(B, 1, 1) * 0.1
    state = GaussianState(mean, cov)
    torch.manual_seed(1)
    measures = torch.randn(N, B, M, 1, dtype=dtype, device=dev)

    # warmup (小 N)
    kf.filter(state, measures[:min(N, 32)], return_all=False)
    torch.cuda.synchronize()

    t0 = time.time()
    kf.filter(state, measures, return_all=False)
    torch.cuda.synchronize()
    elapsed = time.time() - t0

    total_steps = B * N
    return dict(version='torch_kf', batch=B, n_steps=N, elapsed_s=elapsed,
                throughput_steps_per_s=total_steps / elapsed,
                throughput_traj_per_s=B / elapsed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    args = ap.parse_args()

    N_sweep = 2000 if not args.quick else 500
    sm = torch.cuda.get_device_properties(0).multi_processor_count
    Bs = [1, 16, 64, sm]

    print('=' * 70)
    print('torch-kf baseline (batched linear KF, d=15 m=3 fp64)')
    print('=' * 70)

    results = {'hardware': {'gpu': torch.cuda.get_device_name(0),
                            'sm': torch.cuda.get_device_properties(0).multi_processor_count},
               'batch_sweep': [], 'full_N': None}

    print(f'\n[B sweep, N={N_sweep}]  (steps/s 为 B×N 总值)')
    for B in Bs:
        m = run_torchkf(B, N_sweep)
        results['batch_sweep'].append(m)
        per_traj = m['throughput_steps_per_s'] / B
        print(f"  B={B:4d}: {m['throughput_steps_per_s']:11.0f} steps/s "
              f"({per_traj:8.0f} steps/s/traj)")

    if not args.quick:
        print(f'\n[full N={N_FULL}, B={sm}] ...')
        m = run_torchkf(sm, N_FULL)
        results['full_N'] = m
        print(f"  B={sm}: {m['throughput_steps_per_s']:.0f} steps/s "
              f"({m['throughput_traj_per_s']:.3f} traj/s, {m['elapsed_s']:.1f}s)")

    out = 'data/results/a100_torchkf.json'
    json.dump(results, open(out, 'w'), indent=2, ensure_ascii=False)
    print(f'\nsaved -> {out}')


if __name__ == '__main__':
    main()