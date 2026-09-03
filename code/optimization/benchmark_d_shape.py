"""
shape 轴之「状态维 d」扫描 —— 评估最终图「吞吐 vs 输入 shape」唯一新实验。

方法线：eager / compile / torch-kf / Trident(fused triton)，同一线性滤波模型(d 状态, m=3)。
y = throughput (steps/s, batch=1)。d ∈ {6,9,15,18,21,27,30}。

用法：python code/optimization/benchmark_d_shape.py [--quick]
"""
import os
import sys
import json
import time
import argparse
import numpy as np
import torch

_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)
os.chdir(os.path.abspath(os.path.join(_here, '..', '..')))

from generic_filter import build_model, run_eager, run_compile, run_torchkf, run_triton

D_SWEEP = [6, 9, 15, 18, 21, 27, 30]
N = 2000
N_REPEATS = 3


def measure(fn, n_reps):
    """fn() 执行一次完整 N 步；返回 steps/s mean。"""
    torch.cuda.synchronize()
    vals = []
    for _ in range(n_reps):
        t0 = time.time()
        fn()
        torch.cuda.synchronize()
        vals.append(N / (time.time() - t0))
    return float(np.mean(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    args = ap.parse_args()
    dvals = [15] if args.quick else D_SWEEP

    results = {'N': N, 'hardware': {'gpu': torch.cuda.get_device_name(0),
                                    'sm': torch.cuda.get_device_properties(0).multi_processor_count},
               'd_sweep': []}
    print(f"d-shape sweep (N={N}, m=3, fp32, batch=1)")
    print(f"{'d':>3} {'eager':>10} {'compile':>10} {'torch-kf':>10} {'triton':>10}")

    for d in dvals:
        F, H, Q, R = build_model(d)
        z = torch.randn(N, 3, dtype=torch.float32, device='cuda')

        # warmup 各方法（eager 的 cuSOLVER / compile 的 JIT / triton 的 JIT）
        run_eager(F, H, Q, R, z, 16)
        run_compile(F, H, Q, R, z, 16)
        run_torchkf(F, H, Q, R, z, 16)
        run_triton(F, H, Q, R, z, 16)
        torch.cuda.synchronize()

        eager = measure(lambda: run_eager(F, H, Q, R, z, N), n_reps=1)          # 慢，跑 1 次
        comp = measure(lambda: run_compile(F, H, Q, R, z, N), n_reps=N_REPEATS)
        tkf = measure(lambda: run_torchkf(F, H, Q, R, z, N), n_reps=N_REPEATS)
        tri = measure(lambda: run_triton(F, H, Q, R, z, N), n_reps=N_REPEATS)

        results['d_sweep'].append(dict(d=d, eager=eager, compile=comp,
                                       torch_kf=tkf, triton=tri))
        print(f"{d:>3} {eager:>10.0f} {comp:>10.0f} {tkf:>10.0f} {tri:>10.0f}")

    out = 'data/results/a100_scaling_d.json'
    json.dump(results, open(out, 'w'), indent=2, ensure_ascii=False)
    print(f'\nsaved -> {out}')


if __name__ == '__main__':
    main()