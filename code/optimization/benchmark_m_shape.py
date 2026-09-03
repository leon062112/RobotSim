"""
shape 轴之「观测维 m」扫描（d=15 固定，m ∈ {2,3,6,9,12}）。

方法线：eager / compile / torch-kf / Trident(fused triton)。y = throughput (steps/s)。
与 benchmark_d_shape.py 成对：那张扫 d、这张扫 m，合起来 x 轴 = (d, m)。

用法：python code/optimization/benchmark_m_shape.py [--quick]
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

M_SWEEP = [2, 3, 6, 9, 12]
D = 15
N = 2000
N_REPEATS = 3


def measure(fn, n_reps):
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
    mvals = [3] if args.quick else M_SWEEP

    results = {'N': N, 'd': D, 'hardware': {'gpu': torch.cuda.get_device_name(0),
                                           'sm': torch.cuda.get_device_properties(0).multi_processor_count},
               'm_sweep': []}
    print(f"m-shape sweep (d={D}, N={N}, fp32, batch=1)")
    print(f"{'m':>3} {'eager':>10} {'compile':>10} {'torch-kf':>10} {'triton':>10}")

    for m in mvals:
        F, H, Q, R = build_model(D, m)
        z = torch.randn(N, m, dtype=torch.float32, device='cuda')

        run_eager(F, H, Q, R, z, 16)
        run_compile(F, H, Q, R, z, 16)
        run_torchkf(F, H, Q, R, z, 16)
        run_triton(F, H, Q, R, z, 16)
        torch.cuda.synchronize()

        eager = measure(lambda: run_eager(F, H, Q, R, z, N), n_reps=1)
        comp = measure(lambda: run_compile(F, H, Q, R, z, N), n_reps=N_REPEATS)
        tkf = measure(lambda: run_torchkf(F, H, Q, R, z, N), n_reps=N_REPEATS)
        tri = measure(lambda: run_triton(F, H, Q, R, z, N), n_reps=N_REPEATS)

        results['m_sweep'].append(dict(m=m, eager=eager, compile=comp,
                                       torch_kf=tkf, triton=tri))
        print(f"{m:>3} {eager:>10.0f} {comp:>10.0f} {tkf:>10.0f} {tri:>10.0f}")

    out = 'data/results/a100_scaling_m.json'
    json.dump(results, open(out, 'w'), indent=2, ensure_ascii=False)
    print(f'\nsaved -> {out}')


if __name__ == '__main__':
    main()