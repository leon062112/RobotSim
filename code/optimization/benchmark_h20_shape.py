"""
H20 全量 (d,m) shape benchmark —— 5 条方法线（eager/compile/torch-kf/prefix/triton）。

由于现网 a100_scaling_*.json 均为 A100/108SM 数据，本脚本在 H20/78SM 上重跑全部方法，
保证最终图 5 条线同机同源。输出 h20_scaling_d.json 与 h20_scaling_m.json。

用法：python code/optimization/benchmark_h20_shape.py [--quick]
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

from generic_filter import build_model, run_eager, run_compile, run_torchkf, run_triton, run_prefix

D_SWEEP = [6, 9, 15, 18, 21, 27, 30]
M_SWEEP = [1, 2, 3]
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


def run_sweep(sweep_name, sweep_vals, fixed_d, fixed_m):
    results = []
    print(f"\n[{sweep_name}-sweep, N={N}, fp32, batch=1]")
    print(f"{sweep_name:>3} {'eager':>10} {'compile':>10} {'torch-kf':>10} {'prefix':>10} {'triton':>10}")

    for val in sweep_vals:
        d = val if sweep_name == 'd' else fixed_d
        m = val if sweep_name == 'm' else fixed_m

        F, H, Q, R = build_model(d, m=m)
        z = torch.randn(N, m, dtype=torch.float32, device='cuda')

        # warmup 各方法
        run_eager(F, H, Q, R, z, 16)
        run_compile(F, H, Q, R, z, 16)
        run_torchkf(F, H, Q, R, z, 16)
        run_prefix(F, H, Q, R, z, 16)
        run_triton(F, H, Q, R, z, 16)
        torch.cuda.synchronize()

        eager = measure(lambda: run_eager(F, H, Q, R, z, N), n_reps=1)          # 慢，跑 1 次
        comp = measure(lambda: run_compile(F, H, Q, R, z, N), n_reps=N_REPEATS)
        tkf = measure(lambda: run_torchkf(F, H, Q, R, z, N), n_reps=N_REPEATS)
        pre = measure(lambda: run_prefix(F, H, Q, R, z, N), n_reps=N_REPEATS)
        tri = measure(lambda: run_triton(F, H, Q, R, z, N), n_reps=N_REPEATS)

        results.append(dict(**{sweep_name: val}, eager=eager, compile=comp,
                            torch_kf=tkf, prefix=pre, triton=tri))
        print(f"{val:>3} {eager:>10.0f} {comp:>10.0f} {tkf:>10.0f} {pre:>10.0f} {tri:>10.0f}")

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    args = ap.parse_args()

    hw = {'gpu': torch.cuda.get_device_name(0),
          'sm': torch.cuda.get_device_properties(0).multi_processor_count}

    if args.quick:
        d_results = run_sweep('d', [15], fixed_d=15, fixed_m=3)
        m_results = run_sweep('m', [3], fixed_d=15, fixed_m=3)
    else:
        d_results = run_sweep('d', D_SWEEP, fixed_d=15, fixed_m=3)
        m_results = run_sweep('m', M_SWEEP, fixed_d=15, fixed_m=3)

    out_d = {'N': N, 'hardware': hw, 'd_sweep': d_results}
    out_m = {'N': N, 'd': 15, 'hardware': hw, 'm_sweep': m_results}

    json.dump(out_d, open('data/results/h20_scaling_d.json', 'w'), indent=2, ensure_ascii=False)
    json.dump(out_m, open('data/results/h20_scaling_m.json', 'w'), indent=2, ensure_ascii=False)
    print(f'\nsaved -> data/results/h20_scaling_d.json / h20_scaling_m.json')


if __name__ == '__main__':
    main()
