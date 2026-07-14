"""
系统化 Benchmark (TODO #5)
统一对比所有实现版本，输出 JSON + 对比表，并校验 RMSE 与金标准一致性。

对比对象：
  1. v0  原始 PyTorch eager CPU
  2. v0  原始 PyTorch eager GPU
  3. v1  compile-friendly eager CPU
  4. v1  torch.compile CPU/GPU
  5. v2  CUDA Graph (mega-kernel)
另含参考算子库微基准 (cuBLAS gemm / cuSOLVER inv) 见 lib_reference_bench()。
"""
import os
import json
import time
import torch
import numpy as np

GOLDEN = dict(rmse_x_mm=1017.838305, rmse_y_mm=16.834549,
              rmse_z_mm=8.140964, rho_pct=0.175540)


def check_golden(m, tol=0.2):
    # tol in mm; v3 (Triton fp64, 不同累加顺序) 允许亚毫米级偏差
    ok = (abs(m['rmse_x_mm'] - GOLDEN['rmse_x_mm']) < tol and
          abs(m['rmse_y_mm'] - GOLDEN['rmse_y_mm']) < tol and
          abs(m['rmse_z_mm'] - GOLDEN['rmse_z_mm']) < tol)
    return ok


def lib_reference_bench(n_iter=2000):
    """参考: 单独测 15x15 gemm + 3x3 求解的 GPU 算子库吞吐 (理论上限参照)."""
    if not torch.cuda.is_available():
        return {}
    dev = 'cuda'
    P = torch.randn(15, 15, dtype=torch.float64, device=dev)
    F = torch.eye(15, dtype=torch.float64, device=dev)
    S = torch.eye(3, dtype=torch.float64, device=dev) + 0.1
    H = torch.randn(3, 15, dtype=torch.float64, device=dev)
    torch.cuda.synchronize()
    # gemm F@P@F'
    t0 = time.time()
    for _ in range(n_iter):
        _ = F @ P @ F.T
    torch.cuda.synchronize()
    gemm_us = (time.time() - t0) / n_iter * 1e6
    # cuSOLVER inv 3x3
    t0 = time.time()
    for _ in range(n_iter):
        _ = torch.linalg.inv(S)
    torch.cuda.synchronize()
    inv_us = (time.time() - t0) / n_iter * 1e6
    return {'gemm_FPFt_us': gemm_us, 'cusolver_inv3_us': inv_us,
            'note': '纯算子库单步耗时 (us), 反映 kernel-launch-bound 下界'}


def main(quick=False):
    _here = os.path.dirname(os.path.abspath(__file__))
    import sys
    if _here not in sys.path:
        sys.path.insert(0, _here)
    os.chdir(_here.rsplit('/opt', 1)[0])
    from ekf_baseline import run_ekf_baseline
    from ekf_v1 import run_ekf_v1
    has_cuda = torch.cuda.is_available()

    N_full = None  # 全量
    N_gpu = 20001  # eager GPU 太慢(~400s全量), 限步; v2/v3 用全量
    results = []

    def rec(label, m):
        m = dict(m); m['label'] = label
        partial = (m['n_steps'] < 160000)
        m['partial_run'] = partial
        m['golden_match'] = (None if partial else check_golden(m))
        results.append(m)
        if partial:
            flag = 'PART'
        else:
            flag = 'OK ' if m['golden_match'] else 'DIFF'
        print(f"[{flag}] {label:34s} {m['throughput_steps_per_s']:7.0f} steps/s "
              f"({m['elapsed_s']:6.2f}s, n={m['n_steps']})  "
              f"RMSE Y={m['rmse_y_mm']:.3f} Z={m['rmse_z_mm']:.3f}")

    print("=" * 80)
    print("EKF 系统化 Benchmark")
    print("=" * 80)

    # v0 baseline
    _, _, _, _, m = run_ekf_baseline(device='cpu', n_steps=N_full, verbose=False)
    rec('v0 eager CPU (golden)', m)

    # v1 eager
    _, _, _, _, m = run_ekf_v1(device='cpu', n_steps=N_full, compile_mode=None, verbose=False)
    rec('v1 eager CPU', m)

    if has_cuda:
        _, _, _, _, m = run_ekf_baseline(device='cuda', n_steps=N_gpu, verbose=False)
        rec('v0 eager GPU', m)
        _, _, _, _, m = run_ekf_v1(device='cuda', n_steps=N_gpu, compile_mode=None, verbose=False)
        rec('v1 eager GPU', m)
        _, _, _, _, m = run_ekf_v1(device='cuda', n_steps=N_gpu, compile_mode='default', verbose=False)
        rec('v1 compile(default) GPU', m)
        from ekf_v2 import run_ekf_v2
        _, _, _, _, m = run_ekf_v2(n_steps=N_full, unroll=10, verbose=False)
        rec('v2 CUDA Graph GPU', m)

        from ekf_v3 import run_ekf_v3
        _, _, _, _, m = run_ekf_v3(n_steps=N_full, verbose=False)
        rec('v3 Triton Mega fp64 GPU', m)

        from ekf_v4 import run_ekf_v4
        _, _, _, _, m = run_ekf_v4(n_steps=N_full, precision='fp32', verbose=False)
        rec('v4 Triton Mega fp32 GPU', m)
        _, _, _, _, m = run_ekf_v4(n_steps=N_full, precision='tf32', verbose=False)
        rec('v4 Triton Mega tf32 GPU', m)

        from ekf_v5 import run_ekf_v5
        for B in (78, 256):
            _, _, _, _, m = run_ekf_v5(n_steps=N_full, batch=B, precision='fp32', verbose=False)
            rec(f'v5 Batch x{B} fp32 GPU', m)

        # v6 mixed precision (贡献点2 深化)
        from ekf_v6_mixed import run_ekf_v6_mixed
        for combo in ['fp32-all', 'bf16-tolerant', 'bf16-moderate', 'tf32-moderate']:
            _, _, _, _, m = run_ekf_v6_mixed(n_steps=N_full, combo=combo, verbose=False)
            rec(f'v6 mixed {combo} GPU', m)

        # v7 concurrency (贡献点3 深化) — event-based timing, single-stream
        from ekf_v7_concurrency import run_ekf_v7_events
        for B in (1, 78, 128, 256, 312):
            _, _, _, _, m = run_ekf_v7_events(n_steps=N_full, batch=B, precision='fp32',
                                               n_streams=1, verbose=False)
            rec(f'v7 events B={B} fp32 GPU', m)

        libref = lib_reference_bench()
        print("\n[lib reference] 15x15 gemm:", f"{libref['gemm_FPFt_us']:.1f}us/step",
              "| cuSOLVER inv3:", f"{libref['cusolver_inv3_us']:.1f}us/step")
    else:
        libref = {}

    out = {'results': results, 'lib_reference': libref, 'golden': GOLDEN}
    json.dump(out, open('opt/results/benchmark_summary.json', 'w'), indent=2, ensure_ascii=False)
    print("\nsaved -> opt/results/benchmark_summary.json")

    # 加速比 (相对 v0 CPU full)
    base = next(r for r in results if r['label'].startswith('v0 eager CPU'))
    print("\n相对 v0 eager CPU 加速比:")
    for r in results:
        sp = r['throughput_steps_per_s'] / base['throughput_steps_per_s']
        print(f"  {r['label']:34s} {sp:5.2f}x")


if __name__ == '__main__':
    import sys
    main(quick='--quick' in sys.argv)
