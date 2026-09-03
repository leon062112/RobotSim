"""
A100 全量复跑 + baseline 补全（Plan: A100 重跑 + Baseline 补全实验）

相对 benchmark.py 的差异：
  1. 记录完整硬件元数据（GPU/SM/显存/驱动/CUDA/torch/triton）到 results['hardware']
  2. 每个配置 ≥N_REPEATS 次，报 mean±std
  3. 新增 batched 库微基准（batched cuBLAS gemm / batched cuSOLVER inv）
  4. 新增独立轨迹（蒙特卡洛）batch 输入列（run_ekf_v5_independent）
  5. batch 扫描扩展至 A100 饱和点（SM=108，B 到 512）

用法（从仓库根目录）：
  python code/optimization/benchmark_a100.py                 # 全量
  python code/optimization/benchmark_a100.py --quick         # 小 N 冒烟
  python code/optimization/benchmark_a100.py --skip-cpu-full # 跳过 3min 的 CPU 全量
"""
import os
import sys
import json
import time
import subprocess
import argparse
import numpy as np
import torch
import triton

_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)
os.chdir(os.path.abspath(os.path.join(_here, '..', '..')))

from ekf_baseline import run_ekf_baseline
from ekf_v1 import run_ekf_v1

GOLDEN = dict(rmse_x_mm=1017.838305, rmse_y_mm=16.834549,
              rmse_z_mm=8.140964, rho_pct=0.175540)

N_FULL = 166667
N_GPU_EAGER = 20001    # eager GPU 全量极慢；用 partial 标注
def b_sweep():
    """batch 扫描点，随 SM 数自适应：线性点 = SM、饱和点 ≈ SM×4。"""
    sm = torch.cuda.get_device_properties(0).multi_processor_count
    return sorted(set([1, 16, 32, 64, sm, 128, 256, sm * 4, sm * 4 + 64]))
N_REPEATS = 3


def get_hardware():
    props = torch.cuda.get_device_properties(0)
    driver = 'unknown'
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
            text=True, timeout=10).strip().splitlines()
        driver = out[0] if out else 'unknown'
    except Exception:
        pass
    return {
        'gpu_name': torch.cuda.get_device_name(0),
        'sm_count': props.multi_processor_count,
        'total_memory_gb': round(props.total_memory / 1e9, 1),
        'compute_capability': list(torch.cuda.get_device_capability(0)),
        'driver_version': driver,
        'cuda_version': torch.version.cuda,
        'torch_version': torch.__version__,
        'triton_version': triton.__version__,
    }


def check_golden(m, tol=0.2):
    return (abs(m['rmse_x_mm'] - GOLDEN['rmse_x_mm']) < tol and
            abs(m['rmse_y_mm'] - GOLDEN['rmse_y_mm']) < tol and
            abs(m['rmse_z_mm'] - GOLDEN['rmse_z_mm']) < tol)


def bench_repeat(fn, n=N_REPEATS, warmup=True, **kwargs):
    """fn(**kwargs) -> metrics dict；先 warmup 一次（JIT 编译），再重复 n 次计时。

    聚合吞吐 mean±std，其余字段取末次，另存 reps 列表以便核查冷/热轮。
    """
    if warmup:
        fn(**kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    outs = []
    for _ in range(n):
        m = fn(**kwargs)
        if isinstance(m, tuple):
            m = m[-1]
        outs.append(m)
    k = ('throughput_steps_per_s' if 'throughput_steps_per_s' in outs[0]
         else 'throughput_traj_per_s')
    vals = [o[k] for o in outs]
    agg = dict(outs[-1])
    agg[f'{k}_mean'] = float(np.mean(vals))
    agg[f'{k}_std'] = float(np.std(vals))
    agg[f'{k}_reps'] = [float(v) for v in vals]
    agg['n_repeats'] = n
    return agg


def lib_batched_bench(n_iter=2000):
    """batched 库微基准：剥离逐 op dispatch 后的 cuBLAS / cuSOLVER 代价。"""
    dev = 'cuda'
    out = {}
    torch.cuda.synchronize()

    S1 = torch.eye(3, dtype=torch.float64, device=dev) + 0.1
    t0 = time.time()
    for _ in range(n_iter):
        _ = torch.linalg.inv(S1)
    torch.cuda.synchronize()
    out['single_inv3_us'] = (time.time() - t0) / n_iter * 1e6

    F1 = torch.eye(15, dtype=torch.float64, device=dev)
    P1 = torch.randn(15, 15, dtype=torch.float64, device=dev)
    t0 = time.time()
    for _ in range(n_iter):
        _ = F1 @ P1
    torch.cuda.synchronize()
    out['single_gemm15_us'] = (time.time() - t0) / n_iter * 1e6

    for B in (78, 256):
        Sb = torch.eye(3, dtype=torch.float64, device=dev).repeat(B, 1, 1) + 0.1
        t0 = time.time()
        for _ in range(n_iter):
            _ = torch.linalg.inv(Sb)
        torch.cuda.synchronize()
        out[f'batched_inv3_us_per_op_B{B}'] = (time.time() - t0) / n_iter / B * 1e6

        Fb = torch.eye(15, dtype=torch.float64, device=dev).repeat(B, 1, 1)
        Pb = torch.randn(B, 15, 15, dtype=torch.float64, device=dev)
        t0 = time.time()
        for _ in range(n_iter):
            _ = Fb @ Pb
        torch.cuda.synchronize()
        out[f'batched_gemm15_us_per_op_B{B}'] = (time.time() - t0) / n_iter / B * 1e6

    out['note'] = ('single_* = 单次 launch 下界；batched_*_per_op = B 条批量化摊薄后的单条耗时。'
                   '单步 EKF ≈ 8 gemm + 1 inv。')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--skip-cpu-full', action='store_true')
    args = ap.parse_args()

    has_cuda = torch.cuda.is_available()
    hardware = get_hardware()
    print('=' * 84)
    print('A100 EKF Benchmark')
    print('=' * 84)
    print(json.dumps(hardware, indent=2))

    n_cpu = 20000 if args.quick else None
    n_gpu = 20000 if args.quick else N_FULL
    n_gpu_eager = 5000 if args.quick else N_GPU_EAGER

    results = {'hardware': hardware, 'golden': GOLDEN}

    # ---------- CPU 金标准基线 ----------
    if not args.skip_cpu_full:
        print('\n[CPU] v0 eager full (~3 min)...')
        m = run_ekf_baseline(device='cpu', n_steps=n_cpu, verbose=False)[-1]
        m['golden_match'] = (None if (n_cpu or 1) < 160000 else check_golden(m))
        m['n_repeats'] = 1
        results['v0_cpu'] = m
        print(f"  v0_cpu: {m['throughput_steps_per_s']:.1f} steps/s  "
              f"RMSE X={m['rmse_x_mm']:.3f}  golden_match={m['golden_match']}")
    else:
        results['v0_cpu'] = {'throughput_steps_per_s': 926.68,
                             'note': 'skip-cpu-full; 用本机实测默认值'}
        print('\n[CPU] skipped (--skip-cpu-full), base=926.68 steps/s')

    if not has_cuda:
        json.dump(results, open('data/results/a100_benchmark_summary.json', 'w'),
                  indent=2, ensure_ascii=False)
        print('no CUDA; saved & exit')
        return

    # ---------- GPU 单轨迹延迟轴 ----------
    print('\n[GPU] single-trajectory ladder...')
    from ekf_v2 import run_ekf_v2
    from ekf_v3 import run_ekf_v3
    from ekf_v4 import run_ekf_v4

    def wrap(fn):
        """把返回 tuple 的 run 函数包成返回 metrics dict。"""
        return lambda **kw: fn(**kw)[-1]

    ladder = []

    m = wrap(run_ekf_baseline)(device='cuda', n_steps=n_gpu_eager)
    m['partial_run'] = (m['n_steps'] < 160000); m['n_repeats'] = 1
    results['v0_gpu_eager'] = m; ladder.append('v0_gpu_eager')

    m = wrap(run_ekf_v1)(device='cuda', n_steps=n_gpu_eager, compile_mode=None)
    m['partial_run'] = (m['n_steps'] < 160000); m['n_repeats'] = 1
    results['v1_gpu_eager'] = m; ladder.append('v1_gpu_eager')

    m = wrap(run_ekf_v1)(device='cuda', n_steps=n_gpu_eager, compile_mode='default')
    m['partial_run'] = (m['n_steps'] < 160000); m['n_repeats'] = 1
    results['v1_gpu_compile'] = m; ladder.append('v1_gpu_compile')

    m = bench_repeat(wrap(run_ekf_v2), n=N_REPEATS, n_steps=n_gpu, unroll=10)
    results['v2_cudagraph'] = m; ladder.append('v2_cudagraph')

    m = bench_repeat(wrap(run_ekf_v3), n=N_REPEATS, n_steps=n_gpu)
    m['golden_match'] = check_golden(m)
    results['v3_triton_fp64'] = m; ladder.append('v3_triton_fp64')

    m = bench_repeat(wrap(run_ekf_v4), n=N_REPEATS, n_steps=n_gpu, precision='fp32')
    m['golden_match'] = check_golden(m)
    results['v4_triton_fp32'] = m; ladder.append('v4_triton_fp32')

    m = bench_repeat(wrap(run_ekf_v4), n=N_REPEATS, n_steps=n_gpu, precision='tf32')
    results['v4_triton_tf32'] = m; ladder.append('v4_triton_tf32')

    print('\n单轨迹阶梯（steps/s，mean over repeats）:')
    base = results['v0_cpu'].get('throughput_steps_per_s', 1)
    for k in ladder:
        mm = results[k]
        s = mm.get('throughput_steps_per_s_mean', mm.get('throughput_steps_per_s'))
        mm['speedup_vs_cpu'] = round(s / base, 2)
        print(f"  {k:18s} {s:12.1f}  speedup={mm['speedup_vs_cpu']:.2f}x")

    # ---------- GPU batch 吞吐轴（replicate 全扫描）----------
    print('\n[GPU] batch scaling (replicate)...')
    from ekf_v5 import run_ekf_v5
    batch_repl = {}
    for B in b_sweep():
        m = bench_repeat(wrap(run_ekf_v5), n=N_REPEATS, n_steps=n_gpu, batch=B, precision='fp32')
        batch_repl[str(B)] = m
        print(f"  B={B:4d}: {m['throughput_steps_per_s_mean']:12.1f} steps/s "
              f"({m['throughput_traj_per_s']:8.1f} traj/s)")
    results['batch_replicate'] = batch_repl

    # ---------- GPU batch 吞吐轴（独立轨迹，蒙特卡洛）----------
    npy = 'data/trajectory/traj_batch_fp64.npy'
    if os.path.exists(npy):
        print('\n[GPU] batch scaling (independent seeds)...')
        from ekf_v5 import run_ekf_v5_independent
        batch_indep = {}
        n_indep = int(np.load(npy, mmap_mode='r').shape[0])
        indep_B = [b for b in b_sweep() if b <= n_indep]
        for B in indep_B:
            m = bench_repeat(wrap(run_ekf_v5_independent), n=N_REPEATS,
                             traj_npy=npy, n_steps=n_gpu, batch=B, precision='fp32')
            batch_indep[str(B)] = m
            print(f"  B={B:4d}: {m['throughput_steps_per_s_mean']:12.1f} steps/s "
                  f"RMSE X={m['rmse_x_mm_mean']:.2f}±{m['rmse_x_mm_std']:.2f}")
        results['batch_independent'] = batch_indep
    else:
        print(f'\n[warn] {npy} not found, skip independent batch')

    # ---------- batched 库微基准 ----------
    print('\n[lib] batched cuBLAS / cuSOLVER microbench...')
    results['lib_batched'] = lib_batched_bench()
    for k, v in results['lib_batched'].items():
        if isinstance(v, float):
            print(f"  {k:36s} {v:10.4f} us")

    out_path = 'data/results/a100_benchmark_summary.json'
    json.dump(results, open(out_path, 'w'), indent=2, ensure_ascii=False)
    print(f'\nsaved -> {out_path}')


if __name__ == '__main__':
    main()