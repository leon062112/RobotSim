"""
H20 全量 (d,m) shape benchmark —— 5 条方法线（eager/compile/torch-kf/prefix/triton）。

同机同源的全量 (d,m) shape benchmark —— 5 条方法线（eager/compile/torch-kf/prefix/triton）。
输出文件名按当前 GPU 自动命名：A100 → a100_scaling_d/m.json，H20 → h20_scaling_d/m.json。

用法：python code/optimization/benchmark_h20_shape.py [--quick]
精度对拍（G2/G3）：python code/optimization/benchmark_h20_shape.py --accuracy [--N 166667]
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

# G2/G3 精度对拍配置点（全量 N=166667，fp64 CPU eager 为参考）
ACC_CONFIGS = [(6, 3), (15, 3), (30, 3), (15, 1)]


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


def run_accuracy_sweep(n_steps=166667):
    """G2/G3 精度对拍：(d,m) in ACC_CONFIGS，以 CPU eager fp64 为参考。

    对比 GPU 5 方法在 fp32 与 fp64（若支持）下的终态 |x - x_ref| 与 |P - P_ref| 的 max。
    """
    print(f"\n[G2/G3 accuracy sweep, N={n_steps}, reference = CPU eager fp64]")
    print(f"{'(d,m)':>8} {'method':>10} {'dtype':>6} {'max|Δx|':>12} {'max|ΔP|':>12} {'steps/s':>10}")

    rows = []
    for (d, m) in ACC_CONFIGS:
        torch.manual_seed(0)
        # 固定模型生成（与 build_model 逻辑一致，但支持 float64）
        F64 = torch.eye(d, dtype=torch.float64)
        H64 = torch.zeros(m, d, dtype=torch.float64)
        for i in range(m):
            H64[i, i % d] = 1.0
        Q64 = torch.eye(d, dtype=torch.float64) * 1e-6
        R64 = torch.eye(m, dtype=torch.float64) * 1e-3
        z64 = torch.randn(n_steps, m, dtype=torch.float64)

        # 1. CPU eager fp64 参考
        t0 = time.time()
        xref, Pref = run_eager(F64, H64, Q64, R64, z64, n_steps)
        s_ref = n_steps / (time.time() - t0)
        rows.append({'d': d, 'm': m, 'method': 'cpu_eager', 'dtype': 'fp64',
                     'max_dx': 0.0, 'max_dp': 0.0, 'steps_per_s': float(s_ref)})
        print(f"{(f'({d},{m})'):>8} {'cpu_eager':>10} {'fp64':>6} {0.0:12.2e} {0.0:12.2e} {s_ref:10.0f}")

        # GPU fp32 输入
        F32 = F64.float().cuda()
        H32 = H64.float().cuda()
        Q32 = Q64.float().cuda()
        R32 = R64.float().cuda()
        z32 = z64.float().cuda()

        def rec(name, dt_label, fn):
            torch.cuda.synchronize()
            t0 = time.time()
            x, P = fn()
            torch.cuda.synchronize()
            s = n_steps / (time.time() - t0)
            if hasattr(x, 'squeeze'):
                x = x.squeeze()
            if hasattr(P, 'squeeze'):
                P = P.squeeze()
            dx = float((x.double().cpu() - xref).abs().max().item())
            dp = float((P.double().cpu() - Pref).abs().max().item())
            row = {'d': d, 'm': m, 'method': name, 'dtype': dt_label,
                   'max_dx': dx, 'max_dp': dp, 'steps_per_s': float(s)}
            rows.append(row)
            print(f"{'':>8} {name:>10} {dt_label:>6} {dx:12.2e} {dp:12.2e} {s:10.0f}")
            return row

        # 2. eager fp32
        rec('eager', 'fp32', lambda: run_eager(F32, H32, Q32, R32, z32, n_steps))
        # 3. compile fp32
        rec('compile', 'fp32', lambda: run_compile(F32, H32, Q32, R32, z32, n_steps))
        # 4. torch-kf fp32
        rec('torch_kf', 'fp32', lambda: run_torchkf(F32, H32, Q32, R32, z32, n_steps))
        # 5. prefix fp32
        rec('prefix', 'fp32', lambda: run_prefix(F32, H32, Q32, R32, z32, n_steps, dtype=torch.float32))
        # 6. prefix fp64 (Särkkä 支持原生 fp64)
        rec('prefix', 'fp64', lambda: run_prefix(F64.cuda(), H64.cuda(), Q64.cuda(), R64.cuda(),
                                                 z64.cuda(), n_steps, dtype=torch.float64))
        # 7. triton fp32 (ieee dot —— 精度对拍口径)
        rec('triton', 'fp32', lambda: run_triton(F32, H32, Q32, R32, z32, n_steps,
                                                 warmup=True, full=True, ip='ieee'))
        # 8. triton fp32 + tf32 dot (历史默认路径, 对照 IP 影响)
        rec('triton_tf32', 'fp32', lambda: run_triton(F32, H32, Q32, R32, z32, n_steps,
                                                      warmup=True, full=True, ip='tf32'))

    return rows


def main():
    global N
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--accuracy', action='store_true', help='运行 G2/G3 精度对拍模式')
    ap.add_argument('--N', type=int, default=None, help='覆盖 N (默认 accuracy=166667, perf=2000)')
    args = ap.parse_args()

    hw = {'gpu': torch.cuda.get_device_name(0),
          'sm': torch.cuda.get_device_properties(0).multi_processor_count}
    gpu = hw['gpu'].lower()
    tag = 'h20' if 'h20' in gpu else ('a100' if 'a100' in gpu else 'gpu')

    if args.accuracy:
        n_steps = args.N if args.N is not None else 166667
        rows = run_accuracy_sweep(n_steps=n_steps)
        out = {'hardware': hw, 'N': n_steps, 'accuracy_results': rows}
        out_path = f'data/results/linear_accuracy_{tag}.json'
        json.dump(out, open(out_path, 'w'), indent=2, ensure_ascii=False)
        print(f'\nsaved -> {out_path}')
        return

    n_perf = args.N if args.N is not None else N
    N = n_perf
    if args.quick:
        d_results = run_sweep('d', [15], fixed_d=15, fixed_m=3)
        m_results = run_sweep('m', [3], fixed_d=15, fixed_m=3)
    else:
        d_results = run_sweep('d', D_SWEEP, fixed_d=15, fixed_m=3)
        m_results = run_sweep('m', M_SWEEP, fixed_d=15, fixed_m=3)

    out_d = {'N': N, 'hardware': hw, 'd_sweep': d_results}
    out_m = {'N': N, 'd': 15, 'hardware': hw, 'm_sweep': m_results}

    json.dump(out_d, open(f'data/results/{tag}_scaling_d.json', 'w'), indent=2, ensure_ascii=False)
    json.dump(out_m, open(f'data/results/{tag}_scaling_m.json', 'w'), indent=2, ensure_ascii=False)
    print(f'\nsaved -> data/results/{tag}_scaling_d.json / {tag}_scaling_m.json')


if __name__ == '__main__':
    main()
