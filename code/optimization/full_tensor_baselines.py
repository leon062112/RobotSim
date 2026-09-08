"""
G1 (P0): eager / torch.compile 全量 fp64 —— timing + RMSE + Max Δ vs CPU golden。

补齐论文 §5.3 Table (precision-fidelity) 缺失的 torch eager / torch.compile 两行。
全量 166,667 步、fp64、3 reps；Max Δ 为逐时刻逐轴 |pos - pos_ref| 的最大值 (mm)。

产出: data/results/full_tensor_baselines_{tag}.json  （不覆盖任何归档文件）
用法: python code/optimization/full_tensor_baselines.py
"""
import json
import os
import sys
import numpy as np
import torch

_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)
os.chdir(os.path.abspath(os.path.join(_here, '..', '..')))

from ekf_baseline import run_ekf_baseline          # v0 eager
from ekf_v1 import run_ekf_v1                      # v1 (compile-friendly) / torch.compile
from benchmark_a100 import get_hardware, check_golden, GOLDEN

CSV = 'data/trajectory/PipeRobot_Trajectory.csv'    # 全量 166,667 步
N_REPEATS = 3


def tag_of():
    gpu = torch.cuda.get_device_name(0).lower()
    return 'a100' if 'a100' in gpu else ('h20' if 'h20' in gpu else 'gpu')


def run_rep(fn, pos_ref, **kwargs):
    """跑 1 次全量，返回 metrics + Max Δ (mm, 逐时刻逐轴 |pos - pos_ref| max)。"""
    pos, vel, pos_true, t, m = fn(**kwargs)
    m = dict(m)
    if pos_ref is not None:
        m['max_delta_mm'] = (pos.double().cpu() - pos_ref).abs().max().item() * 1000
    else:
        m['max_delta_mm'] = None
    return m


def bench(fn, pos_ref, n=N_REPEATS, **kwargs):
    """重复 n 次计时，吞吐报 mean±std，精度字段取末次（确定性数值栈，rep 间一致）。"""
    outs = [run_rep(fn, pos_ref, **kwargs) for _ in range(n)]
    k = 'throughput_steps_per_s'
    vals = [o[k] for o in outs]
    agg = dict(outs[-1])
    agg[f'{k}_mean'] = float(np.mean(vals))
    agg[f'{k}_std'] = float(np.std(vals))
    agg[f'{k}_reps'] = [float(v) for v in vals]
    agg['n_repeats'] = n
    agg['golden_match'] = check_golden(agg)
    return agg


def main():
    tag = tag_of()
    out = {'hardware': get_hardware(), 'golden': GOLDEN, 'n_steps': 166667}
    print(json.dumps(out['hardware'], indent=2))

    # ---- CPU fp64 参考（数值金标准 + 本机 CPU 吞吐行；约 2-3 min）----
    print('\n[1/3] CPU fp64 reference full run ...')
    pos_ref, _, _, _, m_cpu = run_ekf_baseline(CSV, device='cpu', verbose=True)
    m_cpu = dict(m_cpu)
    m_cpu['golden_match'] = check_golden(m_cpu)
    m_cpu['max_delta_mm'] = 0.0
    out['cpu_fp64_reference'] = m_cpu
    pos_ref = pos_ref.double()
    print(f"  cpu: {m_cpu['throughput_steps_per_s']:.1f} steps/s "
          f"RMSE X={m_cpu['rmse_x_mm']:.6f} golden_match={m_cpu['golden_match']}")

    # ---- v0 eager GPU fp64 全量 ×3（约 11 min/run）----
    print('\n[2/3] torch eager GPU fp64 full x3 ...')
    out['v0_gpu_eager_full'] = bench(run_ekf_baseline, pos_ref, device='cuda')
    m = out['v0_gpu_eager_full']
    print(f"  eager: {m['throughput_steps_per_s_mean']:.1f}±{m['throughput_steps_per_s_std']:.1f} steps/s "
          f"RMSE X={m['rmse_x_mm']:.6f} MaxΔ={m['max_delta_mm']:.6f}mm golden_match={m['golden_match']}")

    # ---- torch.compile GPU fp64 全量 ×3（每次调用内含 Inductor 编译，约 1.5 min/run）----
    print('\n[3/3] torch.compile GPU fp64 full x3 ...')
    out['v1_gpu_compile_full'] = bench(run_ekf_v1, pos_ref, device='cuda', compile_mode='default')
    m = out['v1_gpu_compile_full']
    print(f"  compile: {m['throughput_steps_per_s_mean']:.1f}±{m['throughput_steps_per_s_std']:.1f} steps/s "
          f"RMSE X={m['rmse_x_mm']:.6f} MaxΔ={m['max_delta_mm']:.6f}mm golden_match={m['golden_match']}")

    path = f'data/results/full_tensor_baselines_{tag}.json'
    json.dump(out, open(path, 'w'), indent=2, ensure_ascii=False)
    print(f'\nsaved -> {path}')

    # 验收：golden_match 全 True；吞吐 std/mean < 5%
    ok = all(out[k]['golden_match'] for k in ('cpu_fp64_reference', 'v0_gpu_eager_full', 'v1_gpu_compile_full'))
    for k in ('v0_gpu_eager_full', 'v1_gpu_compile_full'):
        r = out[k]['throughput_steps_per_s_std'] / out[k]['throughput_steps_per_s_mean']
        print(f'  {k}: std/mean = {r*100:.2f}% (要求 <5%)')
        ok = ok and r < 0.05
    print('ACCEPT' if ok else 'REJECT — 检查 golden_match / 计时方差')


if __name__ == '__main__':
    main()
