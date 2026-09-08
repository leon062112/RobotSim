"""
Task4: 在本机 A100 复跑 Trident 单轨迹精度配置 —— 保证论文 §5.3 表单一来源。

配置: v3 fp64 (Triton fused fp64) / v4 fp32 (adopted) / v4 tf32 (对照)。
全量 166,667 步，3 reps，吞吐 mean±std；Max Δ = 逐时刻逐轴 |pos - CPU fp64 ref| max (mm)。

产出: data/results/trident_precision_{tag}.json（不覆盖归档）；参考轨迹缓存到
data/trajectory/ref_fp64_pos.npy 以便复用。
用法: python code/optimization/trident_precision_a100.py
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

from ekf_baseline import run_ekf_baseline
from ekf_v3 import run_ekf_v3
from ekf_v4 import run_ekf_v4
from benchmark_a100 import get_hardware, check_golden, GOLDEN

CSV = 'data/trajectory/PipeRobot_Trajectory.csv'
REF_NPY = 'data/trajectory/ref_fp64_pos.npy'
N_REPEATS = 3


def tag_of():
    gpu = torch.cuda.get_device_name(0).lower()
    return 'a100' if 'a100' in gpu else ('h20' if 'h20' in gpu else 'gpu')


def bench(fn, pos_ref, **kwargs):
    outs = []
    for _ in range(N_REPEATS):
        pos, vel, pos_true, t, m = fn(**kwargs)
        m = dict(m)
        m['max_delta_mm'] = (pos.double().cpu() - pos_ref).abs().max().item() * 1000
        outs.append(m)
    k = 'throughput_steps_per_s'
    vals = [o[k] for o in outs]
    agg = dict(outs[-1])
    agg[f'{k}_mean'] = float(np.mean(vals))
    agg[f'{k}_std'] = float(np.std(vals))
    agg[f'{k}_reps'] = [float(v) for v in vals]
    agg['n_repeats'] = N_REPEATS
    agg['golden_match'] = check_golden(agg)
    return agg


def main():
    tag = tag_of()
    out = {'hardware': get_hardware(), 'golden': GOLDEN, 'n_steps': 166667}
    print(json.dumps(out['hardware'], indent=2))

    # CPU fp64 参考轨迹（若无缓存则跑一遍，约 3 min）
    if os.path.exists(REF_NPY):
        pos_ref = torch.from_numpy(np.load(REF_NPY))
        print(f'reference loaded from {REF_NPY}')
    else:
        print('\n[ref] CPU fp64 full run (~3 min) ...')
        pos, _, _, _, m_cpu = run_ekf_baseline(CSV, device='cpu', verbose=True)
        pos_ref = pos.double()
        np.save(REF_NPY, pos_ref.numpy())
        print(f'reference saved -> {REF_NPY}  (RMSE X={m_cpu["rmse_x_mm"]:.6f})')

    for label, fn, kw in [
        ('trident_fp64', run_ekf_v3, {}),
        ('trident_fp32', run_ekf_v4, {'precision': 'fp32'}),
        ('trident_tf32', run_ekf_v4, {'precision': 'tf32'}),
    ]:
        print(f'\n[{label}] x{N_REPEATS} ...')
        m = bench(fn, pos_ref, **kw)
        out[label] = m
        print(f"  {m['throughput_steps_per_s_mean']:.1f}±{m['throughput_steps_per_s_std']:.1f} steps/s  "
              f"RMSE X/Y/Z = {m['rmse_x_mm']:.3f}/{m['rmse_y_mm']:.3f}/{m['rmse_z_mm']:.3f} mm  "
              f"MaxΔ={m['max_delta_mm']:.4f} mm  golden_match={m['golden_match']}")

    path = f'data/results/trident_precision_{tag}.json'
    json.dump(out, open(path, 'w'), indent=2, ensure_ascii=False)
    print(f'\nsaved -> {path}')


if __name__ == '__main__':
    main()
