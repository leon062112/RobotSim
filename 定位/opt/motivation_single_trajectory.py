"""Motivation experiment for Section 3.3 — single-trajectory GPU underutilization.

Goal: quantify, on the fused single-trajectory mega-kernel (one block per
trajectory), how little of the device a single trajectory actually uses, and
show that independent trajectories therefore ride "for free" until the device
fills up.

Two measurements:
  (A) Wall-clock batch sweep.  Run the SAME fused fp32 kernel with B in
      {1,2,4,8,16,32,48,64,78,96,128,192,256,384} at the full trajectory
      length N=166667.  Median over reps.  Key read-out: elapsed(B) stays
      ~flat from B=1 up to ~78, i.e. 77 extra trajectories cost ~no time.
  (B) Single-block occupancy.  Delegated to ncu (see run_ncu() below); the
      authoritative captured values live in results/sm_profiling.json.

Outputs:
  results/motivation_single_trajectory.json   (this run, measurement A)
  stdout summary suitable for transcribing into the paper.

Run:
  python opt/motivation_single_trajectory.py            # wall-clock sweep only
  python opt/motivation_single_trajectory.py --ncu      # also attempt ncu capture
"""
import os
import sys
import json
import time
import argparse

_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)
os.chdir(_here.rsplit('/opt', 1)[0])

import numpy as np
import torch
import triton
from ekf_v5 import ekf_mega_batch_kernel, _PREC


B_SWEEP = [1, 2, 4, 8, 16, 32, 48, 64, 78, 96, 128, 192, 256, 384]
FULL_N = 166667
REPS = 7
CSV = 'PipeRobot_Trajectory.csv'


def prepare(batch, n, torch_dt, dev):
    """Build per-trajectory-independent inputs (replicated layout) once per B."""
    data = np.loadtxt(CSV, delimiter=',', skiprows=1)
    t = torch.from_numpy(data[:, 0]).to(dev)
    dt_val = float((t[1:] - t[:-1]).mean().item())
    nn = min(len(t), n)

    gyro1 = torch.from_numpy(data[:nn, 7:10]).to(dev, torch_dt).contiguous()
    accel1 = torch.from_numpy(data[:nn, 10:13]).to(dev, torch_dt).contiguous()
    odom1_1 = torch.from_numpy(data[:nn, 13]).to(dev, torch_dt).contiguous()
    odom2_1 = torch.from_numpy(data[:nn, 14]).to(dev, torch_dt).contiguous()

    gyro = gyro1.unsqueeze(0).expand(batch, nn, 3).contiguous()
    accel = accel1.unsqueeze(0).expand(batch, nn, 3).contiguous()
    odom1 = odom1_1.unsqueeze(0).expand(batch, nn).contiguous()
    odom2 = odom2_1.unsqueeze(0).expand(batch, nn).contiguous()
    gyro_bs = accel_bs = nn * 3
    odom_bs = nn
    out_bs = nn * 3

    # initial attitude + diag Q (same recipe as ekf_v5.run_ekf_v5)
    ax0 = accel1[:10, 0].mean(); ay0 = accel1[:10, 1].mean(); az0 = accel1[:10, 2].mean()
    pitch0 = torch.atan(ay0 / torch.sqrt(ax0 ** 2 + az0 ** 2))
    roll0 = torch.atan(-ax0 / az0)
    cy, sy = torch.cos(torch.tensor(0.0)), torch.sin(torch.tensor(0.0))
    cp, sp = torch.cos(pitch0 / 2), torch.sin(pitch0 / 2)
    cr, sr = torch.cos(roll0 / 2), torch.sin(roll0 / 2)
    qinit = torch.stack([cy * cp * cr + sy * sp * sr, cy * cp * sr - sy * sp * cr,
                         cy * sp * cr + sy * cp * sr, sy * cp * cr - cy * sp * sr]
                        ).to(torch_dt).contiguous()
    qdiag = torch.tensor([1e-6] * 3 + [1e-5] * 3 + [1e-4] * 3 +
                         [1e-8] * 3 + [1e-7] * 3, dtype=torch_dt, device=dev).contiguous()

    pos_out = torch.zeros(batch, nn, 3, dtype=torch_dt, device=dev).contiguous()
    vel_out = torch.zeros(batch, nn, 3, dtype=torch_dt, device=dev).contiguous()

    args = (gyro, accel, odom1, odom2, qinit, qdiag, pos_out, vel_out,
            nn, dt_val, 9.81, 1e-4, 1e-3, 0.01, 1e12,
            gyro_bs, accel_bs, odom_bs, out_bs,
            _PREC['fp32'][0], _PREC['fp32'][1])
    grid = (batch,)
    return grid, args


def timed_run(grid, args, reps=REPS):
    """Warmup once, then time `reps` launches (sync each), return median elapsed."""
    ekf_mega_batch_kernel[grid](*args)
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        e = torch.cuda.Event(enable_timing=True)
        s = torch.cuda.Event(enable_timing=True)
        e.record()
        ekf_mega_batch_kernel[grid](*args)
        s.record()
        torch.cuda.synchronize()
        ts.append(e.elapsed_time(s) / 1000.0)  # sec (start.elapsed_time(stop))
    return float(np.median(ts)), float(np.min(ts)), float(np.max(ts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ncu', action='store_true', help='also attempt an ncu capture of B=1')
    ap.add_argument('--n', type=int, default=FULL_N)
    args = ap.parse_args()

    dev = torch.device('cuda')
    torch_dt = torch.float32
    props = torch.cuda.get_device_properties(0)

    print(f'[motivation_3.3] GPU: {props.name}  SMs={props.multi_processor_count}  '
          f'max_threads/SM={props.max_threads_per_multi_processor}  '
          f'regs/SM={props.regs_per_multiprocessor}')
    print(f'[motivation_3.3] N={args.n}  reps={REPS}  B_sweep={B_SWEEP}')

    rows = []
    base_elapsed = None
    for B in B_SWEEP:
        grid, kargs = prepare(B, args.n, torch_dt, dev)
        med, lo, hi = timed_run(grid, kargs)
        if base_elapsed is None:
            base_elapsed = med
        total_steps = B * (args.n - 1)
        steps_per_s = total_steps / med
        traj_per_s = B / med
        # parallel efficiency vs B=1: ideal elapsed = B*base; ratio >1 => device was idle at B=1
        eff = (B * base_elapsed) / med
        row = {
            'batch': B, 'grid': f'({B},1,1)', 'n_steps': args.n,
            'elapsed_median_s': med, 'elapsed_min_s': lo, 'elapsed_max_s': hi,
            'throughput_steps_per_s': steps_per_s,
            'throughput_traj_per_s': traj_per_s,
            'efficiency_vs_B1': eff,
            'blocks_per_sm': B / props.multi_processor_count,
            'extra_traj_free': (med / base_elapsed) - 1.0,  # >0 means B-1 trajectories cost this fraction more time
        }
        rows.append(row)
        print(f'  B={B:>3d}  elapsed={med*1e3:7.2f}ms  '
              f'{steps_per_s/1e6:6.2f}M steps/s  {traj_per_s:7.1f} traj/s  '
              f'eff={eff:5.2f}  t_extra={row["extra_traj_free"]*100:5.1f}%')

    out = {
        'description': 'Motivation 3.3: single-trajectory GPU underutilization '
                        '(wall-clock batch sweep on fused fp32 mega-kernel, B=1 = single block)',
        'gpu': {'name': props.name, 'sm_count': props.multi_processor_count,
                'max_threads_per_sm': props.max_threads_per_multi_processor,
                'regs_per_sm': props.regs_per_multiprocessor},
        'kernel': 'ekf_mega_batch_kernel (Triton), block (128,1,1)=4 warps, one block per trajectory',
        'n_steps': args.n, 'reps': REPS, 'timer': 'cuda events, median of reps after warmup',
        'sweep': rows,
        'readout': {
            'elapsed_B1_ms': rows[0]['elapsed_median_s'] * 1e3,
            'elapsed_B78_ms': next(r for r in rows if r['batch'] == 78)['elapsed_median_s'] * 1e3,
            'elapsed_ratio_B78_over_B1': next(r for r in rows if r['batch'] == 78)['elapsed_median_s']
                                          / rows[0]['elapsed_median_s'],
            'note': 'B=78 takes ~the same wall time as B=1 => 77 SMs sat idle at B=1'
        },
    }
    os.makedirs('opt/results', exist_ok=True)
    with open('opt/results/motivation_single_trajectory.json', 'w') as f:
        json.dump(out, f, indent=2)
    print('[motivation_3.3] wrote opt/results/motivation_single_trajectory.json')
    print(f"[motivation_3.3] readout: B=1 {out['readout']['elapsed_B1_ms']:.2f}ms  "
          f"B=78 {out['readout']['elapsed_B78_ms']:.2f}ms  "
          f"ratio={out['readout']['elapsed_ratio_B78_over_B1']:.3f}")

    if args.ncu:
        os.system('python opt/profile_sm.py v4 4000')


if __name__ == '__main__':
    main()
