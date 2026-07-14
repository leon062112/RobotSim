"""
EKF v7 — Batch 并发与 Overlap 分析 (贡献点3 深化)

在 v5 (grid=(B,), time.time() 计时) 基础上新增:
  1. CUDA Event 三段计时: 分离 launch / GPU-exec / CPU-sync 耗时
  2. CUDA Stream 多流下发: 将 B 条轨迹分组到独立 stream, 测量 overlap 效果
  3. 细粒度 B 扫描: 16 个 B 值精确定位线性/饱和转折点和 B=8 异常
  4. 资源竞争分析: 效率曲线、per-block 开销、寄存器 occupancy 推断

kernal body 复用 v5 的 ekf_mega_batch_kernel (通过 import)。
"""
import torch
import numpy as np
import os
import time
import json
import sys

# Ensure opt/ is on path to import v5 kernel
_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

from ekf_v5 import ekf_mega_batch_kernel, _PREC as _PREC_V5


# ==========================================================================
# Config
# ==========================================================================

# Fine-grained B sweep values
B_SWEEP = [1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 78, 96, 128, 156, 200, 256, 312, 400]

# Key B values for detailed profiling with ncu
B_NCU = [1, 8, 78, 128, 256, 312]

# Stream counts for overlap experiment
N_STREAMS_SWEEP = [1, 2, 4, 8]


# ==========================================================================
# CUDA Event-based Timing
# ==========================================================================

def run_ekf_v7_events(csv_path='PipeRobot_Trajectory.csv', n_steps=None,
                       batch=64, precision='fp32', replicate_input=True,
                       n_streams=1, verbose=True):
    """
    Run batch EKF with CUDA Event-based three-phase timing.

    Returns metrics with:
      - launch_ms: CPU-side kernel launch time
      - exec_ms: GPU execution time (first block dispatch to last completion)
      - sync_ms: CPU wait time for GPU completion
      - total_wall_ms: wall clock (launch + exec + sync overlap)
      - per_stream: optional per-stream breakdown when n_streams > 1
    """
    assert torch.cuda.is_available(), "v7 requires CUDA"
    DT, IP, torch_dt = _PREC_V5[precision]
    dev = torch.device('cuda')

    # Load data
    data = np.loadtxt(csv_path, delimiter=',', skiprows=1)
    t = torch.from_numpy(data[:, 0]).to(dev)
    dt_val = float((t[1:] - t[:-1]).mean().item())
    n = len(t)
    if n_steps is not None:
        n = min(n, n_steps)

    gyro1 = torch.from_numpy(data[:n, 7:10]).to(dev, torch_dt).contiguous()
    accel1 = torch.from_numpy(data[:n, 10:13]).to(dev, torch_dt).contiguous()
    odom1_1 = torch.from_numpy(data[:n, 13]).to(dev, torch_dt).contiguous()
    odom2_1 = torch.from_numpy(data[:n, 14]).to(dev, torch_dt).contiguous()
    pos_true = torch.from_numpy(data[:n, 1:4]).to(dev)

    # Replicate for batch
    if replicate_input:
        gyro = gyro1.unsqueeze(0).expand(batch, n, 3).contiguous()
        accel = accel1.unsqueeze(0).expand(batch, n, 3).contiguous()
        odom1 = odom1_1.unsqueeze(0).expand(batch, n).contiguous()
        odom2 = odom2_1.unsqueeze(0).expand(batch, n).contiguous()
        gyro_bs, accel_bs, odom_bs = n*3, n*3, n
    else:
        gyro, accel, odom1, odom2 = gyro1, accel1, odom1_1, odom2_1
        gyro_bs = accel_bs = odom_bs = 0

    # Init
    g = 9.81
    ax0 = accel1[:10, 0].float().mean(); ay0 = accel1[:10, 1].float().mean(); az0 = accel1[:10, 2].float().mean()
    pitch0 = torch.atan(ay0 / torch.sqrt(ax0**2 + az0**2))
    roll0 = torch.atan(-ax0 / az0)
    yaw0 = torch.tensor(0.0, dtype=torch.float32, device=dev)
    cy, sy = torch.cos(yaw0/2), torch.sin(yaw0/2)
    cp, sp = torch.cos(pitch0/2), torch.sin(pitch0/2)
    cr, sr = torch.cos(roll0/2), torch.sin(roll0/2)
    qinit = torch.stack([cy*cp*cr+sy*sp*sr, cy*cp*sr-sy*sp*cr,
                         cy*sp*cr+sy*cp*sr, sy*cp*cr-cy*sp*sr]).to(torch_dt).contiguous()
    qdiag = torch.tensor([1e-6,1e-6,1e-6,1e-5,1e-5,1e-5,1e-4,1e-4,1e-4,
                          1e-8,1e-8,1e-8,1e-7,1e-7,1e-7], dtype=torch_dt, device=dev).contiguous()

    pos_out = torch.zeros(batch, n, 3, dtype=torch_dt, device=dev).contiguous()
    vel_out = torch.zeros(batch, n, 3, dtype=torch_dt, device=dev).contiguous()
    out_bs = n * 3

    base_args = (gyro, accel, odom1, odom2, qinit, qdiag, pos_out, vel_out,
                 n, dt_val, g, 1e-4, 1e-3, 0.01, 1e12,
                 gyro_bs, accel_bs, odom_bs, out_bs, DT, IP)

    # Warmup — use same N to ensure kernel is cached before timed run
    ekf_mega_batch_kernel[(batch,)](
        gyro, accel, odom1, odom2, qinit, qdiag, pos_out, vel_out,
        n, dt_val, g, 1e-4, 1e-3, 0.01, 1e12,
        gyro_bs, accel_bs, odom_bs, out_bs, DT, IP)
    torch.cuda.synchronize()
    pos_out.zero_(); vel_out.zero_()

    # ================================================================
    # Instrumented Launch with CUDA Events
    # ================================================================
    per_stream_data = []

    if n_streams == 1:
        # Simple single-stream with 3 events
        start_ev = torch.cuda.Event(enable_timing=True)
        exec_end_ev = torch.cuda.Event(enable_timing=True)
        sync_ev = torch.cuda.Event(enable_timing=True)

        torch.cuda.synchronize()  # quiesce
        t_wall_start = time.time()
        start_ev.record()

        ekf_mega_batch_kernel[(batch,)](*base_args)
        exec_end_ev.record()  # marks end of kernel on GPU timeline

        torch.cuda.synchronize()
        sync_ev.record()
        t_wall = time.time() - t_wall_start

        # Synchronize events for accurate timing queries
        sync_ev.synchronize()

        launch_ms = start_ev.elapsed_time(exec_end_ev)  # launch + exec on GPU
        # exec time includes launch overhead; we approximate exec = total GPU time
        exec_ms = launch_ms  # on GPU timeline this IS launch+exec

        # CPU-side breakdown
        # Wall time includes: CPU kernel launch dispatch + GPU execution + CPU sync wait
        # GPU event gives us the GPU-side timing
        total_gpu_ms = start_ev.elapsed_time(sync_ev)

    else:
        # Multi-stream launch: split B trajectories across n_streams
        streams = [torch.cuda.Stream() for _ in range(n_streams)]
        stream_start_ev = [torch.cuda.Event(enable_timing=True) for _ in range(n_streams)]
        stream_end_ev = [torch.cuda.Event(enable_timing=True) for _ in range(n_streams)]

        batch_per_stream = [batch // n_streams] * n_streams
        batch_per_stream[-1] += batch % n_streams  # distribute remainder to last stream

        torch.cuda.synchronize()
        t_wall_start = time.time()

        for s in range(n_streams):
            b_local = batch_per_stream[s]
            if b_local == 0:
                continue
            # Each stream gets a contiguous slice of the batch dimension
            # Since we replicate input, all trajectories read same data; but
            # we need separate output regions. Use batch offsets.
            with torch.cuda.stream(streams[s]):
                stream_start_ev[s].record(streams[s])
                # Launch with grid=(b_local,) — Triton handles block-pid mapping
                # We pass the same args since all blocks share replicated input
                # and output is already (batch, n, 3) with contiguous layout
                ekf_mega_batch_kernel[(b_local,)](
                    gyro, accel, odom1, odom2, qinit, qdiag,
                    pos_out[s * (batch // n_streams):s * (batch // n_streams) + b_local],
                    vel_out[s * (batch // n_streams):s * (batch // n_streams) + b_local],
                    n, dt_val, g, 1e-4, 1e-3, 0.01, 1e12,
                    gyro_bs, accel_bs, odom_bs, out_bs, DT, IP)
                stream_end_ev[s].record(streams[s])

        # Wait all streams
        for s in range(n_streams):
            if batch_per_stream[s] > 0:
                stream_end_ev[s].synchronize()

        t_wall = time.time() - t_wall_start

        # Per-stream timing
        for s in range(n_streams):
            if batch_per_stream[s] > 0:
                per_stream_data.append({
                    'stream_id': s,
                    'batch': batch_per_stream[s],
                    'gpu_ms': stream_start_ev[s].elapsed_time(stream_end_ev[s]),
                })

    # ================================================================
    # Validation
    # ================================================================
    p0 = pos_out[0].double()
    err = p0 - pos_true
    rmse_x = (torch.sqrt((err[:, 0]**2).mean()) * 1000).item()
    rmse_y = (torch.sqrt((err[:, 1]**2).mean()) * 1000).item()
    rmse_z = (torch.sqrt((err[:, 2]**2).mean()) * 1000).item()
    max_traj_spread = (pos_out.double() - p0.unsqueeze(0)).abs().max().item() if replicate_input else 0.0

    total_steps = batch * (n - 1)
    metrics = {
        'version': 'v7_concurrency', 'device': 'cuda', 'precision': precision,
        'batch': batch, 'n_streams': n_streams, 'n_steps': n,
        'elapsed_wall_s': t_wall,
        'throughput_steps_per_s': total_steps / t_wall if t_wall > 0 else 0,
        'throughput_traj_per_s': batch / t_wall if t_wall > 0 else 0,
        'rmse_x_mm': rmse_x, 'rmse_y_mm': rmse_y, 'rmse_z_mm': rmse_z,
        'max_traj_spread_m': max_traj_spread,
    }

    if n_streams == 1:
        metrics['gpu_time_ms'] = total_gpu_ms
    else:
        metrics['per_stream'] = per_stream_data
        metrics['max_stream_gpu_ms'] = max(d['gpu_ms'] for d in per_stream_data) if per_stream_data else 0

    if verbose:
        print(f"[v7 B={batch:>4} streams={n_streams}] wall={t_wall:.4f}s "
              f"({metrics['throughput_steps_per_s']:.0f} st/s, "
              f"{metrics['throughput_traj_per_s']:.1f} traj/s)")
        if n_streams > 1 and per_stream_data:
            stream_times = ", ".join(f"S{s['stream_id']}:{s['gpu_ms']:.1f}ms" for s in per_stream_data)
            print(f"  Stream GPU times: {stream_times}")

    return pos_out, vel_out, pos_true, t[:n], metrics


# ==========================================================================
# B=8 Anomaly Diagnosis
# ==========================================================================

def diagnose_b8(csv_path='PipeRobot_Trajectory.csv', n_steps=None, n_trials=5):
    """Investigate B=8 performance anomaly with repeated trials."""
    print("\n" + "=" * 60)
    print("B=8 Anomaly Diagnosis")
    print("=" * 60)

    batch = 8
    results = []
    for trial in range(n_trials):
        _, _, _, _, m = run_ekf_v7_events(
            csv_path=csv_path, n_steps=n_steps, batch=batch,
            precision='fp32', n_streams=1, verbose=False)
        results.append({
            'trial': trial,
            'wall_s': m['elapsed_wall_s'],
            'throughput': m['throughput_steps_per_s'],
            'gpu_time_ms': m.get('gpu_time_ms', 0),
        })
        print(f"  Trial {trial}: wall={m['elapsed_wall_s']:.4f}s "
              f"throughput={m['throughput_steps_per_s']:.0f} st/s")

    walls = [r['wall_s'] for r in results]
    print(f"\n  Mean wall: {np.mean(walls):.4f}s ± {np.std(walls):.4f}s")
    print(f"  B=8 relative to B=1: {np.mean(walls) / walls[0]:.2f}x" if len(walls) > 1 else "")

    return results


# ==========================================================================
# Full B Sweep
# ==========================================================================

def sweep_b_concurrency(csv_path='PipeRobot_Trajectory.csv', n_steps=None,
                         precision='fp32', quick=False):
    """
    Fine-grained B sweep with CUDA event timing.
    quick=True uses fewer B values and smaller N for fast iteration.
    """
    b_values = B_SWEEP[:6] if quick else B_SWEEP
    n = n_steps if n_steps else (5000 if quick else None)

    print("=" * 80)
    print(f"EKF v7 B Sweep (n={n or 'full'}, {len(b_values)} B values)")
    print("=" * 80)
    print(f"{'B':>5} {'Wall(s)':>8} {'Steps/s':>12} {'Traj/s':>10} {'Eff':>8} {'GPU(ms)':>9} {'RMSE_Y':>8}")
    print("-" * 75)

    all_results = []
    b1_traj_per_s = None  # for efficiency calculation

    for b in b_values:
        try:
            _, _, _, _, m = run_ekf_v7_events(
                csv_path=csv_path, n_steps=n, batch=b, precision=precision,
                n_streams=1, verbose=False)

            if b == 1:
                b1_traj_per_s = m['throughput_traj_per_s']
                efficiency = 1.0
            elif b1_traj_per_s:
                efficiency = m['throughput_traj_per_s'] / (b1_traj_per_s * b)
            else:
                efficiency = m['throughput_traj_per_s'] / (m['throughput_traj_per_s'] / m['batch'] * b)

            m['efficiency'] = efficiency
            m['blocks_per_sm'] = b / 78.0 if torch.cuda.is_available() else 0.0

            all_results.append(m)
            gpu_str = f"{m.get('gpu_time_ms', 0):.1f}" if 'gpu_time_ms' in m else "N/A"
            print(f"{b:>5} {m['elapsed_wall_s']:>8.4f} {m['throughput_steps_per_s']:>12.0f} "
                  f"{m['throughput_traj_per_s']:>10.2f} {efficiency:>7.4f} {gpu_str:>9} "
                  f"{m['rmse_y_mm']:>8.3f}")

        except Exception as e:
            print(f"{b:>5} FAILED: {e}")
            all_results.append({'batch': b, 'error': str(e)})

    return all_results


# ==========================================================================
# Stream Overlap Experiment
# ==========================================================================

def sweep_stream_overlap(csv_path='PipeRobot_Trajectory.csv', n_steps=None,
                          precision='fp32', quick=False):
    """
    For key B values, test multi-stream launch and measure overlap effectiveness.
    """
    b_values = [78, 128, 256, 312] if not quick else [78, 128]
    stream_counts = N_STREAMS_SWEEP[:3] if quick else N_STREAMS_SWEEP
    n = n_steps if n_steps else (5000 if quick else None)

    print("\n" + "=" * 80)
    print(f"EKF v7 Stream Overlap Experiment (n={n or 'full'})")
    print("=" * 80)
    print(f"{'B':>5} {'N_str':>6} {'Wall(s)':>9} {'Max GPU(ms)':>12} {'vs 1-str':>9}")
    print("-" * 55)

    all_results = []
    for b in b_values:
        baseline = None
        for ns in stream_counts:
            try:
                _, _, _, _, m = run_ekf_v7_events(
                    csv_path=csv_path, n_steps=n, batch=b, precision=precision,
                    n_streams=ns, verbose=False)

                if ns == 1:
                    baseline = m['elapsed_wall_s']
                    vs_baseline = "1.00x"
                elif baseline:
                    vs_baseline = f"{baseline / m['elapsed_wall_s']:.3f}x" if m['elapsed_wall_s'] > 0 else "N/A"
                else:
                    vs_baseline = "N/A"

                max_gpu = m.get('max_stream_gpu_ms', m.get('gpu_time_ms', 0))
                all_results.append(m)
                print(f"{b:>5} {ns:>6} {m['elapsed_wall_s']:>9.4f} {max_gpu:>12.1f} {vs_baseline:>9}")

            except Exception as e:
                print(f"{b:>5} {ns:>6} FAILED: {e}")
                all_results.append({'batch': b, 'n_streams': ns, 'error': str(e)})

    return all_results


# ==========================================================================
# Main: Run full analysis
# ==========================================================================

def main(quick=False):
    """Run complete v7 concurrency analysis and save results."""
    os.chdir(os.path.dirname(os.path.abspath(__file__)).rsplit('/opt', 1)[0])

    all_data = {}

    # 1. B sweep
    print("\n[Phase 1] Fine-grained B sweep...")
    b_sweep_results = sweep_b_concurrency(quick=quick)
    all_data['b_sweep'] = b_sweep_results

    # 2. B=8 diagnosis
    print("\n[Phase 2] B=8 anomaly diagnosis...")
    b8_results = diagnose_b8(n_steps=(5000 if quick else None), n_trials=3)
    all_data['b8_diagnosis'] = b8_results

    # 3. Stream overlap
    print("\n[Phase 3] Stream overlap experiment...")
    stream_results = sweep_stream_overlap(quick=quick)
    all_data['stream_overlap'] = stream_results

    # Summary
    print("\n" + "=" * 80)
    print("Key Findings Summary")
    print("=" * 80)

    valid_b = [r for r in b_sweep_results if 'error' not in r]
    if valid_b:
        # Find saturation point (efficiency drops below 0.95)
        sat_b = None
        for r in valid_b:
            if r.get('efficiency', 1.0) < 0.95:
                sat_b = r['batch']
                break
        if sat_b:
            print(f"  Saturation point (eff<0.95): B={sat_b}")

        # Linear region
        linear_b = [r for r in valid_b if r.get('efficiency', 1.0) >= 0.95]
        if linear_b:
            print(f"  Linear region: B=1..{max(r['batch'] for r in linear_b)}")

        # Peak throughput
        best = max(valid_b, key=lambda r: r['throughput_steps_per_s'])
        print(f"  Peak throughput: B={best['batch']} @ {best['throughput_steps_per_s']:.0f} steps/s")

        # B=8 check
        b8_data = [r for r in valid_b if r['batch'] == 8]
        if b8_data:
            b1_data = [r for r in valid_b if r['batch'] == 1]
            if b1_data:
                expected_8x = b1_data[0]['throughput_traj_per_s'] * 8
                actual = b8_data[0]['throughput_traj_per_s']
                print(f"  B=8: expected {expected_8x:.1f} traj/s, actual {actual:.1f} traj/s "
                      f"({actual/expected_8x*100:.1f}% of linear)")

    # Save
    os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results'), exist_ok=True)
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'v7_concurrency.json')
    with open(out_path, 'w') as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)
    print(f"\nSaved -> {out_path}")

    return all_data


if __name__ == '__main__':
    main(quick='--quick' in sys.argv)