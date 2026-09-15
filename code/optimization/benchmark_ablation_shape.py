"""
消融图数据: 全因子 2^3 组合 (fusion x precision x batch), 端到端 SINS/EKF 工作负载, A100。

八条线 (吞吐 = filter steps/s):
  none              v1 eager fp64, B=1           —— 三个贡献点都不加 (baseline)
  precision_only    v1 eager fp32, B=1           —— 只加精度 (component-aware 所选 fp32 计划)
  batch_only        v1 eager fp64, B=108 批量    —— 只加 batch (不融合, 逐步 launch)
  precision_batch   v1 eager fp32, B=108 批量    —— 精度 + batch (不融合)
  fusion_only       v3 Triton fused fp64, B=1
  fusion_precision  v4 Triton fused fp32, B=1
  fusion_batch      v5 Triton fused fp64, B=108
  all               v5 Triton fused fp32, B=108  —— 三个全加 (TRIDENT)

launch-bound 的 eager 变体吞吐与 N 无关, 只测小 N; fused 变体测到全程 166,667。

用法:
  python code/optimization/benchmark_ablation_shape.py            # 全部 8 组合 (覆盖 JSON)
  python code/optimization/benchmark_ablation_shape.py fusion_batch precision_batch
                                                      # 只跑指定组合, 结果合并进已有 JSON
"""
import os
import sys
import json
import time
import numpy as np
import torch

_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)
os.chdir(os.path.abspath(os.path.join(_here, '..', '..')))

from ekf_v1 import run_ekf_v1
from ekf_v3 import run_ekf_v3
from ekf_v4 import run_ekf_v4
from ekf_v5 import run_ekf_v5

CSV = 'data/trajectory/PipeRobot_Trajectory.csv'
BATCH = 108                      # A100 SM 数, 每 SM 一条轨迹
N_FAST = [2000, 5000, 20000, 50000, 166667]   # fused 变体
N_SLOW = [2000, 5000, 20000]                  # eager 变体 (吞吐与 N 无关)
REPS_FAST, REPS_SLOW = 5, 3
OUT = 'data/results/a100_ablation_shape.json'


# ---------------------------------------------------------------------------
# batched eager: ekf_v1.ekf_step 的逐行批量移植 (leading batch 维, 不融合)
# ---------------------------------------------------------------------------
def ekf_step_batched(pos_prev, vel_prev, q_prev, P, gyro, accel, odom1_k, odom2_k,
                     dt, g_vec, I15, I3, I4, Q_noise, R_odo, R_vcon, delta_thresh):
    B = pos_prev.shape[0]
    dev, dtp = pos_prev.device, pos_prev.dtype
    z0 = torch.zeros(B, dtype=dtp, device=dev)

    theta_x, theta_y, theta_z = gyro[:, 0] * dt, gyro[:, 1] * dt, gyro[:, 2] * dt
    theta_norm = torch.sqrt(theta_x**2 + theta_y**2 + theta_z**2)
    theta_mat = torch.stack([
        torch.stack([z0, -theta_x, -theta_y, -theta_z], 1),
        torch.stack([-theta_x, z0, -theta_z, -theta_y], 1),
        torch.stack([-theta_y, -theta_z, z0, -theta_x], 1),
        torch.stack([-theta_z, -theta_y, -theta_x, z0], 1),
    ], 1)  # [B,4,4]

    small = theta_norm > 1e-10
    denom = torch.where(small, theta_norm, torch.ones_like(theta_norm))
    coef = torch.where(small, torch.sin(theta_norm / 2) / denom,
                       0.5 * torch.ones_like(theta_norm))
    q_update = torch.cos(theta_norm / 2)[:, None, None] * I4 + coef[:, None, None] * theta_mat

    q_k = (q_update @ q_prev.unsqueeze(-1)).squeeze(-1)
    q_k = q_k / q_k.norm(dim=1, keepdim=True)

    q0, q1, q2, q3 = q_k[:, 0], q_k[:, 1], q_k[:, 2], q_k[:, 3]
    Cnb = torch.stack([
        torch.stack([q0**2+q1**2-q2**2-q3**2, 2*(q1*q2+q0*q3), 2*(q1*q3-q0*q2)], 1),
        torch.stack([2*(q1*q2-q0*q3), q0**2-q1**2+q2**2-q3**2, 2*(q2*q3+q0*q1)], 1),
        torch.stack([2*(q1*q3+q0*q2), 2*(q2*q3-q0*q1), q0**2-q1**2-q2**2+q3**2], 1),
    ], 1)  # [B,3,3]

    f_b = accel
    vel_k = vel_prev + (Cnb @ (f_b - g_vec).unsqueeze(-1)).squeeze(-1) * dt
    pos_k = pos_prev + vel_prev * dt

    delta_D = (odom1_k + odom2_k) / 2 * dt
    delta_S = (pos_k - pos_prev).norm(dim=1)

    psi = torch.atan2(Cnb[:, 0, 1], Cnb[:, 0, 0])
    spsi, cpsi = torch.sin(psi), torch.cos(psi)

    h_row0 = torch.cat([torch.zeros(B, 3, dtype=dtp, device=dev), Cnb[:, 0, :],
                        torch.zeros(B, 9, dtype=dtp, device=dev)], 1)
    h_row1 = torch.cat([torch.stack([-spsi, cpsi, z0], 1),
                        torch.zeros(B, 12, dtype=dtp, device=dev)], 1)
    h_row2 = torch.cat([torch.stack([z0, z0, z0 + 1.0], 1),
                        torch.zeros(B, 12, dtype=dtp, device=dev)], 1)
    H = torch.stack([h_row0, h_row1, h_row2], 1)  # [B,3,15]

    z = torch.stack([delta_S - delta_D, vel_k[:, 1], vel_k[:, 2]], 1)

    normal = torch.abs(delta_D - delta_S) < delta_thresh
    r_odo_eff = torch.where(normal, R_odo, R_odo * 0 + 1e12)
    R = torch.diag_embed(torch.stack(
        [r_odo_eff, R_vcon.expand(B), R_vcon.expand(B)], 1))  # [B,3,3]

    f_n = (Cnb @ f_b.unsqueeze(-1)).squeeze(-1)
    sk_fn = torch.stack([
        torch.stack([z0, -f_n[:, 2], f_n[:, 1]], 1),
        torch.stack([f_n[:, 2], z0, -f_n[:, 0]], 1),
        torch.stack([-f_n[:, 1], f_n[:, 0], z0], 1)], 1)
    sk_w = torch.stack([
        torch.stack([z0, -gyro[:, 2], gyro[:, 1]], 1),
        torch.stack([gyro[:, 2], z0, -gyro[:, 0]], 1),
        torch.stack([-gyro[:, 1], gyro[:, 0], z0], 1)], 1)
    F = I15.expand(B, 15, 15).clone()
    F[:, 0:3, 3:6] = I3 * dt
    F[:, 3:6, 6:9] = -sk_fn * dt
    F[:, 3:6, 12:15] = -Cnb * dt
    F[:, 6:9, 6:9] = -sk_w * dt
    F[:, 6:9, 9:12] = -I3 * dt

    P_pred = F @ P @ F.transpose(-1, -2) + Q_noise
    S = H @ P_pred @ H.transpose(-1, -2) + R
    K = P_pred @ H.transpose(-1, -2) @ torch.linalg.inv(S)
    x_ekf = (K @ z.unsqueeze(-1)).squeeze(-1)
    P_new = (I15 - K @ H) @ P_pred

    pos_k = pos_k - x_ekf[:, 0:3]
    vel_k = vel_k - x_ekf[:, 3:6]
    phi = x_ekf[:, 6:9]
    dq = torch.stack([z0 + 1.0, 0.5 * phi[:, 0], 0.5 * phi[:, 1], 0.5 * phi[:, 2]], 1)
    dq = dq / dq.norm(dim=1, keepdim=True)
    a, b = q_k, dq
    q_k = torch.stack([
        a[:, 0]*b[:, 0] - a[:, 1]*b[:, 1] - a[:, 2]*b[:, 2] - a[:, 3]*b[:, 3],
        a[:, 0]*b[:, 1] + a[:, 1]*b[:, 0] + a[:, 2]*b[:, 3] - a[:, 3]*b[:, 2],
        a[:, 0]*b[:, 2] - a[:, 1]*b[:, 3] + a[:, 2]*b[:, 0] + a[:, 3]*b[:, 1],
        a[:, 0]*b[:, 3] + a[:, 1]*b[:, 2] - a[:, 2]*b[:, 1] + a[:, 3]*b[:, 0]], 1)
    q_k = q_k / q_k.norm(dim=1, keepdim=True)
    return pos_k, vel_k, q_k, P_new


def run_eager_batched(n_steps, batch=BATCH, precision='fp64'):
    """B 条相同轨迹的批量 eager (逐步 launch, 不融合)。吞吐按 batch*(n-1)/wall 计。"""
    dev = torch.device('cuda')
    dtp = torch.float64 if precision == 'fp64' else torch.float32
    data = np.loadtxt(CSV, delimiter=',', skiprows=1)
    t = torch.from_numpy(data[:, 0]).to(dev)
    dt = (t[1:] - t[:-1]).mean().item()
    n = min(len(t), n_steps) if n_steps is not None else len(t)

    gyro = torch.from_numpy(data[:, 7:10]).to(dev, dtp)
    accel = torch.from_numpy(data[:, 10:13]).to(dev, dtp)
    odom1 = torch.from_numpy(data[:, 13]).to(dev, dtp)
    odom2 = torch.from_numpy(data[:, 14]).to(dev, dtp)

    g_vec = torch.tensor([0, 0, 9.81], dtype=dtp, device=dev)
    I15 = torch.eye(15, dtype=dtp, device=dev)
    I3 = torch.eye(3, dtype=dtp, device=dev)
    I4 = torch.eye(4, dtype=dtp, device=dev)
    Q_noise = torch.diag(torch.tensor([
        1e-6, 1e-6, 1e-6, 1e-5, 1e-5, 1e-5, 1e-4, 1e-4, 1e-4,
        1e-8, 1e-8, 1e-8, 1e-7, 1e-7, 1e-7], dtype=dtp, device=dev))
    R_odo = torch.tensor(1e-4, dtype=dtp, device=dev)
    R_vcon = torch.tensor(1e-3, dtype=dtp, device=dev)
    delta_thresh = torch.tensor(0.01, dtype=dtp, device=dev)
    dt_t = torch.tensor(dt, dtype=dtp, device=dev)

    # 初始对准 (与 v1 相同)
    ax0, ay0, az0 = accel[:10, 0].mean(), accel[:10, 1].mean(), accel[:10, 2].mean()
    pitch0 = torch.atan(ay0 / torch.sqrt(ax0**2 + az0**2))
    roll0 = torch.atan(-ax0 / az0)
    yaw0 = torch.tensor(0.0, dtype=dtp, device=dev)
    cy, sy = torch.cos(yaw0/2), torch.sin(yaw0/2)
    cp, sp = torch.cos(pitch0/2), torch.sin(pitch0/2)
    cr, sr = torch.cos(roll0/2), torch.sin(roll0/2)
    q0 = torch.stack([cy*cp*cr + sy*sp*sr, cy*cp*sr - sy*sp*cr,
                      cy*sp*cr + sy*cp*sr, sy*cp*cr - cy*sp*sr])

    pos = torch.zeros(batch, 3, dtype=dtp, device=dev)
    vel = torch.zeros(batch, 3, dtype=dtp, device=dev)
    q = q0.unsqueeze(0).expand(batch, 4).contiguous()
    P = (torch.eye(15, dtype=dtp, device=dev) * 0.1).unsqueeze(0).expand(batch, 15, 15).contiguous()

    gyro_b = gyro.unsqueeze(0).expand(batch, -1, 3).contiguous()
    accel_b = accel.unsqueeze(0).expand(batch, -1, 3).contiguous()
    odom1_b = odom1.unsqueeze(0).expand(batch, -1).contiguous()
    odom2_b = odom2.unsqueeze(0).expand(batch, -1).contiguous()

    torch.cuda.synchronize()
    t0 = time.time()
    for k in range(1, n):
        pos, vel, q, P = ekf_step_batched(
            pos, vel, q, P, gyro_b[:, k-1], accel_b[:, k-1],
            odom1_b[:, k], odom2_b[:, k],
            dt_t, g_vec, I15, I3, I4, Q_noise, R_odo, R_vcon, delta_thresh)
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    return {'throughput_steps_per_s': batch * (n - 1) / elapsed,
            'elapsed_s': elapsed, 'pos': pos}


def validate_batched_eager():
    """B=4 条相同轨迹, N=2000: 第 0 条应与 v1 单条 eager fp64 逐位接近。"""
    m = run_eager_batched(2000, batch=4)
    pos_v1 = run_ekf_v1(device='cuda', n_steps=2000, verbose=False)[0]
    diff = (m['pos'][0].cpu().double() - pos_v1[-1].cpu().double()).abs().max().item()
    print(f"[validate] batched-eager traj0 vs v1 final pos max|diff| = {diff:.3e} m")
    return diff


def med_rep(fn, reps):
    vals = []
    for _ in range(reps):
        m = fn()
        if isinstance(m, tuple):
            m = m[-1]
        vals.append(m['throughput_steps_per_s'])
    return float(np.median(vals))


def all_combos():
    return [
        # fused 变体先跑 (快), 含 JIT warmup
        ('fusion_only', N_FAST, REPS_FAST,
         lambda N: run_ekf_v3(n_steps=N, verbose=False), dict(n_steps=166667)),
        ('fusion_precision', N_FAST, REPS_FAST,
         lambda N: run_ekf_v4(n_steps=N, precision='fp32', verbose=False), dict(n_steps=166667)),
        ('fusion_batch', N_FAST, REPS_FAST,
         lambda N: run_ekf_v5(n_steps=N, batch=BATCH, precision='fp64', verbose=False),
         dict(n_steps=2000)),
        ('all', N_FAST, REPS_FAST,
         lambda N: run_ekf_v5(n_steps=N, batch=BATCH, precision='fp32', verbose=False),
         dict(n_steps=2000)),
        # eager 变体 (慢, 吞吐与 N 无关, 只测小 N)
        ('none', N_SLOW, REPS_SLOW,
         lambda N: run_ekf_v1(device='cuda', n_steps=N, verbose=False), None),
        ('precision_only', N_SLOW, REPS_SLOW,
         lambda N: run_ekf_v1(device='cuda', n_steps=N, precision='fp32', verbose=False), None),
        ('batch_only', N_SLOW, REPS_SLOW,
         lambda N: run_eager_batched(N, batch=BATCH), None),
        ('precision_batch', N_SLOW, REPS_SLOW,
         lambda N: run_eager_batched(N, batch=BATCH, precision='fp32'), None),
    ]


def main():
    # 可选位置参数: 只跑指定组合并合并进已有 JSON (默认全部重跑, 覆盖 JSON)
    selected = sys.argv[1:]
    combos = all_combos()
    names = {c[0] for c in combos}
    unknown = set(selected) - names
    if unknown:
        raise SystemExit(f'unknown combo(s): {sorted(unknown)}; choices: {sorted(names)}')
    if selected:
        combos = [c for c in combos if c[0] in selected]

    if selected and os.path.exists(OUT):
        results = json.load(open(OUT))
        results['combos'] = results.get('combos', {})
    else:
        results = {
            'hardware': {'gpu': torch.cuda.get_device_name(0),
                         'sm': torch.cuda.get_device_properties(0).multi_processor_count,
                         'torch': torch.__version__, 'triton': __import__('triton').__version__},
            'batch': BATCH,
            'combos': {},
        }

    print('== validate batched eager ==')
    results['validation'] = {'batched_eager_vs_v1_max_diff_m': validate_batched_eager()}

    def save():
        json.dump(results, open(OUT, 'w'), indent=2, ensure_ascii=False)
        print(f'  saved -> {OUT}')

    for name, ngrid, reps, fn, warmkw in combos:
        if warmkw is not None:
            print(f'== {name}: JIT warmup (excluded from timing) ==')
            fn(166667 if name in ('fusion_only', 'fusion_precision') else 2000)
        pts = []
        for N in ngrid:
            t0 = time.time()
            tp = med_rep(lambda: fn(N), reps)
            pts.append({'n': N, 'throughput': tp})
            print(f'  {name:18s} N={N:7d}: {tp:12.0f} steps/s   (wall {time.time()-t0:.1f}s)')
        results['combos'][name] = pts
        save()

    print('\nsummary:')
    for name, pts in results['combos'].items():
        print(f'  {name:18s}', [f"{p['n']}:{p['throughput']:.0f}" for p in pts])


if __name__ == '__main__':
    main()
