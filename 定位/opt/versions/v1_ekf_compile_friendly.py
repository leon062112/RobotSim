"""
EKF v1 — compile-friendly 重构 (TODO #1, #2)

相对 v0 的核心改造（保持数值等价于金标准）：
1. 消除所有 python 标量 / .item() / torch.tensor([标量]) 现场构造
   - theta_mat / H / F / dq 全部用 torch.stack / cat 在张量上构造
2. 统一观测模型为固定 3 维 (消除数据相关 shape 变化导致的 graph break)
   - 异常时把里程计通道噪声 R_odo 置极大 (1e12)，使该通道卡尔曼增益≈0
   - 数学上等价于 v0 "丢弃里程计观测行" (对角 R 下独立量测)
3. theta_norm 小角度分支用 torch.where (安全除法) 替代 python if
4. 单步 step 抽成纯张量函数 ekf_step()，可被 torch.compile 整体捕获 (mega-kernel 思路)

state 仅需 (pos_prev, vel_prev, q_prev, P)；x_ekf 每步重置为0，等价于 x_ekf=K@z。
"""
import torch
import numpy as np
import os
import time

R_BIG = 1e12  # 等效拒绝里程计观测


def quat2dcm_t(q):
    q0, q1, q2, q3 = q[0], q[1], q[2], q[3]
    r0 = torch.stack([q0**2+q1**2-q2**2-q3**2, 2*(q1*q2+q0*q3), 2*(q1*q3-q0*q2)])
    r1 = torch.stack([2*(q1*q2-q0*q3), q0**2-q1**2+q2**2-q3**2, 2*(q2*q3+q0*q1)])
    r2 = torch.stack([2*(q1*q3+q0*q2), 2*(q2*q3-q0*q1), q0**2-q1**2-q2**2+q3**2])
    return torch.stack([r0, r1, r2])


def quatmultiply_t(q1, q2):
    a0, a1, a2, a3 = q1[0], q1[1], q1[2], q1[3]
    b0, b1, b2, b3 = q2[0], q2[1], q2[2], q2[3]
    return torch.stack([
        a0*b0 - a1*b1 - a2*b2 - a3*b3,
        a0*b1 + a1*b0 + a2*b3 - a3*b2,
        a0*b2 - a1*b3 + a2*b0 + a3*b1,
        a0*b3 + a1*b2 - a2*b1 + a3*b0,
    ])


def ekf_step(pos_prev, vel_prev, q_prev, P, gyro_prev, accel_prev,
             odom1_k, odom2_k, dt, g_vec, I15, I3, I4, Q_noise,
             R_odo, R_vcon, delta_thresh):
    wx, wy, wz = gyro_prev[0], gyro_prev[1], gyro_prev[2]
    z0 = torch.zeros((), dtype=pos_prev.dtype, device=pos_prev.device)

    theta_x = wx * dt
    theta_y = wy * dt
    theta_z = wz * dt
    theta_norm = torch.sqrt(theta_x**2 + theta_y**2 + theta_z**2)

    tm0 = torch.stack([z0, -theta_x, -theta_y, -theta_z])
    tm1 = torch.stack([-theta_x, z0, -theta_z, -theta_y])
    tm2 = torch.stack([-theta_y, -theta_z, z0, -theta_x])
    tm3 = torch.stack([-theta_z, -theta_y, -theta_x, z0])
    theta_mat = torch.stack([tm0, tm1, tm2, tm3])

    small = theta_norm > 1e-10
    denom = torch.where(small, theta_norm, torch.ones_like(theta_norm))
    coef = torch.where(small, torch.sin(theta_norm / 2) / denom,
                       0.5 * torch.ones_like(theta_norm))
    q_update = torch.cos(theta_norm / 2) * I4 + coef * theta_mat

    q_k = q_update @ q_prev
    q_k = q_k / q_k.norm()
    Cnb = quat2dcm_t(q_k)

    f_b = accel_prev
    vel_k = vel_prev + (Cnb @ (f_b - g_vec)) * dt
    pos_k = pos_prev + vel_prev * dt

    delta_D = (odom1_k + odom2_k) / 2 * dt
    delta_S = (pos_k - pos_prev).norm()

    psi = torch.atan2(Cnb[0, 1], Cnb[0, 0])
    spsi, cpsi = torch.sin(psi), torch.cos(psi)

    # 固定 3x15 观测矩阵
    pad12 = torch.zeros(12, dtype=pos_prev.dtype, device=pos_prev.device)
    pad9 = torch.zeros(9, dtype=pos_prev.dtype, device=pos_prev.device)
    h_row0 = torch.cat([torch.zeros(3, dtype=pos_prev.dtype, device=pos_prev.device), Cnb[0], pad9])
    h_row1 = torch.cat([torch.stack([-spsi, cpsi, z0]), pad12])
    h_row2 = torch.cat([torch.stack([z0, z0, z0 + 1.0]), pad12])
    H = torch.stack([h_row0, h_row1, h_row2])

    z = torch.stack([delta_S - delta_D, vel_k[1], vel_k[2]])

    # 异常判别 -> 通过 R 抑制里程计通道 (branch-free)
    normal = torch.abs(delta_D - delta_S) < delta_thresh
    r_odo_eff = torch.where(normal, R_odo, R_odo * 0 + R_BIG)
    R = torch.diag(torch.stack([r_odo_eff, R_vcon, R_vcon]))

    # F 矩阵 (15x15)
    f_n = Cnb @ f_b
    sk_fn = torch.stack([
        torch.stack([z0, -f_n[2], f_n[1]]),
        torch.stack([f_n[2], z0, -f_n[0]]),
        torch.stack([-f_n[1], f_n[0], z0]),
    ])
    sk_w = torch.stack([
        torch.stack([z0, -wz, wy]),
        torch.stack([wz, z0, -wx]),
        torch.stack([-wy, wx, z0]),
    ])
    F = I15.clone()
    F[0:3, 3:6] = I3 * dt
    F[3:6, 6:9] = -sk_fn * dt
    F[3:6, 12:15] = -Cnb * dt
    F[6:9, 6:9] = -sk_w * dt
    F[6:9, 9:12] = -I3 * dt

    # EKF: x 进入时为 0 -> x_post = K@z
    P_pred = F @ P @ F.T + Q_noise
    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ torch.linalg.inv(S)
    x_ekf = K @ z
    P_new = (I15 - K @ H) @ P_pred

    pos_k = pos_k - x_ekf[0:3]
    vel_k = vel_k - x_ekf[3:6]

    phi = x_ekf[6:9]
    dq = torch.stack([z0 + 1.0, 0.5*phi[0], 0.5*phi[1], 0.5*phi[2]])
    dq = dq / dq.norm()
    q_k = quatmultiply_t(q_k, dq)
    q_k = q_k / q_k.norm()

    return pos_k, vel_k, q_k, P_new


def run_ekf_v1(csv_path='PipeRobot_Trajectory.csv', device='cpu', n_steps=None,
               compile_mode=None, verbose=True):
    dev = torch.device(device)
    data = np.loadtxt(csv_path, delimiter=',', skiprows=1)
    t = torch.from_numpy(data[:, 0]).to(dev)
    dt = (t[1:] - t[:-1]).mean().item()
    n = len(t)
    if n_steps is not None:
        n = min(n, n_steps)

    gyro = torch.from_numpy(data[:, 7:10]).to(dev)
    accel = torch.from_numpy(data[:, 10:13]).to(dev)
    odom1 = torch.from_numpy(data[:, 13]).to(dev)
    odom2 = torch.from_numpy(data[:, 14]).to(dev)
    pos_true = torch.from_numpy(data[:, 1:4]).to(dev)

    g = 9.81
    dtype = torch.float64
    g_vec = torch.tensor([0, 0, g], dtype=dtype, device=dev)
    I15 = torch.eye(15, dtype=dtype, device=dev)
    I3 = torch.eye(3, dtype=dtype, device=dev)
    I4 = torch.eye(4, dtype=dtype, device=dev)
    Q_noise = torch.diag(torch.tensor([
        1e-6, 1e-6, 1e-6, 1e-5, 1e-5, 1e-5, 1e-4, 1e-4, 1e-4,
        1e-8, 1e-8, 1e-8, 1e-7, 1e-7, 1e-7], dtype=dtype, device=dev))
    R_odo = torch.tensor(1e-4, dtype=dtype, device=dev)
    R_vcon = torch.tensor(1e-3, dtype=dtype, device=dev)
    delta_thresh = torch.tensor(0.01, dtype=dtype, device=dev)
    dt_t = torch.tensor(dt, dtype=dtype, device=dev)

    # 初始对准
    ax0 = accel[:10, 0].mean()
    ay0 = accel[:10, 1].mean()
    az0 = accel[:10, 2].mean()
    pitch0 = torch.atan(ay0 / torch.sqrt(ax0**2 + az0**2))
    roll0 = torch.atan(-ax0 / az0)
    yaw0 = torch.tensor(0.0, dtype=dtype, device=dev)
    cy, sy = torch.cos(yaw0/2), torch.sin(yaw0/2)
    cp, sp = torch.cos(pitch0/2), torch.sin(pitch0/2)
    cr, sr = torch.cos(roll0/2), torch.sin(roll0/2)
    q_prev = torch.stack([
        cy*cp*cr + sy*sp*sr, cy*cp*sr - sy*sp*cr,
        cy*sp*cr + sy*cp*sr, sy*cp*cr - cy*sp*sr])

    pos_prev = torch.zeros(3, dtype=dtype, device=dev)
    vel_prev = torch.zeros(3, dtype=dtype, device=dev)
    P = torch.eye(15, dtype=dtype, device=dev) * 0.1

    pos_fusion = torch.zeros(n, 3, dtype=dtype, device=dev)
    vel_fusion = torch.zeros(n, 3, dtype=dtype, device=dev)

    step_fn = ekf_step
    if compile_mode is not None:
        step_fn = torch.compile(ekf_step, mode=compile_mode, dynamic=False)

    if dev.type == 'cuda':
        torch.cuda.synchronize()
    t_start = time.time()

    for k in range(1, n):
        pos_prev, vel_prev, q_prev, P = step_fn(
            pos_prev, vel_prev, q_prev, P,
            gyro[k-1], accel[k-1], odom1[k], odom2[k],
            dt_t, g_vec, I15, I3, I4, Q_noise, R_odo, R_vcon, delta_thresh)
        pos_fusion[k] = pos_prev
        vel_fusion[k] = vel_prev

    if dev.type == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.time() - t_start

    pos_true_n = pos_true[:n]
    error_fusion = pos_fusion - pos_true_n
    rmse_x = (torch.sqrt((error_fusion[:, 0]**2).mean()) * 1000).item()
    rmse_y = (torch.sqrt((error_fusion[:, 1]**2).mean()) * 1000).item()
    rmse_z = (torch.sqrt((error_fusion[:, 2]**2).mean()) * 1000).item()
    l = (pos_true_n[:, 0].max() - pos_true_n[:, 0].min()).item()
    x_dist = torch.sqrt(((pos_fusion[-1] - pos_fusion[0])**2).sum()).item()
    rho = (l - x_dist) / l if l != 0 else 0.0

    metrics = {
        'version': 'v1', 'device': device, 'compile_mode': compile_mode,
        'n_steps': n, 'elapsed_s': elapsed,
        'throughput_steps_per_s': (n - 1) / elapsed,
        'rmse_x_mm': rmse_x, 'rmse_y_mm': rmse_y, 'rmse_z_mm': rmse_z,
        'rho_pct': rho * 100,
    }
    if verbose:
        print(f"[v1] device={device} compile={compile_mode} n={n} "
              f"elapsed={elapsed:.3f}s ({metrics['throughput_steps_per_s']:.0f} steps/s)")
        print(f"  RMSE mm: X={rmse_x:.6f} Y={rmse_y:.6f} Z={rmse_z:.6f} | rho={rho*100:.6f}%")
    return pos_fusion, vel_fusion, pos_true_n, t[:n], metrics


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)).rsplit('/opt', 1)[0])
    run_ekf_v1(device='cpu', n_steps=5000)
