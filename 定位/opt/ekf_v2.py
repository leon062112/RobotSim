"""
EKF v2 — CUDA Graph capture/replay "Mega Kernel" (TODO #2)

思路: 卡尔曼递归是严格串行 scan (步k依赖k-1)，无法跨步并行。
但每步的计算图是固定的 (v1 已证明 0 graph break)。
=> 把"单步"用 CUDA Graph 捕获一次，之后每步只 replay：
   - N-1 次 python 循环里，每次只发 1 个 graph launch (而非几十个 kernel launch)
   - 消除每步的 CPU 调度 / kernel launch 开销
   - 所有 state / 输入数据常驻 GPU 静态 buffer，零 H2D/同步

实现要点:
- 全部输入数据预加载到 GPU 静态张量
- step 索引 idx 用 GPU long 标量，graph 内部自增 (idx.add_(1))，CPU 零参与
- state(pos/vel/q/P) 用 in-place copy_ 更新到静态 buffer
- 输出 pos_fusion[idx]/vel_fusion[idx] 用 index_put_ 写入
"""
import torch
import numpy as np
import os
import time

R_BIG = 1e12


def inv3x3(M):
    """3x3 矩阵解析逆 (adjugate/det)，纯 elementwise，CUDA-Graph 安全，无 cuSOLVER。"""
    a, b, c = M[0, 0], M[0, 1], M[0, 2]
    d, e, f = M[1, 0], M[1, 1], M[1, 2]
    g, h, i = M[2, 0], M[2, 1], M[2, 2]
    A = e * i - f * h
    B = -(d * i - f * g)
    C = d * h - e * g
    D = -(b * i - c * h)
    E = a * i - c * g
    F = -(a * h - b * g)
    G = b * f - c * e
    H_ = -(a * f - c * d)
    I_ = a * e - b * d
    det = a * A + b * B + c * C
    inv_det = 1.0 / det
    r0 = torch.stack([A, D, G])
    r1 = torch.stack([B, E, H_])
    r2 = torch.stack([C, F, I_])
    return torch.stack([r0, r1, r2]) * inv_det


def _build_graph_step(buf):
    """返回一个无参函数，对静态 buffer 做单步 EKF 原位更新。"""
    dtype = buf['pos'].dtype
    dev = buf['pos'].device
    z0 = torch.zeros((), dtype=dtype, device=dev)
    one = torch.ones((), dtype=dtype, device=dev)

    gyro_all = buf['gyro']; accel_all = buf['accel']
    odom1_all = buf['odom1']; odom2_all = buf['odom2']
    pos = buf['pos']; vel = buf['vel']; q = buf['q']; P = buf['P']
    pos_fusion = buf['pos_fusion']; vel_fusion = buf['vel_fusion']
    idx = buf['idx']
    dt = buf['dt']; g_vec = buf['g_vec']
    I15 = buf['I15']; I3 = buf['I3']; I4 = buf['I4']
    Q_noise = buf['Q']; R_odo = buf['R_odo']; R_vcon = buf['R_vcon']
    delta_thresh = buf['delta_thresh']

    def step():
        k = idx  # 1-elem long tensor on GPU
        km1 = k - 1
        gyro_prev = torch.index_select(gyro_all, 0, km1)[0]
        accel_prev = torch.index_select(accel_all, 0, km1)[0]
        odom1_k = torch.index_select(odom1_all, 0, k)[0]
        odom2_k = torch.index_select(odom2_all, 0, k)[0]

        pos_prev = pos.clone()
        vel_prev = vel.clone()
        q_prev = q.clone()
        P_prev = P.clone()

        wx, wy, wz = gyro_prev[0], gyro_prev[1], gyro_prev[2]
        theta_x = wx * dt; theta_y = wy * dt; theta_z = wz * dt
        theta_norm = torch.sqrt(theta_x**2 + theta_y**2 + theta_z**2)

        tm0 = torch.stack([z0, -theta_x, -theta_y, -theta_z])
        tm1 = torch.stack([-theta_x, z0, -theta_z, -theta_y])
        tm2 = torch.stack([-theta_y, -theta_z, z0, -theta_x])
        tm3 = torch.stack([-theta_z, -theta_y, -theta_x, z0])
        theta_mat = torch.stack([tm0, tm1, tm2, tm3])

        small = theta_norm > 1e-10
        denom = torch.where(small, theta_norm, one)
        coef = torch.where(small, torch.sin(theta_norm / 2) / denom, 0.5 * one)
        q_update = torch.cos(theta_norm / 2) * I4 + coef * theta_mat

        q_k = q_update @ q_prev
        q_k = q_k / q_k.norm()

        q0, q1, q2, q3 = q_k[0], q_k[1], q_k[2], q_k[3]
        cr0 = torch.stack([q0**2+q1**2-q2**2-q3**2, 2*(q1*q2+q0*q3), 2*(q1*q3-q0*q2)])
        cr1 = torch.stack([2*(q1*q2-q0*q3), q0**2-q1**2+q2**2-q3**2, 2*(q2*q3+q0*q1)])
        cr2 = torch.stack([2*(q1*q3+q0*q2), 2*(q2*q3-q0*q1), q0**2-q1**2-q2**2+q3**2])
        Cnb = torch.stack([cr0, cr1, cr2])

        f_b = accel_prev
        vel_k = vel_prev + (Cnb @ (f_b - g_vec)) * dt
        pos_k = pos_prev + vel_prev * dt

        delta_D = (odom1_k + odom2_k) / 2 * dt
        delta_S = (pos_k - pos_prev).norm()

        psi = torch.atan2(Cnb[0, 1], Cnb[0, 0])
        spsi, cpsi = torch.sin(psi), torch.cos(psi)

        zero3 = torch.zeros(3, dtype=dtype, device=dev)
        pad12 = torch.zeros(12, dtype=dtype, device=dev)
        h_row0 = torch.cat([zero3, Cnb[0], torch.zeros(9, dtype=dtype, device=dev)])
        h_row1 = torch.cat([torch.stack([-spsi, cpsi, z0]), pad12])
        h_row2 = torch.cat([torch.stack([z0, z0, one]), pad12])
        H = torch.stack([h_row0, h_row1, h_row2])

        z = torch.stack([delta_S - delta_D, vel_k[1], vel_k[2]])

        normal = torch.abs(delta_D - delta_S) < delta_thresh
        r_odo_eff = torch.where(normal, R_odo, R_odo * 0 + R_BIG)
        R = torch.diag(torch.stack([r_odo_eff, R_vcon, R_vcon]))

        f_n = Cnb @ f_b
        sk_fn = torch.stack([
            torch.stack([z0, -f_n[2], f_n[1]]),
            torch.stack([f_n[2], z0, -f_n[0]]),
            torch.stack([-f_n[1], f_n[0], z0])])
        sk_w = torch.stack([
            torch.stack([z0, -wz, wy]),
            torch.stack([wz, z0, -wx]),
            torch.stack([-wy, wx, z0])])
        F = I15.clone()
        F[0:3, 3:6] = I3 * dt
        F[3:6, 6:9] = -sk_fn * dt
        F[3:6, 12:15] = -Cnb * dt
        F[6:9, 6:9] = -sk_w * dt
        F[6:9, 9:12] = -I3 * dt

        P_pred = F @ P_prev @ F.T + Q_noise
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ inv3x3(S)
        x_ekf = K @ z
        P_new = (I15 - K @ H) @ P_pred

        pos_k = pos_k - x_ekf[0:3]
        vel_k = vel_k - x_ekf[3:6]
        phi = x_ekf[6:9]
        dq = torch.stack([one, 0.5*phi[0], 0.5*phi[1], 0.5*phi[2]])
        dq = dq / dq.norm()
        a0, a1, a2, a3 = q_k[0], q_k[1], q_k[2], q_k[3]
        b0, b1, b2, b3 = dq[0], dq[1], dq[2], dq[3]
        q_k = torch.stack([
            a0*b0 - a1*b1 - a2*b2 - a3*b3,
            a0*b1 + a1*b0 + a2*b3 - a3*b2,
            a0*b2 - a1*b3 + a2*b0 + a3*b1,
            a0*b3 + a1*b2 - a2*b1 + a3*b0])
        q_k = q_k / q_k.norm()

        # 原位写回静态 buffer
        pos.copy_(pos_k)
        vel.copy_(vel_k)
        q.copy_(q_k)
        P.copy_(P_new)
        pos_fusion.index_copy_(0, k, pos_k.unsqueeze(0))
        vel_fusion.index_copy_(0, k, vel_k.unsqueeze(0))
        idx.add_(1)
    return step


def run_ekf_v2(csv_path='PipeRobot_Trajectory.csv', n_steps=None, unroll=1, verbose=True):
    assert torch.cuda.is_available(), "v2 需要 CUDA"
    dev = torch.device('cuda')
    dtype = torch.float64

    data = np.loadtxt(csv_path, delimiter=',', skiprows=1)
    t = torch.from_numpy(data[:, 0]).to(dev)
    dt_val = (t[1:] - t[:-1]).mean().item()
    n = len(t)
    if n_steps is not None:
        n = min(n, n_steps)

    gyro = torch.from_numpy(data[:, 7:10]).to(dev).contiguous()
    accel = torch.from_numpy(data[:, 10:13]).to(dev).contiguous()
    odom1 = torch.from_numpy(data[:, 13]).to(dev).contiguous()
    odom2 = torch.from_numpy(data[:, 14]).to(dev).contiguous()
    pos_true = torch.from_numpy(data[:, 1:4]).to(dev)

    g = 9.81
    ax0 = accel[:10, 0].mean(); ay0 = accel[:10, 1].mean(); az0 = accel[:10, 2].mean()
    pitch0 = torch.atan(ay0 / torch.sqrt(ax0**2 + az0**2))
    roll0 = torch.atan(-ax0 / az0)
    yaw0 = torch.tensor(0.0, dtype=dtype, device=dev)
    cy, sy = torch.cos(yaw0/2), torch.sin(yaw0/2)
    cp, sp = torch.cos(pitch0/2), torch.sin(pitch0/2)
    cr, sr = torch.cos(roll0/2), torch.sin(roll0/2)
    q0 = torch.stack([cy*cp*cr+sy*sp*sr, cy*cp*sr-sy*sp*cr,
                      cy*sp*cr+sy*cp*sr, sy*cp*cr-cy*sp*sr])

    buf = {
        'gyro': gyro, 'accel': accel, 'odom1': odom1, 'odom2': odom2,
        'pos': torch.zeros(3, dtype=dtype, device=dev),
        'vel': torch.zeros(3, dtype=dtype, device=dev),
        'q': q0.clone(),
        'P': torch.eye(15, dtype=dtype, device=dev) * 0.1,
        'pos_fusion': torch.zeros(n, 3, dtype=dtype, device=dev),
        'vel_fusion': torch.zeros(n, 3, dtype=dtype, device=dev),
        'idx': torch.tensor([1], dtype=torch.long, device=dev),
        'dt': torch.tensor(dt_val, dtype=dtype, device=dev),
        'g_vec': torch.tensor([0, 0, g], dtype=dtype, device=dev),
        'I15': torch.eye(15, dtype=dtype, device=dev),
        'I3': torch.eye(3, dtype=dtype, device=dev),
        'I4': torch.eye(4, dtype=dtype, device=dev),
        'Q': torch.diag(torch.tensor([1e-6,1e-6,1e-6,1e-5,1e-5,1e-5,1e-4,1e-4,1e-4,
                                      1e-8,1e-8,1e-8,1e-7,1e-7,1e-7], dtype=dtype, device=dev)),
        'R_odo': torch.tensor(1e-4, dtype=dtype, device=dev),
        'R_vcon': torch.tensor(1e-3, dtype=dtype, device=dev),
        'delta_thresh': torch.tensor(0.01, dtype=dtype, device=dev),
    }

    step = _build_graph_step(buf)

    def step_block():
        for _ in range(unroll):
            step()

    def reset_state():
        buf['pos'].zero_(); buf['vel'].zero_(); buf['q'].copy_(q0)
        buf['P'].copy_(torch.eye(15, dtype=dtype, device=dev) * 0.1)
        buf['pos_fusion'].zero_(); buf['vel_fusion'].zero_()
        buf['idx'].fill_(1)

    # ---- warmup on side stream (CUDA Graph 要求) ----
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            step_block()
    torch.cuda.current_stream().wait_stream(s)

    reset_state()

    # ---- capture ----
    g_cap = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g_cap):
        step_block()

    reset_state()

    # n-1 步, 每个 graph 跑 unroll 步
    n_replays = (n - 1) // unroll
    torch.cuda.synchronize()
    t_start = time.time()
    for _ in range(n_replays):
        g_cap.replay()
    torch.cuda.synchronize()
    elapsed = time.time() - t_start
    steps_done = n_replays * unroll

    pos_fusion = buf['pos_fusion']
    m = steps_done + 1  # 有效行 [0, steps_done]
    pos_fusion = pos_fusion[:m]
    pos_true_n = pos_true[:m]
    error_fusion = pos_fusion - pos_true_n
    rmse_x = (torch.sqrt((error_fusion[:, 0]**2).mean()) * 1000).item()
    rmse_y = (torch.sqrt((error_fusion[:, 1]**2).mean()) * 1000).item()
    rmse_z = (torch.sqrt((error_fusion[:, 2]**2).mean()) * 1000).item()
    l = (pos_true_n[:, 0].max() - pos_true_n[:, 0].min()).item()
    x_dist = torch.sqrt(((pos_fusion[-1] - pos_fusion[0])**2).sum()).item()
    rho = (l - x_dist) / l if l != 0 else 0.0

    metrics = {
        'version': 'v2_cudagraph', 'device': 'cuda', 'n_steps': m, 'unroll': unroll,
        'elapsed_s': elapsed, 'throughput_steps_per_s': steps_done / elapsed,
        'rmse_x_mm': rmse_x, 'rmse_y_mm': rmse_y, 'rmse_z_mm': rmse_z, 'rho_pct': rho * 100,
    }
    if verbose:
        print(f"[v2 CUDAGraph] n={m} unroll={unroll} elapsed={elapsed:.3f}s "
              f"({metrics['throughput_steps_per_s']:.0f} steps/s)")
        print(f"  RMSE mm: X={rmse_x:.6f} Y={rmse_y:.6f} Z={rmse_z:.6f} | rho={rho*100:.6f}%")
    return pos_fusion, buf['vel_fusion'][:m], pos_true_n, t[:m], metrics


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)).rsplit('/opt', 1)[0])
    run_ekf_v2(n_steps=5000)
