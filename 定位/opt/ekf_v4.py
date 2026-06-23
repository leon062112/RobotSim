"""
EKF v4 — 精度可配置的 Mega Kernel (TODO 6.22 #1, #5: fp64 -> fp32 / TF32)

在 v3 (单 kernel 整段 scan) 基础上，把数据类型 DT 与 tl.dot 的 input_precision IP
做成 constexpr 模板参数，一份 kernel 跑三档精度：
  - fp64  : 与 v3 等价 (金标准对照)
  - fp32  : IEEE fp32 全程 (精度换吞吐, 验证 mm 级需求是否满足)
  - tf32  : tl.dot 用 TensorCore TF32 (Hopper), 其余 fp32

贡献点 3 (算法/精度优化) 的核心实验: 降精度对 mm 级定位误差的影响 vs 吞吐收益。
数学流程与 v1/v2/v3 完全一致 (固定 3 维观测 + R 抑制异常通道 + 3x3 解析逆)。
"""
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice
import numpy as np
import os
import time


@triton.jit
def ekf_mega_kernel_p(
    gyro_ptr, accel_ptr, odom1_ptr, odom2_ptr,
    qinit_ptr, qdiag_ptr, pos_out_ptr, vel_out_ptr,
    N, dt, g, R_odo, R_vcon, delta_thresh, R_big,
    DT: tl.constexpr, IP: tl.constexpr,
):
    i = tl.arange(0, 16)
    j = tl.arange(0, 16)
    r = i[:, None]
    c = j[None, :]
    eye = (r == c).to(DT)

    q0v = tl.load(qinit_ptr + i, mask=i < 4, other=0.0).to(DT)
    qdiag = tl.load(qdiag_ptr + i, mask=i < 15, other=0.0).to(DT)
    pos = tl.zeros((16,), dtype=DT)
    vel = tl.zeros((16,), dtype=DT)
    q = q0v
    P = eye * 0.1 * ((r < 15) & (c < 15)).to(DT)
    Qmat = tl.where(r == c, qdiag[:, None], tl.zeros((16, 16), dtype=DT))

    for k in range(1, N):
        wx = tl.load(gyro_ptr + (k - 1) * 3 + 0).to(DT)
        wy = tl.load(gyro_ptr + (k - 1) * 3 + 1).to(DT)
        wz = tl.load(gyro_ptr + (k - 1) * 3 + 2).to(DT)
        fx = tl.load(accel_ptr + (k - 1) * 3 + 0).to(DT)
        fy = tl.load(accel_ptr + (k - 1) * 3 + 1).to(DT)
        fz = tl.load(accel_ptr + (k - 1) * 3 + 2).to(DT)
        o1 = tl.load(odom1_ptr + k).to(DT)
        o2 = tl.load(odom2_ptr + k).to(DT)

        pos_prev = pos
        vel_prev = vel

        tx = wx * dt
        ty = wy * dt
        tz = wz * dt
        theta_norm = libdevice.sqrt(tx * tx + ty * ty + tz * tz)
        small = theta_norm > 1e-10
        denom = tl.where(small, theta_norm, 1.0)
        coef = tl.where(small, libdevice.sin(theta_norm / 2) / denom, 0.5)
        cosv = libdevice.cos(theta_norm / 2)

        tm = tl.zeros((16, 16), dtype=DT)
        tm += tl.where((r == 0) & (c == 1), -tx, 0.0)
        tm += tl.where((r == 0) & (c == 2), -ty, 0.0)
        tm += tl.where((r == 0) & (c == 3), -tz, 0.0)
        tm += tl.where((r == 1) & (c == 0), -tx, 0.0)
        tm += tl.where((r == 1) & (c == 2), -tz, 0.0)
        tm += tl.where((r == 1) & (c == 3), -ty, 0.0)
        tm += tl.where((r == 2) & (c == 0), -ty, 0.0)
        tm += tl.where((r == 2) & (c == 1), -tz, 0.0)
        tm += tl.where((r == 2) & (c == 3), -tx, 0.0)
        tm += tl.where((r == 3) & (c == 0), -tz, 0.0)
        tm += tl.where((r == 3) & (c == 1), -ty, 0.0)
        tm += tl.where((r == 3) & (c == 2), -tx, 0.0)
        q4mask = ((r < 4) & (c < 4)).to(DT)
        q_update = (cosv * eye + coef * tm) * q4mask

        q = tl.sum(q_update * q[None, :], axis=1)
        qn = libdevice.sqrt(tl.sum(q * q))
        q = q / qn

        qa = tl.sum(tl.where(i == 0, q, 0.0))
        qb = tl.sum(tl.where(i == 1, q, 0.0))
        qc = tl.sum(tl.where(i == 2, q, 0.0))
        qd = tl.sum(tl.where(i == 3, q, 0.0))

        C00 = qa*qa+qb*qb-qc*qc-qd*qd
        C01 = 2*(qb*qc+qa*qd)
        C02 = 2*(qb*qd-qa*qc)
        C10 = 2*(qb*qc-qa*qd)
        C11 = qa*qa-qb*qb+qc*qc-qd*qd
        C12 = 2*(qc*qd+qa*qb)
        C20 = 2*(qb*qd+qa*qc)
        C21 = 2*(qc*qd-qa*qb)
        C22 = qa*qa-qb*qb-qc*qc+qd*qd

        fbx, fby, fbz = fx, fy, fz - g
        fn0 = C00*fbx + C01*fby + C02*fbz
        fn1 = C10*fbx + C11*fby + C12*fbz
        fn2 = C20*fbx + C21*fby + C22*fbz
        velinc = tl.zeros((16,), dtype=DT)
        velinc += tl.where(i == 0, fn0 * dt, 0.0)
        velinc += tl.where(i == 1, fn1 * dt, 0.0)
        velinc += tl.where(i == 2, fn2 * dt, 0.0)
        vel = vel_prev + velinc
        pos = pos_prev + vel_prev * dt

        rfn0 = C00*fx + C01*fy + C02*fz
        rfn1 = C10*fx + C11*fy + C12*fz
        rfn2 = C20*fx + C21*fy + C22*fz

        dpx = tl.sum(tl.where(i == 0, pos - pos_prev, 0.0))
        dpy = tl.sum(tl.where(i == 1, pos - pos_prev, 0.0))
        dpz = tl.sum(tl.where(i == 2, pos - pos_prev, 0.0))
        delta_S = libdevice.sqrt(dpx*dpx + dpy*dpy + dpz*dpz)
        delta_D = (o1 + o2) / 2 * dt
        normal = libdevice.abs(delta_D - delta_S) < delta_thresh
        r_odo_eff = tl.where(normal, R_odo, R_big)

        vy = tl.sum(tl.where(i == 1, vel, 0.0))
        vz = tl.sum(tl.where(i == 2, vel, 0.0))
        psi = libdevice.atan2(C01, C00)
        spsi = libdevice.sin(psi)
        cpsi = libdevice.cos(psi)

        zvec = tl.zeros((16,), dtype=DT)
        zvec += tl.where(i == 0, delta_S - delta_D, 0.0)
        zvec += tl.where(i == 1, vy, 0.0)
        zvec += tl.where(i == 2, vz, 0.0)

        H = tl.zeros((16, 16), dtype=DT)
        H += tl.where((r == 0) & (c == 3), C00, 0.0)
        H += tl.where((r == 0) & (c == 4), C01, 0.0)
        H += tl.where((r == 0) & (c == 5), C02, 0.0)
        H += tl.where((r == 1) & (c == 0), -spsi, 0.0)
        H += tl.where((r == 1) & (c == 1), cpsi, 0.0)
        H += tl.where((r == 2) & (c == 2), 1.0, 0.0)

        F = eye * ((r < 15) & (c < 15)).to(DT)
        F += tl.where((r < 3) & (c == r + 3), dt, 0.0)
        F += tl.where((r == 3) & (c == 7), rfn2 * dt, 0.0)
        F += tl.where((r == 3) & (c == 8), -rfn1 * dt, 0.0)
        F += tl.where((r == 4) & (c == 6), -rfn2 * dt, 0.0)
        F += tl.where((r == 4) & (c == 8), rfn0 * dt, 0.0)
        F += tl.where((r == 5) & (c == 6), rfn1 * dt, 0.0)
        F += tl.where((r == 5) & (c == 7), -rfn0 * dt, 0.0)
        F += tl.where((r == 3) & (c == 12), -C00 * dt, 0.0)
        F += tl.where((r == 3) & (c == 13), -C01 * dt, 0.0)
        F += tl.where((r == 3) & (c == 14), -C02 * dt, 0.0)
        F += tl.where((r == 4) & (c == 12), -C10 * dt, 0.0)
        F += tl.where((r == 4) & (c == 13), -C11 * dt, 0.0)
        F += tl.where((r == 4) & (c == 14), -C12 * dt, 0.0)
        F += tl.where((r == 5) & (c == 12), -C20 * dt, 0.0)
        F += tl.where((r == 5) & (c == 13), -C21 * dt, 0.0)
        F += tl.where((r == 5) & (c == 14), -C22 * dt, 0.0)
        F += tl.where((r == c) & (r >= 6) & (r < 9), -1.0, 0.0)
        F += tl.where((r == 6) & (c == 7), wz * dt, 0.0)
        F += tl.where((r == 6) & (c == 8), -wy * dt, 0.0)
        F += tl.where((r == 7) & (c == 6), -wz * dt, 0.0)
        F += tl.where((r == 7) & (c == 8), wx * dt, 0.0)
        F += tl.where((r == 8) & (c == 6), wy * dt, 0.0)
        F += tl.where((r == 8) & (c == 7), -wx * dt, 0.0)
        F += tl.where((r >= 6) & (r < 9) & (c == r + 3), -dt, 0.0)

        FP = tl.dot(F, P, input_precision=IP)
        P_pred = tl.dot(FP, tl.trans(F), input_precision=IP) + Qmat

        HP = tl.dot(H, P_pred, input_precision=IP)
        S = tl.dot(HP, tl.trans(H), input_precision=IP)
        S += tl.where((r == 0) & (c == 0), r_odo_eff, 0.0)
        S += tl.where((r == 1) & (c == 1), R_vcon, 0.0)
        S += tl.where((r == 2) & (c == 2), R_vcon, 0.0)

        s00 = tl.sum(tl.where((r == 0) & (c == 0), S, 0.0))
        s01 = tl.sum(tl.where((r == 0) & (c == 1), S, 0.0))
        s02 = tl.sum(tl.where((r == 0) & (c == 2), S, 0.0))
        s10 = tl.sum(tl.where((r == 1) & (c == 0), S, 0.0))
        s11 = tl.sum(tl.where((r == 1) & (c == 1), S, 0.0))
        s12 = tl.sum(tl.where((r == 1) & (c == 2), S, 0.0))
        s20 = tl.sum(tl.where((r == 2) & (c == 0), S, 0.0))
        s21 = tl.sum(tl.where((r == 2) & (c == 1), S, 0.0))
        s22 = tl.sum(tl.where((r == 2) & (c == 2), S, 0.0))

        A = s11*s22 - s12*s21
        B = -(s10*s22 - s12*s20)
        Cc = s10*s21 - s11*s20
        D = -(s01*s22 - s02*s21)
        E = s00*s22 - s02*s20
        Ff = -(s00*s21 - s01*s20)
        G = s01*s12 - s02*s11
        Hh = -(s00*s12 - s02*s10)
        Ii = s00*s11 - s01*s10
        det = s00*A + s01*B + s02*Cc
        idet = 1.0 / det
        Si = tl.zeros((16, 16), dtype=DT)
        Si += tl.where((r == 0) & (c == 0), A*idet, 0.0)
        Si += tl.where((r == 0) & (c == 1), D*idet, 0.0)
        Si += tl.where((r == 0) & (c == 2), G*idet, 0.0)
        Si += tl.where((r == 1) & (c == 0), B*idet, 0.0)
        Si += tl.where((r == 1) & (c == 1), E*idet, 0.0)
        Si += tl.where((r == 1) & (c == 2), Hh*idet, 0.0)
        Si += tl.where((r == 2) & (c == 0), Cc*idet, 0.0)
        Si += tl.where((r == 2) & (c == 1), Ff*idet, 0.0)
        Si += tl.where((r == 2) & (c == 2), Ii*idet, 0.0)

        PHt = tl.dot(P_pred, tl.trans(H), input_precision=IP)
        K = tl.dot(PHt, Si, input_precision=IP)
        x = tl.sum(K * zvec[None, :], axis=1)
        KH = tl.dot(K, H, input_precision=IP)
        P = tl.dot(eye - KH, P_pred, input_precision=IP)

        x0 = tl.sum(tl.where(i == 0, x, 0.0))
        x1 = tl.sum(tl.where(i == 1, x, 0.0))
        x2 = tl.sum(tl.where(i == 2, x, 0.0))
        x3 = tl.sum(tl.where(i == 3, x, 0.0))
        x4 = tl.sum(tl.where(i == 4, x, 0.0))
        x5 = tl.sum(tl.where(i == 5, x, 0.0))
        p6 = tl.sum(tl.where(i == 6, x, 0.0))
        p7 = tl.sum(tl.where(i == 7, x, 0.0))
        p8 = tl.sum(tl.where(i == 8, x, 0.0))
        poscorr = tl.zeros((16,), dtype=DT)
        poscorr += tl.where(i == 0, x0, 0.0)
        poscorr += tl.where(i == 1, x1, 0.0)
        poscorr += tl.where(i == 2, x2, 0.0)
        velcorr = tl.zeros((16,), dtype=DT)
        velcorr += tl.where(i == 0, x3, 0.0)
        velcorr += tl.where(i == 1, x4, 0.0)
        velcorr += tl.where(i == 2, x5, 0.0)
        pos = pos - poscorr
        vel = vel - velcorr

        d0 = 1.0
        d1 = 0.5 * p6
        d2 = 0.5 * p7
        d3 = 0.5 * p8
        dn = libdevice.sqrt(d0*d0 + d1*d1 + d2*d2 + d3*d3)
        d0 = d0/dn; d1 = d1/dn; d2 = d2/dn; d3 = d3/dn
        nqa = qa*d0 - qb*d1 - qc*d2 - qd*d3
        nqb = qa*d1 + qb*d0 + qc*d3 - qd*d2
        nqc = qa*d2 - qb*d3 + qc*d0 + qd*d1
        nqd = qa*d3 + qb*d2 - qc*d1 + qd*d0
        nn = libdevice.sqrt(nqa*nqa + nqb*nqb + nqc*nqc + nqd*nqd)
        nqa = nqa/nn; nqb = nqb/nn; nqc = nqc/nn; nqd = nqd/nn
        q = tl.zeros((16,), dtype=DT)
        q += tl.where(i == 0, nqa, 0.0)
        q += tl.where(i == 1, nqb, 0.0)
        q += tl.where(i == 2, nqc, 0.0)
        q += tl.where(i == 3, nqd, 0.0)

        tl.store(pos_out_ptr + k * 3 + 0, tl.sum(tl.where(i == 0, pos, 0.0)))
        tl.store(pos_out_ptr + k * 3 + 1, tl.sum(tl.where(i == 1, pos, 0.0)))
        tl.store(pos_out_ptr + k * 3 + 2, tl.sum(tl.where(i == 2, pos, 0.0)))
        tl.store(vel_out_ptr + k * 3 + 0, tl.sum(tl.where(i == 0, vel, 0.0)))
        tl.store(vel_out_ptr + k * 3 + 1, tl.sum(tl.where(i == 1, vel, 0.0)))
        tl.store(vel_out_ptr + k * 3 + 2, tl.sum(tl.where(i == 2, vel, 0.0)))


_PREC = {
    'fp64': (tl.float64, 'ieee', torch.float64),
    'fp32': (tl.float32, 'ieee', torch.float32),
    'tf32': (tl.float32, 'tf32', torch.float32),
}


def run_ekf_v4(csv_path='PipeRobot_Trajectory.csv', n_steps=None, precision='fp32',
               verbose=True):
    assert torch.cuda.is_available(), "v4 需要 CUDA"
    DT, IP, torch_dt = _PREC[precision]
    dev = torch.device('cuda')

    data = np.loadtxt(csv_path, delimiter=',', skiprows=1)
    t = torch.from_numpy(data[:, 0]).to(dev)
    dt_val = float((t[1:] - t[:-1]).mean().item())
    n = len(t)
    if n_steps is not None:
        n = min(n, n_steps)

    gyro = torch.from_numpy(data[:, 7:10]).to(dev, torch_dt).contiguous()
    accel = torch.from_numpy(data[:, 10:13]).to(dev, torch_dt).contiguous()
    odom1 = torch.from_numpy(data[:, 13]).to(dev, torch_dt).contiguous()
    odom2 = torch.from_numpy(data[:, 14]).to(dev, torch_dt).contiguous()
    pos_true = torch.from_numpy(data[:, 1:4]).to(dev)  # truth 保持 fp64 比对

    g = 9.81
    ax0 = accel[:10, 0].mean(); ay0 = accel[:10, 1].mean(); az0 = accel[:10, 2].mean()
    pitch0 = torch.atan(ay0 / torch.sqrt(ax0**2 + az0**2))
    roll0 = torch.atan(-ax0 / az0)
    yaw0 = torch.tensor(0.0, dtype=torch_dt, device=dev)
    cy, sy = torch.cos(yaw0/2), torch.sin(yaw0/2)
    cp, sp = torch.cos(pitch0/2), torch.sin(pitch0/2)
    cr, sr = torch.cos(roll0/2), torch.sin(roll0/2)
    qinit = torch.stack([cy*cp*cr+sy*sp*sr, cy*cp*sr-sy*sp*cr,
                         cy*sp*cr+sy*cp*sr, sy*cp*cr-cy*sp*sr]).to(torch_dt).contiguous()

    qdiag = torch.tensor([1e-6,1e-6,1e-6,1e-5,1e-5,1e-5,1e-4,1e-4,1e-4,
                          1e-8,1e-8,1e-8,1e-7,1e-7,1e-7], dtype=torch_dt, device=dev).contiguous()

    pos_out = torch.zeros(n, 3, dtype=torch_dt, device=dev).contiguous()
    vel_out = torch.zeros(n, 3, dtype=torch_dt, device=dev).contiguous()

    ekf_mega_kernel_p[(1,)](gyro, accel, odom1, odom2, qinit, qdiag,
                            pos_out, vel_out, min(n, 64), dt_val, g,
                            1e-4, 1e-3, 0.01, 1e12, DT, IP)
    torch.cuda.synchronize()
    pos_out.zero_(); vel_out.zero_()

    t_start = time.time()
    ekf_mega_kernel_p[(1,)](gyro, accel, odom1, odom2, qinit, qdiag,
                            pos_out, vel_out, n, dt_val, g,
                            1e-4, 1e-3, 0.01, 1e12, DT, IP)
    torch.cuda.synchronize()
    elapsed = time.time() - t_start

    pos_out64 = pos_out.double()
    pos_true_n = pos_true[:n]
    err = pos_out64 - pos_true_n
    rmse_x = (torch.sqrt((err[:, 0]**2).mean()) * 1000).item()
    rmse_y = (torch.sqrt((err[:, 1]**2).mean()) * 1000).item()
    rmse_z = (torch.sqrt((err[:, 2]**2).mean()) * 1000).item()
    l = (pos_true_n[:, 0].max() - pos_true_n[:, 0].min()).item()
    x_dist = torch.sqrt(((pos_out64[-1] - pos_out64[0])**2).sum()).item()
    rho = (l - x_dist) / l if l != 0 else 0.0

    metrics = {
        'version': f'v4_triton_{precision}', 'device': 'cuda', 'precision': precision,
        'n_steps': n, 'elapsed_s': elapsed, 'throughput_steps_per_s': (n - 1) / elapsed,
        'rmse_x_mm': rmse_x, 'rmse_y_mm': rmse_y, 'rmse_z_mm': rmse_z, 'rho_pct': rho * 100,
    }
    if verbose:
        print(f"[v4 {precision}] n={n} elapsed={elapsed:.4f}s "
              f"({metrics['throughput_steps_per_s']:.0f} steps/s)")
        print(f"  RMSE mm: X={rmse_x:.6f} Y={rmse_y:.6f} Z={rmse_z:.6f} | rho={rho*100:.6f}%")
    return pos_out64, vel_out, pos_true_n, t[:n], metrics


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)).rsplit('/opt', 1)[0])
    for p in ['fp64', 'fp32', 'tf32']:
        run_ekf_v4(n_steps=5000, precision=p)
