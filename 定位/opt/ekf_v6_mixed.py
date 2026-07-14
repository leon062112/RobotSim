"""
EKF v6 — 混合精度 Mega Kernel (贡献点2 深化)

简化但实用的混合精度方案:
  - 保持 v4 的 kernel body 完全不变 (经过充分验证)
  - 在 kernel 外部添加 3 层精度控制:
    1. I/O 精度: sensor 输入和 pos/vel 输出的 dtype
    2. 计算精度 (DT): kernel 内部所有运算的 dtype (fp32 / fp64)
    3. Matmul 精度 (IP): tl.dot 的 input_precision (ieee / tf32)

方案矩阵:
  | 方案            | I/O 精度 | 计算 DT  | tl.dot IP | 含义                    |
  |----------------|---------|---------|-----------|------------------------|
  | fp32-baseline  | fp32    | fp32    | ieee      | = v4 fp32 (基线)        |
  | bf16-io        | bf16    | fp32    | ieee      | I/O 降精度, 计算不变    |
  | tf32-dot       | fp32    | fp32    | tf32      | 仅 matmul 用 TF32 TC   |
  | bf16-io+tf32   | bf16    | fp32    | tf32      | I/O + matmul 同时降精度 |
  | fp16-io        | fp16    | fp32    | ieee      | I/O fp16 (测试可行性)   |
"""
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice
import numpy as np
import os
import time
import json
import copy


# ==========================================================================
# Kernel: IDENTICAL to v4 ekf_mega_kernel_p — proven correct
# ==========================================================================

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


# ==========================================================================
# Precision Combos: (io_dtype, compute_DT, dot_IP, description)
# ==========================================================================
# io_dtype = dtype for sensor input and pos/vel output buffers
# compute_DT = dtype for ALL internal kernel computation (unified, like v4)
# dot_IP = input_precision for all 8 tl.dot calls
_COMBO = {
    'fp32-baseline': {
        'io_dtype': torch.float32,
        'compute_DT': tl.float32,
        'dot_IP': 'ieee',
        'desc': 'v4 fp32 等价基线',
        'category': 'baseline',
    },
    'bf16-io': {
        'io_dtype': torch.bfloat16,
        'compute_DT': tl.float32,
        'dot_IP': 'ieee',
        'desc': 'I/O 降 bf16, 计算 fp32 (省带宽/寄存器)',
        'category': 'io-precision',
    },
    'tf32-dot': {
        'io_dtype': torch.float32,
        'compute_DT': tl.float32,
        'dot_IP': 'tf32',
        'desc': 'I/O fp32, matmul 用 TF32 TensorCore',
        'category': 'dot-precision',
    },
    'bf16-io+tf32': {
        'io_dtype': torch.bfloat16,
        'compute_DT': tl.float32,
        'dot_IP': 'tf32',
        'desc': 'I/O bf16 + matmul TF32 (最激进)',
        'category': 'combined',
    },
    'fp16-io': {
        'io_dtype': torch.float16,
        'compute_DT': tl.float32,
        'dot_IP': 'ieee',
        'desc': 'I/O fp16, 计算 fp32 (测试 fp16 可行性)',
        'category': 'io-precision',
    },
    # Note: fp32-tf32 is a duplicate of tf32-dot; removed to avoid double-counting
}


def run_ekf_v6_mixed(csv_path='PipeRobot_Trajectory.csv', n_steps=None,
                      combo='fp32-baseline', verbose=True):
    """Run EKF with mixed precision config."""
    assert torch.cuda.is_available(), "v6 requires CUDA"
    assert combo in _COMBO, f"Unknown combo: {combo}"

    cfg = _COMBO[combo]
    io_dtype = cfg['io_dtype']
    compute_DT = cfg['compute_DT']
    dot_IP = cfg['dot_IP']
    dev = torch.device('cuda')

    # Load data
    data = np.loadtxt(csv_path, delimiter=',', skiprows=1)
    t = torch.from_numpy(data[:, 0]).to(dev)
    dt_val = float((t[1:] - t[:-1]).mean().item())
    n = len(t)
    if n_steps is not None:
        n = min(n, n_steps)

    # Sensor inputs in io_dtype for I/O bandwidth savings
    gyro = torch.from_numpy(data[:, 7:10]).to(dev, io_dtype).contiguous()
    accel = torch.from_numpy(data[:, 10:13]).to(dev, io_dtype).contiguous()
    odom1 = torch.from_numpy(data[:, 13]).to(dev, io_dtype).contiguous()
    odom2 = torch.from_numpy(data[:, 14]).to(dev, io_dtype).contiguous()
    pos_true = torch.from_numpy(data[:, 1:4]).to(dev)  # truth always fp64

    # Init (compute in fp32 for robustness)
    g = 9.81
    ax0 = accel[:10, 0].float().mean(); ay0 = accel[:10, 1].float().mean(); az0 = accel[:10, 2].float().mean()
    pitch0 = torch.atan(ay0 / torch.sqrt(ax0**2 + az0**2))
    roll0 = torch.atan(-ax0 / az0)
    yaw0 = torch.tensor(0.0, dtype=torch.float32, device=dev)
    cy, sy = torch.cos(yaw0/2), torch.sin(yaw0/2)
    cp, sp = torch.cos(pitch0/2), torch.sin(pitch0/2)
    cr, sr = torch.cos(roll0/2), torch.sin(roll0/2)
    qinit = torch.stack([cy*cp*cr+sy*sp*sr, cy*cp*sr-sy*sp*cr,
                         cy*sp*cr+sy*cp*sr, sy*cp*cr-cy*sp*sr]).to(io_dtype).contiguous()

    # qdiag in io_dtype (values 1e-8 to 1e-4 safe in all tested dtypes)
    qdiag = torch.tensor([1e-6,1e-6,1e-6,1e-5,1e-5,1e-5,1e-4,1e-4,1e-4,
                          1e-8,1e-8,1e-8,1e-7,1e-7,1e-7], dtype=io_dtype, device=dev).contiguous()

    # Output buffers in io_dtype
    pos_out = torch.zeros(n, 3, dtype=io_dtype, device=dev).contiguous()
    vel_out = torch.zeros(n, 3, dtype=io_dtype, device=dev).contiguous()

    # Warmup
    ekf_mega_kernel_p[(1,)](gyro, accel, odom1, odom2, qinit, qdiag,
                             pos_out, vel_out, min(n, 64), dt_val, g,
                             1e-4, 1e-3, 0.01, 1e12, compute_DT, dot_IP)
    torch.cuda.synchronize()
    pos_out.zero_(); vel_out.zero_()

    # Timed run
    t_start = time.time()
    ekf_mega_kernel_p[(1,)](gyro, accel, odom1, odom2, qinit, qdiag,
                             pos_out, vel_out, n, dt_val, g,
                             1e-4, 1e-3, 0.01, 1e12, compute_DT, dot_IP)
    torch.cuda.synchronize()
    elapsed = time.time() - t_start

    # Validation
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
        'version': 'v6_mixed', 'combo': combo, 'category': cfg['category'],
        'io_dtype': str(io_dtype), 'compute_DT': str(compute_DT), 'dot_IP': dot_IP,
        'desc': cfg['desc'],
        'n_steps': n, 'elapsed_s': elapsed, 'throughput_steps_per_s': (n - 1) / elapsed,
        'rmse_x_mm': rmse_x, 'rmse_y_mm': rmse_y, 'rmse_z_mm': rmse_z, 'rho_pct': rho * 100,
    }
    if verbose:
        print(f"[v6 {combo}] n={n} elapsed={elapsed:.4f}s "
              f"({metrics['throughput_steps_per_s']:.0f} steps/s)")
        print(f"  RMSE mm: X={rmse_x:.6f} Y={rmse_y:.6f} Z={rmse_z:.6f} | rho={rho*100:.6f}%")
    return pos_out64, vel_out, pos_true_n, t[:n], metrics


def sweep_v6(csv_path='PipeRobot_Trajectory.csv', n_steps=None, quick=False):
    """Sweep all combos and produce tradeoff analysis."""
    GOLDEN = dict(rmse_x_mm=1017.838305, rmse_y_mm=16.834549,
                  rmse_z_mm=8.140964, rho_pct=0.175540)
    # Expected partial n=5000 RMSE from v4 fp32
    V4_N5000 = dict(rmse_x_mm=948.749, rmse_y_mm=2.758, rmse_z_mm=1.133)

    n = n_steps if n_steps else (5000 if quick else None)
    all_results = []

    print("=" * 80)
    print("EKF v6 Mixed Precision Sweep")
    print(f"  (n={n or 'full'}, I/O precision varied, compute=fp32, dot=IP)")
    print("=" * 80)

    for combo_name in sorted(_COMBO.keys()):
        cfg = _COMBO[combo_name]
        print(f"\n--- {combo_name}: {cfg['desc']} ---")
        try:
            _, _, _, _, m = run_ekf_v6_mixed(csv_path=csv_path, n_steps=n,
                                              combo=combo_name, verbose=True)
            # Compare to expected RMSE for n=5000
            if n == 5000:
                v4ref = V4_N5000
                m['rmse_delta_x_vs_v4_mm'] = abs(m['rmse_x_mm'] - v4ref['rmse_x_mm'])
                m['rmse_delta_y_vs_v4_mm'] = abs(m['rmse_y_mm'] - v4ref['rmse_y_mm'])
                m['rmse_delta_z_vs_v4_mm'] = abs(m['rmse_z_mm'] - v4ref['rmse_z_mm'])
                m['matches_v4'] = (m['rmse_delta_x_vs_v4_mm'] < 1.0 and
                                    m['rmse_delta_y_vs_v4_mm'] < 0.1 and
                                    m['rmse_delta_z_vs_v4_mm'] < 0.1)
            elif n is None or n >= 160000:
                m['golden_match'] = (abs(m['rmse_x_mm'] - GOLDEN['rmse_x_mm']) < 1.0)
            all_results.append(m)
        except Exception as e:
            print(f"  FAILED: {e}")
            all_results.append({'combo': combo_name, 'error': str(e)})

    # Save
    os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results'), exist_ok=True)
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'v6_mixed_precision.json')
    out = {'results': all_results, 'golden': GOLDEN, 'v4_n5000_ref': V4_N5000, 'n_steps': n or 'full'}
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nSaved -> {out_path}")

    # Summary
    baseline_tput = None
    for r in all_results:
        if r.get('combo') == 'fp32-baseline':
            baseline_tput = r.get('throughput_steps_per_s', 0)
            break

    print("\n" + "=" * 100)
    print("Precision-Performance Tradeoff Analysis")
    print("=" * 100)
    header = f"{'Combo':<20} {'Category':<14} {'Steps/s':>10} {'vsBase':>8} {'RMSE_X':>10} {'RMSE_Y':>8} {'RMSE_Z':>8} {'OK':>5}"
    print(header)
    print("-" * 100)
    for r in all_results:
        if 'error' in r:
            print(f"{r['combo']:<20} {'FAILED':>14}")
        else:
            vs = f"{r['throughput_steps_per_s'] / baseline_tput:.2f}x" if baseline_tput else "N/A"
            ok = "YES" if r.get('matches_v4') else (str(r.get('golden_match', 'N/A')))
            print(f"{r['combo']:<20} {r['category']:<14} {r['throughput_steps_per_s']:>10.0f} "
                  f"{vs:>8} {r['rmse_x_mm']:>10.3f} {r['rmse_y_mm']:>8.3f} "
                  f"{r['rmse_z_mm']:>8.3f} {ok:>5}")

    # Tradeoff analysis
    print("\n--- Analysis ---")
    categories = {}
    for r in all_results:
        if 'error' not in r:
            cat = r['category']
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(r)

    print(f"\nBaseline (fp32-baseline): throughput measurement repeatability check")
    for cat, items in categories.items():
        if cat == 'baseline':
            for r in items:
                print(f"  {r['combo']}: {r['throughput_steps_per_s']:.0f} steps/s "
                      f"(RMSE X={r['rmse_x_mm']:.3f} — should match v4 fp32)")

    print(f"\nI/O Precision Impact (same compute, different I/O dtype):")
    for cat, items in categories.items():
        if cat == 'io-precision':
            for r in items:
                print(f"  {r['combo']} ({r['desc']}): {r['throughput_steps_per_s']:.0f} steps/s "
                      f"X={r['rmse_x_mm']:.3f} Y={r['rmse_y_mm']:.3f}")

    print(f"\ntl.dot IP Impact (same I/O, different matmul precision):")
    for cat, items in categories.items():
        if cat == 'dot-precision':
            for r in items:
                print(f"  {r['combo']} ({r['desc']}): {r['throughput_steps_per_s']:.0f} steps/s "
                      f"X={r['rmse_x_mm']:.3f} Y={r['rmse_y_mm']:.3f}")

    return all_results


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)).rsplit('/opt', 1)[0])
    sweep_v6(quick=True)