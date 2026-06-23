"""
EKF v3 — 真正的 Mega Kernel: 整个串行 scan 在单个 Triton kernel 内完成 (TODO #2 终极)

设计:
- 1 个 kernel launch 处理全部 N 步 (vs v2 的 N 次 graph replay; vs v0 的 N×531 kernel)
- 单 program (1 block)，state(pos/vel/q/P) 常驻寄存器，跨步循环零全局内存往返
- 每步仅从 global 读 gyro/accel/odom，写 pos_out/vel_out
- 所有矩阵 padded 到 16×16，用 tl.dot(fp64) 做 matmul，mask 构造/抽取标量
- 3×3 求逆用解析 adjugate (无 cuSOLVER)
- 分支用 where (R_odo→R_big 抑制异常里程计通道)，与 v1/v2 数学等价
"""
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice
import numpy as np
import os
import time


@triton.jit
def ekf_mega_kernel(
    gyro_ptr, accel_ptr, odom1_ptr, odom2_ptr,
    qinit_ptr, qdiag_ptr, pos_out_ptr, vel_out_ptr,
    N, dt, g, R_odo, R_vcon, delta_thresh, R_big,
):
    i = tl.arange(0, 16)
    j = tl.arange(0, 16)
    r = i[:, None]
    c = j[None, :]
    eye = (r == c).to(tl.float64)

    # ---- init state ----
    q0v = tl.load(qinit_ptr + i, mask=i < 4, other=0.0)      # (16,) lanes 0..3
    qdiag = tl.load(qdiag_ptr + i, mask=i < 15, other=0.0)
    pos = tl.zeros((16,), dtype=tl.float64)
    vel = tl.zeros((16,), dtype=tl.float64)
    q = q0v
    P = eye * 0.1 * ((r < 15) & (c < 15)).to(tl.float64)
    Qmat = tl.where(r == c, qdiag[:, None], 0.0)

    for k in range(1, N):
        # ---- load inputs ----
        wx = tl.load(gyro_ptr + (k - 1) * 3 + 0)
        wy = tl.load(gyro_ptr + (k - 1) * 3 + 1)
        wz = tl.load(gyro_ptr + (k - 1) * 3 + 2)
        fx = tl.load(accel_ptr + (k - 1) * 3 + 0)
        fy = tl.load(accel_ptr + (k - 1) * 3 + 1)
        fz = tl.load(accel_ptr + (k - 1) * 3 + 2)
        o1 = tl.load(odom1_ptr + k)
        o2 = tl.load(odom2_ptr + k)

        pos_prev = pos
        vel_prev = vel

        # ---- attitude update ----
        tx = wx * dt
        ty = wy * dt
        tz = wz * dt
        theta_norm = libdevice.sqrt(tx * tx + ty * ty + tz * tz)
        small = theta_norm > 1e-10
        denom = tl.where(small, theta_norm, 1.0)
        coef = tl.where(small, libdevice.sin(theta_norm / 2) / denom, 0.5)
        cosv = libdevice.cos(theta_norm / 2)

        tm = tl.zeros((16, 16), dtype=tl.float64)
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
        q4mask = ((r < 4) & (c < 4)).to(tl.float64)
        q_update = (cosv * eye + coef * tm) * q4mask

        # q_k = q_update @ q_prev  (mat-vec)
        q = tl.sum(q_update * q[None, :], axis=1)
        qn = libdevice.sqrt(tl.sum(q * q))
        q = q / qn

        # extract quat scalars
        qa = tl.sum(tl.where(i == 0, q, 0.0))
        qb = tl.sum(tl.where(i == 1, q, 0.0))
        qc = tl.sum(tl.where(i == 2, q, 0.0))
        qd = tl.sum(tl.where(i == 3, q, 0.0))

        # Cnb (3x3 embedded)
        C00 = qa*qa+qb*qb-qc*qc-qd*qd
        C01 = 2*(qb*qc+qa*qd)
        C02 = 2*(qb*qd-qa*qc)
        C10 = 2*(qb*qc-qa*qd)
        C11 = qa*qa-qb*qb+qc*qc-qd*qd
        C12 = 2*(qc*qd+qa*qb)
        C20 = 2*(qb*qd+qa*qc)
        C21 = 2*(qc*qd-qa*qb)
        C22 = qa*qa-qb*qb-qc*qc+qd*qd
        Cnb = tl.zeros((16, 16), dtype=tl.float64)
        Cnb += tl.where((r == 0) & (c == 0), C00, 0.0)
        Cnb += tl.where((r == 0) & (c == 1), C01, 0.0)
        Cnb += tl.where((r == 0) & (c == 2), C02, 0.0)
        Cnb += tl.where((r == 1) & (c == 0), C10, 0.0)
        Cnb += tl.where((r == 1) & (c == 1), C11, 0.0)
        Cnb += tl.where((r == 1) & (c == 2), C12, 0.0)
        Cnb += tl.where((r == 2) & (c == 0), C20, 0.0)
        Cnb += tl.where((r == 2) & (c == 1), C21, 0.0)
        Cnb += tl.where((r == 2) & (c == 2), C22, 0.0)

        # f_n = Cnb @ f_b ; vel update
        fbx, fby, fbz = fx, fy, fz - g
        fn0 = C00*fbx + C01*fby + C02*fbz
        fn1 = C10*fbx + C11*fby + C12*fbz
        fn2 = C20*fbx + C21*fby + C22*fbz
        # (Cnb@(f_b-g)) for vel uses f_b - [0,0,g] = (fx,fy,fz-g) -> same fn above
        velinc = tl.zeros((16,), dtype=tl.float64)
        velinc += tl.where(i == 0, fn0 * dt, 0.0)
        velinc += tl.where(i == 1, fn1 * dt, 0.0)
        velinc += tl.where(i == 2, fn2 * dt, 0.0)
        vel = vel_prev + velinc
        pos = pos_prev + vel_prev * dt

        # f_n with raw accel (for F matrix): f_n = Cnb @ f_b (no gravity sub in v1)
        rfn0 = C00*fx + C01*fy + C02*fz
        rfn1 = C10*fx + C11*fy + C12*fz
        rfn2 = C20*fx + C21*fy + C22*fz

        # odom branch
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

        # z vector (16,) lanes 0,1,2
        zvec = tl.zeros((16,), dtype=tl.float64)
        zvec += tl.where(i == 0, delta_S - delta_D, 0.0)
        zvec += tl.where(i == 1, vy, 0.0)
        zvec += tl.where(i == 2, vz, 0.0)

        # H (3x15 embedded): row0=[0,0,0,C00,C01,C02,0..], row1=[-spsi,cpsi,0,0..], row2=[0,0,1,0..]
        H = tl.zeros((16, 16), dtype=tl.float64)
        H += tl.where((r == 0) & (c == 3), C00, 0.0)
        H += tl.where((r == 0) & (c == 4), C01, 0.0)
        H += tl.where((r == 0) & (c == 5), C02, 0.0)
        H += tl.where((r == 1) & (c == 0), -spsi, 0.0)
        H += tl.where((r == 1) & (c == 1), cpsi, 0.0)
        H += tl.where((r == 2) & (c == 2), 1.0, 0.0)

        # ---- F matrix (15x15 embedded) ----
        F = eye * ((r < 15) & (c < 15)).to(tl.float64)
        # F[0:3,3:6] = I3*dt
        F += tl.where((r < 3) & (c == r + 3), dt, 0.0)
        # F[3:6,6:9] = -skew(rfn)*dt
        F += tl.where((r == 3) & (c == 7), rfn2 * dt, 0.0)
        F += tl.where((r == 3) & (c == 8), -rfn1 * dt, 0.0)
        F += tl.where((r == 4) & (c == 6), -rfn2 * dt, 0.0)
        F += tl.where((r == 4) & (c == 8), rfn0 * dt, 0.0)
        F += tl.where((r == 5) & (c == 6), rfn1 * dt, 0.0)
        F += tl.where((r == 5) & (c == 7), -rfn0 * dt, 0.0)
        # F[3:6,12:15] = -Cnb*dt
        F += tl.where((r == 3) & (c == 12), -C00 * dt, 0.0)
        F += tl.where((r == 3) & (c == 13), -C01 * dt, 0.0)
        F += tl.where((r == 3) & (c == 14), -C02 * dt, 0.0)
        F += tl.where((r == 4) & (c == 12), -C10 * dt, 0.0)
        F += tl.where((r == 4) & (c == 13), -C11 * dt, 0.0)
        F += tl.where((r == 4) & (c == 14), -C12 * dt, 0.0)
        F += tl.where((r == 5) & (c == 12), -C20 * dt, 0.0)
        F += tl.where((r == 5) & (c == 13), -C21 * dt, 0.0)
        F += tl.where((r == 5) & (c == 14), -C22 * dt, 0.0)
        # F[6:9,6:9] = -skew(w)*dt  (REPLACES block -> diagonal at 6,7,8 must be 0, not 1)
        F += tl.where((r == c) & (r >= 6) & (r < 9), -1.0, 0.0)  # cancel eye diag in this block
        F += tl.where((r == 6) & (c == 7), wz * dt, 0.0)
        F += tl.where((r == 6) & (c == 8), -wy * dt, 0.0)
        F += tl.where((r == 7) & (c == 6), -wz * dt, 0.0)
        F += tl.where((r == 7) & (c == 8), wx * dt, 0.0)
        F += tl.where((r == 8) & (c == 6), wy * dt, 0.0)
        F += tl.where((r == 8) & (c == 7), -wx * dt, 0.0)
        # F[6:9,9:12] = -I3*dt
        F += tl.where((r >= 6) & (r < 9) & (c == r + 3), -dt, 0.0)

        # ---- EKF predict ----
        FP = tl.dot(F, P)
        P_pred = tl.dot(FP, tl.trans(F)) + Qmat

        # ---- S = H P_pred H^T + R (3x3) ----
        HP = tl.dot(H, P_pred)
        S = tl.dot(HP, tl.trans(H))
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
        # Sinv (transpose of cofactor / det)
        Si = tl.zeros((16, 16), dtype=tl.float64)
        Si += tl.where((r == 0) & (c == 0), A*idet, 0.0)
        Si += tl.where((r == 0) & (c == 1), D*idet, 0.0)
        Si += tl.where((r == 0) & (c == 2), G*idet, 0.0)
        Si += tl.where((r == 1) & (c == 0), B*idet, 0.0)
        Si += tl.where((r == 1) & (c == 1), E*idet, 0.0)
        Si += tl.where((r == 1) & (c == 2), Hh*idet, 0.0)
        Si += tl.where((r == 2) & (c == 0), Cc*idet, 0.0)
        Si += tl.where((r == 2) & (c == 1), Ff*idet, 0.0)
        Si += tl.where((r == 2) & (c == 2), Ii*idet, 0.0)

        # K = P_pred H^T Sinv
        PHt = tl.dot(P_pred, tl.trans(H))
        K = tl.dot(PHt, Si)
        # x = K @ z
        x = tl.sum(K * zvec[None, :], axis=1)
        # P = (I - K H) P_pred
        KH = tl.dot(K, H)
        P = tl.dot(eye - KH, P_pred)

        # ---- error compensation ----
        x0 = tl.sum(tl.where(i == 0, x, 0.0))
        x1 = tl.sum(tl.where(i == 1, x, 0.0))
        x2 = tl.sum(tl.where(i == 2, x, 0.0))
        x3 = tl.sum(tl.where(i == 3, x, 0.0))
        x4 = tl.sum(tl.where(i == 4, x, 0.0))
        x5 = tl.sum(tl.where(i == 5, x, 0.0))
        p6 = tl.sum(tl.where(i == 6, x, 0.0))
        p7 = tl.sum(tl.where(i == 7, x, 0.0))
        p8 = tl.sum(tl.where(i == 8, x, 0.0))
        poscorr = tl.zeros((16,), dtype=tl.float64)
        poscorr += tl.where(i == 0, x0, 0.0)
        poscorr += tl.where(i == 1, x1, 0.0)
        poscorr += tl.where(i == 2, x2, 0.0)
        velcorr = tl.zeros((16,), dtype=tl.float64)
        velcorr += tl.where(i == 0, x3, 0.0)
        velcorr += tl.where(i == 1, x4, 0.0)
        velcorr += tl.where(i == 2, x5, 0.0)
        pos = pos - poscorr
        vel = vel - velcorr

        # attitude correction: dq=[1,0.5phi]; q = quatmult(q,dq)/|.|
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
        q = tl.zeros((16,), dtype=tl.float64)
        q += tl.where(i == 0, nqa, 0.0)
        q += tl.where(i == 1, nqb, 0.0)
        q += tl.where(i == 2, nqc, 0.0)
        q += tl.where(i == 3, nqd, 0.0)

        # write outputs
        tl.store(pos_out_ptr + k * 3 + 0, tl.sum(tl.where(i == 0, pos, 0.0)))
        tl.store(pos_out_ptr + k * 3 + 1, tl.sum(tl.where(i == 1, pos, 0.0)))
        tl.store(pos_out_ptr + k * 3 + 2, tl.sum(tl.where(i == 2, pos, 0.0)))
        tl.store(vel_out_ptr + k * 3 + 0, tl.sum(tl.where(i == 0, vel, 0.0)))
        tl.store(vel_out_ptr + k * 3 + 1, tl.sum(tl.where(i == 1, vel, 0.0)))
        tl.store(vel_out_ptr + k * 3 + 2, tl.sum(tl.where(i == 2, vel, 0.0)))


def run_ekf_v3(csv_path='PipeRobot_Trajectory.csv', n_steps=None, verbose=True):
    assert torch.cuda.is_available(), "v3 需要 CUDA"
    dev = torch.device('cuda')
    dtype = torch.float64

    data = np.loadtxt(csv_path, delimiter=',', skiprows=1)
    t = torch.from_numpy(data[:, 0]).to(dev)
    dt_val = float((t[1:] - t[:-1]).mean().item())
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
    qinit = torch.stack([cy*cp*cr+sy*sp*sr, cy*cp*sr-sy*sp*cr,
                         cy*sp*cr+sy*cp*sr, sy*cp*cr-cy*sp*sr]).contiguous()

    qdiag = torch.tensor([1e-6,1e-6,1e-6,1e-5,1e-5,1e-5,1e-4,1e-4,1e-4,
                          1e-8,1e-8,1e-8,1e-7,1e-7,1e-7], dtype=dtype, device=dev).contiguous()

    pos_out = torch.zeros(n, 3, dtype=dtype, device=dev).contiguous()
    vel_out = torch.zeros(n, 3, dtype=dtype, device=dev).contiguous()

    # warmup (compile)
    ekf_mega_kernel[(1,)](gyro, accel, odom1, odom2, qinit, qdiag,
                          pos_out, vel_out, min(n, 64), dt_val, g,
                          1e-4, 1e-3, 0.01, 1e12)
    torch.cuda.synchronize()
    pos_out.zero_(); vel_out.zero_()

    t_start = time.time()
    ekf_mega_kernel[(1,)](gyro, accel, odom1, odom2, qinit, qdiag,
                          pos_out, vel_out, n, dt_val, g,
                          1e-4, 1e-3, 0.01, 1e12)
    torch.cuda.synchronize()
    elapsed = time.time() - t_start

    pos_true_n = pos_true[:n]
    err = pos_out - pos_true_n
    rmse_x = (torch.sqrt((err[:, 0]**2).mean()) * 1000).item()
    rmse_y = (torch.sqrt((err[:, 1]**2).mean()) * 1000).item()
    rmse_z = (torch.sqrt((err[:, 2]**2).mean()) * 1000).item()
    l = (pos_true_n[:, 0].max() - pos_true_n[:, 0].min()).item()
    x_dist = torch.sqrt(((pos_out[-1] - pos_out[0])**2).sum()).item()
    rho = (l - x_dist) / l if l != 0 else 0.0

    metrics = {
        'version': 'v3_triton_mega', 'device': 'cuda', 'n_steps': n,
        'elapsed_s': elapsed, 'throughput_steps_per_s': (n - 1) / elapsed,
        'rmse_x_mm': rmse_x, 'rmse_y_mm': rmse_y, 'rmse_z_mm': rmse_z, 'rho_pct': rho * 100,
    }
    if verbose:
        print(f"[v3 Triton Mega] n={n} elapsed={elapsed:.4f}s "
              f"({metrics['throughput_steps_per_s']:.0f} steps/s)")
        print(f"  RMSE mm: X={rmse_x:.6f} Y={rmse_y:.6f} Z={rmse_z:.6f} | rho={rho*100:.6f}%")
    return pos_out, vel_out, pos_true_n, t[:n], metrics


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)).rsplit('/opt', 1)[0])
    run_ekf_v3(n_steps=5000)
