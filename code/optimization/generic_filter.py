"""
通用 (d,m) 状态线性高斯递推滤波（P0 壳）——供「吞吐 vs 输入 shape (d,m)」对比。

同一条线性滤波模型（predict/update）用四种方法实现，测 batch=1 吞吐：
  - eager        : torch 逐步 tensor + torch.linalg.inv
  - compile      : torch.compile 单步函数
  - torch-kf     : 领域批量 KF 库 (KalmanFilter.filter)
  - triton fused : Trident 单 kernel 融合整条 scan（m×m 求逆用高斯-约当消元，tl.static_range 展开）

d 状态维 6..30、m 观测维 2..12。所有方法吃同一 F/H/Q/R/z，做完整 predict+update。fp32。
"""
import os
import time
import numpy as np
import torch
import triton
import triton.language as tl

from torch_kf import KalmanFilter, GaussianState


def next_pow2(n):
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def build_model(d, m=3, device='cuda', seed=0):
    """固定线性模型：F=I_d 稳定、H 的 m 个观测各行观测一个状态(循环)、Q/R 对角。"""
    torch.manual_seed(seed)
    F = torch.eye(d, dtype=torch.float32, device=device)
    H = torch.zeros(m, d, dtype=torch.float32, device=device)
    for i in range(m):
        H[i, i % d] = 1.0
    Q = torch.eye(d, dtype=torch.float32, device=device) * 1e-6
    R = torch.eye(m, dtype=torch.float32, device=device) * 1e-3
    return F, H, Q, R


# ---------------------------------------------------------------------------
# eager / compile / torch-kf
# ---------------------------------------------------------------------------

def run_eager(F, H, Q, R, z, N):
    d = F.shape[0]
    x = torch.zeros(d, dtype=F.dtype, device=F.device)
    P = torch.eye(d, dtype=F.dtype, device=F.device) * 0.1
    I = torch.eye(d, dtype=F.dtype, device=F.device)
    for k in range(N):
        x = F @ x
        P = F @ P @ F.T + Q
        S = H @ P @ H.T + R
        K = P @ H.T @ torch.linalg.inv(S)
        x = x + K @ (z[k] - H @ x)
        P = (I - K @ H) @ P
    return x, P


def run_compile(F, H, Q, R, z, N=None):
    if N is None:
        N = z.shape[0]
    d = F.shape[0]
    I = torch.eye(d, dtype=F.dtype, device=F.device)

    def step(x, P, zk):
        x = F @ x
        P = F @ P @ F.T + Q
        S = H @ P @ H.T + R
        K = P @ H.T @ torch.linalg.inv(S)
        x = x + K @ (zk - H @ x)
        P = (I - K @ H) @ P
        return x, P

    step_c = torch.compile(step, dynamic=False)
    x = torch.zeros(d, dtype=F.dtype, device=F.device)
    P = torch.eye(d, dtype=F.dtype, device=F.device) * 0.1
    step_c(x, P, z[0])
    torch.cuda.synchronize()
    for k in range(N):
        x, P = step_c(x, P, z[k])
    return x, P


def run_torchkf(F, H, Q, R, z, N):
    d = F.shape[0]
    Fb = F.unsqueeze(0)
    Hb = H.unsqueeze(0)
    Qb = Q.unsqueeze(0)
    Rb = R.unsqueeze(0)
    kf = KalmanFilter(Fb, Hb, Qb, Rb)
    mean = torch.zeros(1, d, 1, dtype=F.dtype, device=F.device)
    cov = torch.eye(d, dtype=F.dtype, device=F.device).reshape(1, d, d) * 0.1
    state = GaussianState(mean, cov)
    measures = z[:N].unsqueeze(1).unsqueeze(-1)   # (N,1,m,1)
    out = kf.filter(state, measures, return_all=False)
    return out.mean, out.covariance


def run_prefix(F, H, Q, R, z, N, dtype=torch.float32):
    """
    Särkkä & García-Fernández (2021) prefix-sum 并行卡尔曼滤波。
    逐层 batched 张量实现（Hillis-Steele 扫描），所有算子走 batched cuBLAS/cuSOLVER。
    元素为 5 元组 (A, b, C, eta, J)，组合算子含 batched d×d 逆，无 Python 层内循环。
    """
    d = F.shape[0]
    dev = F.device

    # 统一到目标精度
    F_ = F.to(dtype)
    H_ = H.to(dtype)
    Q_ = Q.to(dtype)
    R_ = R.to(dtype)
    z_ = z[:N].to(dtype)
    I = torch.eye(d, dtype=dtype, device=dev)

    # 基础元素构造（时不变模型，A/C/J 对 k>0 恒定）
    S_tilde = H_ @ Q_ @ H_.T + R_
    S_tilde_inv = torch.linalg.inv(S_tilde)
    K_tilde = Q_ @ H_.T @ S_tilde_inv

    A_base = (I - K_tilde @ H_) @ F_
    C_base = (I - K_tilde @ H_) @ Q_
    J_base = F_.T @ H_.T @ S_tilde_inv @ H_ @ F_

    # Hillis-Steele 需要 2 的幂长度
    N_pad = 1 << (N - 1).bit_length()

    # 分配并填充 (N_pad, d, d) / (N_pad, d)
    A = A_base.unsqueeze(0).repeat(N_pad, 1, 1)
    b = torch.zeros(N_pad, d, dtype=dtype, device=dev)
    C = C_base.unsqueeze(0).repeat(N_pad, 1, 1)
    eta = torch.zeros(N_pad, d, dtype=dtype, device=dev)
    J = J_base.unsqueeze(0).repeat(N_pad, 1, 1)

    b[:N] = z_ @ K_tilde.T
    eta[:N] = z_ @ S_tilde_inv.T @ H_ @ F_

    # 第 0 个元素吸收先验 x0=0, P0=0.1*I
    P0 = I * 0.1
    P_pred0 = F_ @ P0 @ F_.T + Q_
    S0 = H_ @ P_pred0 @ H_.T + R_
    K0 = P_pred0 @ H_.T @ torch.linalg.inv(S0)
    A[0] = 0.0
    b[0] = K0 @ z_[0]
    C[0] = (I - K0 @ H_) @ P_pred0
    eta[0] = 0.0
    J[0] = 0.0

    # 末尾 padding 用恒等元 (A=I, b=0, C=0, eta=0, J=0)
    if N_pad > N:
        A[N:] = I
        C[N:] = 0.0
        J[N:] = 0.0

    offset = 1
    while offset < N_pad:
        # 左操作数 j = [0 : N_pad-offset]，右操作数 i = [offset : N_pad]
        A_j = A[:-offset]
        b_j = b[:-offset]
        C_j = C[:-offset]
        eta_j = eta[:-offset]
        J_j = J[:-offset]

        A_i = A[offset:]
        b_i = b[offset:]
        C_i = C[offset:]
        eta_i = eta[offset:]
        J_i = J[offset:]

        # (I + C_j J_i) 与 (I + J_i C_j) 的 batched 逆
        M1 = I.unsqueeze(0) + torch.bmm(C_j, J_i)
        inv_j = torch.linalg.inv(M1)
        M2 = I.unsqueeze(0) + torch.bmm(J_i, C_j)
        inv_i = torch.linalg.inv(M2)

        # 关联组合算子（Särkkä 2021, Eq. 19）
        A_ij = torch.bmm(torch.bmm(A_i, inv_j), A_j)

        Cj_etai = torch.bmm(C_j, eta_i.unsqueeze(-1))
        b_j_plus = b_j.unsqueeze(-1) + Cj_etai
        b_ij = torch.bmm(torch.bmm(A_i, inv_j), b_j_plus).squeeze(-1) + b_i

        C_ij = torch.bmm(torch.bmm(torch.bmm(A_i, inv_j), C_j), A_i.transpose(1, 2)) + C_i

        Ji_bj = torch.bmm(J_i, b_j.unsqueeze(-1))
        eta_diff = eta_i.unsqueeze(-1) - Ji_bj
        eta_ij = torch.bmm(torch.bmm(A_j.transpose(1, 2), inv_i), eta_diff).squeeze(-1) + eta_j

        J_ij = torch.bmm(torch.bmm(A_j.transpose(1, 2), torch.bmm(inv_i, J_i)), A_j) + J_j

        # 并行写回（Hillis-Steele 语义：所有元素基于上一层的值同时更新）
        A[offset:] = A_ij
        b[offset:] = b_ij
        C[offset:] = C_ij
        eta[offset:] = eta_ij
        J[offset:] = J_ij

        offset *= 2

    # 最终滤波均值与协方差在第 N-1 个前缀处
    return b[N - 1], C[N - 1]


# ---------------------------------------------------------------------------
# Trident 融合 Triton 内核（单 launch 跑完整 N 步）
# F/H/Q/R 在 host 已 pad 到 T×T；d×d / m×m 块是真实滤波，pad 块因零填充恒为 0。
# m×m 求逆：高斯-约当消元（无选主元，S 为 SPD），tl.static_range 展开，O(m^3)。
# ---------------------------------------------------------------------------

@triton.jit
def lin_scan_kernel(
    F_ptr, H_ptr, Q_ptr, R_ptr, z_ptr, pout_ptr,
    N, M: tl.constexpr, T: tl.constexpr,
):
    i = tl.arange(0, T)
    j = tl.arange(0, T)
    r = i[:, None]
    c = j[None, :]
    eye = (r == c).to(tl.float32)
    mmask = ((r < M) & (c < M)).to(tl.float32)

    F = tl.load(F_ptr + r * T + c)
    H = tl.load(H_ptr + r * T + c)   # 前 M 行 = H，其余零
    Q = tl.load(Q_ptr + r * T + c)
    Rm = tl.load(R_ptr + r * T + c)  # m×m 块 + 零 pad

    P = eye * 0.1
    x = tl.zeros((T,), dtype=tl.float32)

    for k in range(0, N):
        # predict
        x = tl.sum(F * x[None, :], axis=1)
        FP = tl.dot(F, P)
        P = tl.dot(FP, tl.trans(F)) + Q

        # S = H P H^T + R  (m×m 块)
        HP = tl.dot(H, P)
        S = tl.dot(HP, tl.trans(H)) + Rm

        # ---- 高斯-约当消元求 S^-1 (m×m 块)，每枢轴一次 rank-1 更新，O(m·d²) ----
        A = S
        B = eye * mmask
        for kk in tl.static_range(M):
            piv = tl.sum(tl.where((r == kk) & (c == kk), A, 0.0))
            invp = 1.0 / piv
            rk = (r == kk).to(tl.float32)
            ck = (c == kk).to(tl.float32)
            # normalize row kk
            A = A * (1.0 - rk) + (A * invp) * rk
            B = B * (1.0 - rk) + (B * invp) * rk
            # eliminate column kk：vec = A[:,kk] - e_kk（vec[kk]=0）
            colk = tl.sum(A * ck, axis=0)            # (T,) 列 kk（已归一化，colk[kk]=1）
            vec = colk - (i == kk).to(tl.float32)    # (T,)
            rowkA = tl.sum(A * rk, axis=1)           # (T,) 行 kk
            rowkB = tl.sum(B * rk, axis=1)
            A = A - vec[:, None] * rowkA[None, :]
            B = B - vec[:, None] * rowkB[None, :]
        Si = B

        # K = P H^T S^-1 ; update mean + cov
        PHt = tl.dot(P, tl.trans(H))
        K = tl.dot(PHt, Si)
        zvec = tl.load(z_ptr + k * M + i, mask=i < M, other=0.0)
        hx = tl.sum(H * x[None, :], axis=1)      # T-vec，前 M 个 = H @ x
        innov = zvec - hx
        x = x + tl.sum(K * innov[None, :], axis=1)
        KH = tl.dot(K, H)
        P = tl.dot(eye - KH, P)

    tr = tl.sum(P * eye)
    tl.store(pout_ptr + 0, tr)
    tl.store(pout_ptr + 1, tl.sum(tl.where(i == 0, x, 0.0)))
    tl.store(pout_ptr + 2, tl.sum(tl.where(i == 1, x, 0.0)))


def run_triton(F, H, Q, R, z, N, warmup=True):
    d = F.shape[0]
    m = H.shape[0]
    T = max(next_pow2(d), 16)
    dev = F.device
    Fpad = torch.zeros(T, T, dtype=torch.float32, device=dev)
    Fpad[:d, :d] = F
    Hpad = torch.zeros(T, T, dtype=torch.float32, device=dev)
    Hpad[:m, :d] = H
    Qpad = torch.zeros(T, T, dtype=torch.float32, device=dev)
    Qpad[:d, :d] = Q
    Rpad = torch.zeros(T, T, dtype=torch.float32, device=dev)
    Rpad[:m, :m] = R
    zc = z[:N].contiguous()
    pout = torch.zeros(4, dtype=torch.float32, device=dev)

    grid = (1,)
    if warmup:
        lin_scan_kernel[grid](Fpad, Hpad, Qpad, Rpad, zc, pout, min(N, 32), m, T)
        torch.cuda.synchronize()
        pout.zero_()
    lin_scan_kernel[grid](Fpad, Hpad, Qpad, Rpad, zc, pout, N, m, T)
    torch.cuda.synchronize()
    return pout