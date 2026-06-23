"""
套管井机器人 SINS+双里程计组合导航定位算法 (EKF.m -> PyTorch)
捷联惯导主体 + 双里程计辅助 + EKF 15维滤波 + 异常判别
"""
import torch
import numpy as np
import os


def skew(v):
    """反对称矩阵"""
    return torch.tensor([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ], dtype=torch.float64)


def quat2dcm(q):
    """四元数转方向余弦矩阵 (q = [q0, q1, q2, q3], q0为标量)"""
    q0, q1, q2, q3 = q[0], q[1], q[2], q[3]
    return torch.tensor([
        [q0**2+q1**2-q2**2-q3**2, 2*(q1*q2+q0*q3), 2*(q1*q3-q0*q2)],
        [2*(q1*q2-q0*q3), q0**2-q1**2+q2**2-q3**2, 2*(q2*q3+q0*q1)],
        [2*(q1*q3+q0*q2), 2*(q2*q3-q0*q1), q0**2-q1**2-q2**2+q3**2]
    ], dtype=torch.float64)


def quatmultiply(q1, q2):
    """四元数乘法 [q0,q1,q2,q3]"""
    a0, a1, a2, a3 = q1[0], q1[1], q1[2], q1[3]
    b0, b1, b2, b3 = q2[0], q2[1], q2[2], q2[3]
    return torch.tensor([
        a0*b0 - a1*b1 - a2*b2 - a3*b3,
        a0*b1 + a1*b0 + a2*b3 - a3*b2,
        a0*b2 - a1*b3 + a2*b0 + a3*b1,
        a0*b3 + a1*b2 - a2*b1 + a3*b0
    ], dtype=torch.float64)


def eul2quat(yaw, pitch, roll):
    """欧拉角转四元数 (ZYX顺序)"""
    cy, sy = torch.cos(yaw/2), torch.sin(yaw/2)
    cp, sp = torch.cos(pitch/2), torch.sin(pitch/2)
    cr, sr = torch.cos(roll/2), torch.sin(roll/2)
    return torch.tensor([
        cy*cp*cr + sy*sp*sr,
        cy*cp*sr - sy*sp*cr,
        cy*sp*cr + sy*cp*sr,
        sy*cp*cr - cy*sp*sr
    ], dtype=torch.float64)


def run_ekf(csv_path='PipeRobot_Trajectory.csv'):
    # 1. 加载数据
    data = np.loadtxt(csv_path, delimiter=',', skiprows=1)
    t = torch.from_numpy(data[:, 0])
    dt = (t[1:] - t[:-1]).mean().item()
    n = len(t)

    gyro = torch.from_numpy(data[:, 7:10])    # (n,3)
    accel = torch.from_numpy(data[:, 10:13])   # (n,3)
    odom1 = torch.from_numpy(data[:, 13])
    odom2 = torch.from_numpy(data[:, 14])
    pos_true = torch.from_numpy(data[:, 1:4])  # (n,3)

    g = 9.81
    delta_thresh = 0.01

    # 2. 状态初始化
    pos = torch.zeros(n, 3, dtype=torch.float64)
    vel = torch.zeros(n, 3, dtype=torch.float64)
    q = torch.zeros(n, 4, dtype=torch.float64)
    q[0] = torch.tensor([1.0, 0, 0, 0], dtype=torch.float64)

    # 初始对准
    ax0 = accel[:10, 0].mean()
    ay0 = accel[:10, 1].mean()
    az0 = accel[:10, 2].mean()
    pitch0 = torch.atan(ay0 / torch.sqrt(ax0**2 + az0**2))
    roll0 = torch.atan(-ax0 / az0)
    yaw0 = torch.tensor(0.0, dtype=torch.float64)
    q[0] = eul2quat(yaw0, pitch0, roll0)

    # EKF 15维状态
    x_ekf = torch.zeros(15, dtype=torch.float64)
    P = torch.eye(15, dtype=torch.float64) * 0.1
    Q_noise = torch.diag(torch.tensor([
        1e-6, 1e-6, 1e-6,
        1e-5, 1e-5, 1e-5,
        1e-4, 1e-4, 1e-4,
        1e-8, 1e-8, 1e-8,
        1e-7, 1e-7, 1e-7
    ], dtype=torch.float64))
    R_odo = 1e-4
    R_vcon = 1e-3

    pos_fusion = torch.zeros(n, 3, dtype=torch.float64)
    vel_fusion = torch.zeros(n, 3, dtype=torch.float64)

    # 3. 主循环
    I15 = torch.eye(15, dtype=torch.float64)
    I3 = torch.eye(3, dtype=torch.float64)
    Z33 = torch.zeros(3, 3, dtype=torch.float64)

    for k in range(1, n):
        # 步骤1：SINS 四元数姿态更新
        wx, wy, wz = gyro[k-1, 0], gyro[k-1, 1], gyro[k-1, 2]
        q_prev = q[k-1]

        theta_x = wx * dt
        theta_y = wy * dt
        theta_z = wz * dt
        theta_norm = torch.sqrt(theta_x**2 + theta_y**2 + theta_z**2)

        theta_mat = torch.tensor([
            [0, -theta_x, -theta_y, -theta_z],
            [-theta_x, 0, -theta_z, -theta_y],
            [-theta_y, -theta_z, 0, -theta_x],
            [-theta_z, -theta_y, -theta_x, 0]
        ], dtype=torch.float64)

        if theta_norm > 1e-10:
            q_update = torch.cos(theta_norm/2) * torch.eye(4, dtype=torch.float64) + \
                       (torch.sin(theta_norm/2) / theta_norm) * theta_mat
        else:
            q_update = torch.eye(4, dtype=torch.float64) + 0.5 * theta_mat

        q[k] = q_update @ q_prev
        q[k] = q[k] / q[k].norm()
        Cnb = quat2dcm(q[k])

        # 步骤2：SINS 速度更新
        f_b = accel[k-1]
        vel[k] = vel[k-1] + (Cnb @ (f_b - torch.tensor([0, 0, g], dtype=torch.float64))) * dt

        # 步骤3：SINS 位置更新
        pos[k] = pos[k-1] + vel[k-1] * dt

        # 步骤4：双里程计位移增量
        delta_D = (odom1[k] + odom2[k]) / 2 * dt
        delta_S = (pos[k] - pos[k-1]).norm()

        # 步骤5：里程计正常/异常判别 + 观测构建
        if abs(delta_D - delta_S) < delta_thresh:
            z = torch.tensor([delta_S - delta_D, vel[k, 1], vel[k, 2]], dtype=torch.float64)
            psi = torch.atan2(Cnb[0, 1], Cnb[0, 0])
            H1 = torch.zeros(3, 3, dtype=torch.float64)
            H1[1] = torch.tensor([-torch.sin(psi), torch.cos(psi), 0], dtype=torch.float64)
            H1[2] = torch.tensor([0, 0, 1], dtype=torch.float64)
            q0, q1, q2, q3 = q[k, 0], q[k, 1], q[k, 2], q[k, 3]
            H2 = torch.zeros(3, 3, dtype=torch.float64)
            H2[0] = torch.tensor([q0**2+q1**2-q2**2-q3**2, 2*(q1*q2+q0*q3), 2*(q1*q3-q0*q2)], dtype=torch.float64)
            H = torch.cat([H1, H2, Z33, Z33, Z33], dim=1)
            R = torch.diag(torch.tensor([R_odo, R_vcon, R_vcon], dtype=torch.float64))
        else:
            z = torch.tensor([vel[k, 1], vel[k, 2]], dtype=torch.float64)
            psi = torch.atan2(Cnb[0, 1], Cnb[0, 0])
            H1 = torch.zeros(2, 3, dtype=torch.float64)
            H1[0] = torch.tensor([-torch.sin(psi), torch.cos(psi), 0], dtype=torch.float64)
            H1[1] = torch.tensor([0, 0, 1], dtype=torch.float64)
            H2 = torch.zeros(2, 3, dtype=torch.float64)
            Z23 = torch.zeros(2, 3, dtype=torch.float64)
            H = torch.cat([H1, H2, Z23, Z23, Z23], dim=1)
            R = torch.diag(torch.tensor([R_vcon, R_vcon], dtype=torch.float64))

        # 步骤6：EKF 更新
        f_n = Cnb @ f_b
        F = I15.clone()
        F[0:3, 3:6] = I3 * dt
        F[3:6, 6:9] = -skew(f_n) * dt
        F[3:6, 12:15] = -Cnb * dt
        F[6:9, 6:9] = -skew(torch.tensor([wx, wy, wz], dtype=torch.float64)) * dt
        F[6:9, 9:12] = -I3 * dt

        x_ekf = F @ x_ekf
        P = F @ P @ F.T + Q_noise

        S = H @ P @ H.T + R
        K = P @ H.T @ torch.linalg.inv(S)
        x_ekf = x_ekf + K @ (z - H @ x_ekf)
        P = (I15 - K @ H) @ P

        # 步骤7：误差补偿
        pos[k] = pos[k] - x_ekf[0:3]
        vel[k] = vel[k] - x_ekf[3:6]

        phi = x_ekf[6:9]
        dq = torch.tensor([1.0, 0.5*phi[0], 0.5*phi[1], 0.5*phi[2]], dtype=torch.float64)
        dq = dq / dq.norm()
        q[k] = quatmultiply(q[k], dq)
        q[k] = q[k] / q[k].norm()

        pos_fusion[k] = pos[k]
        vel_fusion[k] = vel[k]

        x_ekf = torch.zeros(15, dtype=torch.float64)

    # 4. 误差评估
    error_fusion = pos_fusion - pos_true
    rmse_x = torch.sqrt((error_fusion[:, 0]**2).mean()) * 1000
    rmse_y = torch.sqrt((error_fusion[:, 1]**2).mean()) * 1000
    rmse_z = torch.sqrt((error_fusion[:, 2]**2).mean()) * 1000

    l = pos_true[:, 0].max() - pos_true[:, 0].min()
    x_dist = torch.sqrt(((pos_fusion[-1] - pos_fusion[0])**2).sum())
    x_error = l - x_dist
    rho = x_error / l

    print('=' * 57)
    print('           三轴位置 RMSE 误差（单位：mm）')
    print('=' * 57)
    print(f'组合定位位置 | X轴：{rmse_x:.6f} | Y轴：{rmse_y:.6f} | Z轴：{rmse_z:.6f}')
    print('=' * 57)
    print(f'定位误差 x_error：{x_error * 1000:.6f} mm')
    print(f'总行程 l = {l:.2f} m')
    print(f'相对定位精度 ρ = {rho * 100:.6f} %')
    print('=' * 57)

    return pos_fusion, vel_fusion, pos_true, t


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    run_ekf()
