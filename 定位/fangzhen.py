"""
套管井机器人轨迹仿真数据生成 (fangzhen.m -> PyTorch)
生成500m管道轨迹，模拟障碍扰动，输出IMU+双里程计数据
"""
import torch
import numpy as np
import os

def generate_trajectory(seed=42, save_path='PipeRobot_Trajectory.csv'):
    torch.manual_seed(seed)

    # 1. 物理参数
    D_pipe = 0.130
    D_robot = 0.080
    D_wheel = 0.015
    gap = (D_pipe - D_robot) / 2

    L_total = 500.0
    v_x = 0.3
    fs = 100
    dt = 1.0 / fs
    N = round(L_total / v_x * fs)
    time = torch.arange(N, dtype=torch.float64) * dt
    g = 9.81

    print(f'总仿真时间：{time[-1]:.1f} 秒 | 总点数：{N}')

    # 2. 基础轨迹生成
    pos_true = torch.zeros(3, N, dtype=torch.float64)
    vel_true = torch.zeros(3, N, dtype=torch.float64)

    pos_true[0] = v_x * time
    vel_true[0] = v_x
    pos_true[1] = 1e-4 * torch.randn(N, dtype=torch.float64)
    pos_true[2] = 1e-4 * torch.randn(N, dtype=torch.float64)
    vel_true[1] = 1e-4 * torch.randn(N, dtype=torch.float64)
    vel_true[2] = 1e-4 * torch.randn(N, dtype=torch.float64)

    # 3. 随机生成三类障碍
    obs_interval_min = 20.0
    obs_interval_max = 40.0
    obs_amplitude_min = 0.001
    obs_amplitude_max = 0.002

    obs_positions = []
    obs_types = []
    obs_amplitudes = []

    current_x = 50.0
    while current_x < L_total - 50:
        current_x += obs_interval_min + (obs_interval_max - obs_interval_min) * torch.rand(1).item()
        if current_x >= L_total:
            break
        obs_positions.append(current_x)
        obs_types.append(torch.randint(1, 4, (1,)).item())
        obs_amplitudes.append(obs_amplitude_min + (obs_amplitude_max - obs_amplitude_min) * torch.rand(1).item())

    print(f'生成障碍总数：{len(obs_positions)} 个')

    # 4. 给轨迹加入障碍扰动
    obs_width = 0.5
    for o in range(len(obs_positions)):
        x_obs = obs_positions[o]
        otype = obs_types[o]
        amp = obs_amplitudes[o]

        idx = torch.where(torch.abs(pos_true[0] - x_obs) < obs_width)[0]
        if len(idx) == 0:
            continue

        profile = amp * torch.exp(-((pos_true[0, idx] - x_obs) / 0.2) ** 2)

        if otype == 1:
            pos_true[2, idx] += profile
        elif otype == 2:
            pos_true[2, idx] -= profile
        else:
            pos_true[1, idx] += 0.6 * profile
            pos_true[2, idx] += 0.8 * profile

    # 速度由位置差分得到
    vel_true[1, 1:] = (pos_true[1, 1:] - pos_true[1, :-1]) / dt
    vel_true[2, 1:] = (pos_true[2, 1:] - pos_true[2, :-1]) / dt

    # 5. 生成 IMU + 里程计数据
    gyro = 0.005 * torch.randn(3, N, dtype=torch.float64)
    accel = torch.zeros(3, N, dtype=torch.float64)
    accel[0] = 0.01 * torch.randn(N, dtype=torch.float64)
    accel[1] = 0.01 * torch.randn(N, dtype=torch.float64)
    accel[2] = g + 0.01 * torch.randn(N, dtype=torch.float64)

    odom1 = v_x + 0.02 * torch.randn(N, dtype=torch.float64)
    odom2 = v_x + 0.02 * torch.randn(N, dtype=torch.float64)

    # 6. 保存 CSV
    data = torch.stack([
        time,
        pos_true[0], pos_true[1], pos_true[2],
        vel_true[0], vel_true[1], vel_true[2],
        gyro[0], gyro[1], gyro[2],
        accel[0], accel[1], accel[2],
        odom1, odom2
    ], dim=1)  # (N, 15)

    header = 'time_s,pos_x,pos_y,pos_z,vel_x,vel_y,vel_z,gyro_x,gyro_y,gyro_z,accel_x,accel_y,accel_z,odom1,odom2'
    np.savetxt(save_path, data.numpy(), delimiter=',', header=header, comments='')
    print(f'轨迹已保存：{save_path}')

    return time, pos_true, vel_true, gyro, accel, odom1, odom2


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    generate_trajectory()
