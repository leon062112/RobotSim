# MATLAB → PyTorch 迁移文档

## 概述

将套管井机器人多传感融合定位仿真从 MATLAB 迁移到 PyTorch，便于后续基于 GPU 加速和深度学习框架进行优化研究。

## 环境

```bash
conda activate selfcross_env
# PyTorch 2.1.2+cu121, Python 3.x
```

## 文件对应关系

| MATLAB 源文件 | PyTorch 文件 | 功能 |
|--------------|-------------|------|
| `定位/fangzhen.m` | `定位/fangzhen.py` | 轨迹仿真数据生成 |
| `定位/EKF.m` | `定位/ekf.py` | SINS/EKF 融合定位 |

## 运行方式

```bash
cd 定位/
python fangzhen.py   # 生成 PipeRobot_Trajectory.csv
python ekf.py        # 运行融合定位，输出 RMSE
```

## 功能对齐逐项检查

### fangzhen.m → fangzhen.py

| 功能模块 | MATLAB | PyTorch | 状态 |
|----------|--------|---------|------|
| 物理参数 | D_pipe=0.13, v_x=0.3, fs=100, N=166667 | 相同 | ✅ |
| 基础轨迹 | pos_x = v_x*time, Y/Z加1e-4噪声 | 相同 | ✅ |
| 障碍生成 | while循环, 间距20-40m, 幅值1-2mm | 相同 | ✅ |
| 障碍扰动 | find全序列搜索 + 高斯profile | torch.where + 相同profile | ✅ |
| 三类障碍 | 接箍+Z, 腐蚀-Z, 结垢0.6Y+0.8Z | 相同 | ✅ |
| Y/Z速度 | diff(pos)/dt | 相同差分 | ✅ |
| IMU噪声 | gyro=0.005*randn, accel_z=g+0.01*randn | 相同 | ✅ |
| 里程计 | v_x + 0.02*randn | 相同 | ✅ |
| CSV输出 | writetable 15列 | np.savetxt 15列 | ✅ |
| 绘图 | figure + subplot | 未迁移（计算-可视化解耦） | ⚠️ |

### EKF.m → ekf.py

| 功能模块 | MATLAB | PyTorch | 状态 |
|----------|--------|---------|------|
| 数据加载 | readtable | np.loadtxt | ✅ |
| 初始对准 | atan(ay/sqrt(ax²+az²)), atan(-ax/az) | 相同 | ✅ |
| eul2quat | MATLAB内置, ZYX顺序 | 自实现, 相同公式 | ✅ |
| 四元数更新矩阵 theta_mat | 对称4×4矩阵 | 相同结构 | ✅ |
| 小角度近似 | eye(4)+0.5*theta_mat | 相同 | ✅ |
| quat2dcm | MATLAB内置 | 自实现, 相同公式 | ✅ |
| SINS速度更新 | Cnb*(f_b-[0;0;g])*dt | 相同 | ✅ |
| SINS位置更新 | pos(k-1)+vel(k-1)*dt | 相同 | ✅ |
| 里程计位移 | (odom1+odom2)/2*dt | 相同 | ✅ |
| 异常判别阈值 | delta=0.01 | 相同 | ✅ |
| 正常模式观测 | [dS-dD; vel_y; vel_z], H=3×15 | 相同 | ✅ |
| 异常模式观测 | [vel_y; vel_z], H=2×15 | 相同 | ✅ |
| F矩阵构造 | eye(15) + 子块赋值 | 相同 | ✅ |
| F(7:9,7:9) | -skew([wx,wy,wz])*dt (覆盖eye) | 相同 | ✅ |
| EKF预测 | F*x, F*P*F'+Q | 相同 | ✅ |
| Kalman增益 | P*H'/(H*P*H'+R) | P@H.T@inv(S) | ✅ |
| 误差补偿 | pos-=x(1:3), vel-=x(4:6) | 相同 | ✅ |
| 姿态补偿 | dq=[1;0.5*phi], quatmultiply | 相同 | ✅ |
| 状态归零 | x_ekf=zeros(15,1) | 相同 | ✅ |
| RMSE计算 | sqrt(mean(error.^2))*1000 | 相同 | ✅ |
| 闭合误差 | l - norm(end-start) | 相同 | ✅ |
| 绘图 | figure + subplot | 未迁移 | ⚠️ |

## 验证结果

使用相同参数（seed=42）生成数据并运行 EKF：

```
组合定位位置 | X轴：1017.84 mm | Y轴：16.83 mm | Z轴：8.14 mm
相对定位精度 ρ = 0.1755 %
```

Y/Z 轴 RMSE 在 10-17mm 量级，相对定位精度 0.18%，符合预期。

## 迁移中发现的差异与修正

| 问题 | 说明 | 处理 |
|------|------|------|
| F(7:9,7:9) 子块 | MATLAB用`-skew*dt`覆盖eye(3)对角块；初版Python误写为`I3 - skew*dt` | 已修正为与MATLAB一致 |
| 随机种子 | MATLAB无固定种子；Python使用`torch.manual_seed(42)` | Python版可复现 |
| 绘图功能 | MATLAB内嵌绘图代码 | Python版解耦，后续按需添加 |
| CSV文件名 | MATLAB代码writetable写`PipeRobot.csv`但print提示`PipeRobot_Trajectory.csv` | Python统一为`PipeRobot_Trajectory.csv` |

## 接口设计

```python
# fangzhen.py
from fangzhen import generate_trajectory
time, pos_true, vel_true, gyro, accel, odom1, odom2 = generate_trajectory(seed=42)

# ekf.py
from ekf import run_ekf
pos_fusion, vel_fusion, pos_true, t = run_ekf('PipeRobot_Trajectory.csv')
```

所有核心数据使用 `torch.float64` 张量，可直接用于后续 GPU 加速（`.cuda()`）或梯度计算（`.requires_grad_()`）。
