%% 套管井机器人 SINS+双里程计组合导航定位算法
% 捷联惯导主体 + 双里程计辅助 + EKF 15维滤波 + 异常判别
clear; clc; close all;

%% 1. 加载数据
data = readtable('PipeRobot_Trajectory.csv');
t       = data.time_s;
dt      = mean(diff(t)); % 采样时间
gyro    = [data.gyro_x, data.gyro_y, data.gyro_z];    % 角速度
accel   = [data.accel_x, data.accel_y, data.accel_z]; % 加速度
odom1   = data.odom1;   % 里程计1速度
odom2   = data.odom2;   % 里程计2速度
pos_true = [data.pos_x, data.pos_y, data.pos_z]; % 真值

%% 2. 算法参数初始化
g = 9.81;          % 重力加速度
n = length(t);     % 数据长度
delta = 0.01;      % 里程计异常判别阈值

% SINS 状态初始化
pos     = zeros(n,3);  % 位置 xyz
vel     = zeros(n,3);  % 速度
q       = zeros(n,4);  % 四元数 q0 q1 q2 q3
q(1,:)  = [1,0,0,0];  % 初始四元数

% 初始对准：静止，利用加速度计解俯仰/横滚，航向=0
ax0 = mean(accel(1:10,1));
ay0 = mean(accel(1:10,2));
az0 = mean(accel(1:10,3));
pitch0 = atan(ay0 / sqrt(ax0^2 + az0^2));   % 俯仰
roll0  = atan(-ax0 / az0);                 % 横滚
yaw0   = 0;                                % 航向

% 四元数初始化（欧拉角转四元数）
q(1,:) = eul2quat([yaw0, pitch0, roll0], 'ZYX');

% EKF 15维状态：[位置误差(3), 速度误差(3), 失准角(3), 陀螺零偏(3), 加计零偏(3)]
x_ekf = zeros(15, 1);
P = eye(15) * 0.1;       % 协方差
Q = diag([
    1e-6, 1e-6, 1e-6, ...    % δP
    1e-5, 1e-5, 1e-5, ...    % δV
    1e-4, 1e-4, 1e-4, ...    % φ (失准角)
    1e-8, 1e-8, 1e-8, ...    % ε_g (陀螺零偏)
    1e-7, 1e-7, 1e-7 ...     % ∇_a (加计零偏)
]);
R_odo  = 1e-4;
R_vcon = 1e-3;            
bias_gyro = zeros(1,3);  % 陀螺零偏估计值 [bgx, bgy, bgz]
bias_acc  = zeros(1,3);  % 加计零偏估计值 [bax, bay, baz]
% 输出保存
pos_sins  = zeros(n,3);
vel_sins  = zeros(n,3);
pos_fusion= zeros(n,3);
vel_fusion= zeros(n,3);

%% 3. 主循环：SINS解算 + EKF融合
for k = 2:n
    % ====================== 步骤1：SINS 四元数姿态更新 ======================
      wx = gyro(k-1,1); wy = gyro(k-1,2); wz = gyro(k-1,3);
    q_prev = q(k-1,:)';
    
    % 计算角增量
    theta_x = wx * dt;
    theta_y = wy * dt;
    theta_z = wz * dt;
    theta_norm = sqrt(theta_x^2 + theta_y^2 + theta_z^2);
    
    % 构建角增量矩阵θ
    theta_mat = [
        0, -theta_x, -theta_y, -theta_z;
        -theta_x, 0, -theta_z, -theta_y;
        -theta_y, -theta_z, 0, -theta_x;
        -theta_z, -theta_y, -theta_x, 0
    ];
    
    % 计算四元数更新矩阵
    if theta_norm > 1e-10
        q_update = cos(theta_norm/2)*eye(4) + (sin(theta_norm/2)/theta_norm)*theta_mat;
    else
        % 小角度近似，避免除以0
        q_update = eye(4) + 0.5*theta_mat;
    end
    
    % 更新四元数并归一化
    q(k,:) = (q_update * q_prev)';
    q(k,:) = q(k,:) / norm(q(k,:));
    Cnb = quat2dcm(q(k,:));

    % ====================== 步骤2：SINS 速度更新 ======================
    f_b = accel(k-1,:)';    
    vel(k,:) = vel(k-1,:) + (Cnb * (f_b - [0;0;g]))' * dt;

    % ====================== 步骤3：SINS 位置更新 ======================
    pos(k,:) = pos(k-1,:) + vel(k-1,:) * dt;

    % 保存纯惯导结果
    pos_sins(k,:) = pos(k,:);
    vel_sins(k,:) = vel(k,:);

    % ====================== 步骤4：双里程计位移增量 ======================
    v_odom1 = odom1(k);
    v_odom2 = odom2(k);
    delta_D = (v_odom1 + v_odom2)/2 * dt;          % 平均位移
    delta_S = norm(pos(k,:) - pos(k-1,:));         % SINS位移

    % ====================== 步骤5：里程计正常/异常判别 ======================
    if abs(delta_D - delta_S) < delta
        % 正常：观测 = 位移差 + y/z速度约束 (3维观测)
        z = [delta_S - delta_D; vel(k,2); vel(k,3)];
         % 航向角 ψ
        psi = atan2(Cnb(1,2), Cnb(1,1));
        % H1: 3x3 位置误差部分
        H1 = zeros(3,3);
        H1(1,:) = [0, 0, 0];
        H1(2,:) = [-sin(psi), cos(psi), 0];
        H1(3,:) = [0, 0, 1];
        % H2: 3x3 速度误差部分
        q0 = q(k,1); q1 = q(k,2); q2 = q(k,3); q3 = q(k,4);
        H2 = zeros(3,3);
        H2(1,:) = [q0^2+q1^2-q2^2-q3^2, 2*(q1*q2+q0*q3), 2*(q1*q3-q0*q2)];
        H2(2,:) = [0, 0, 0];
        H2(3,:) = [0, 0, 0];
        % 完整观测矩阵 H = [H1 H2 0 0 0]
        H = [H1, H2, zeros(3,3), zeros(3,3), zeros(3,3)];
        R = diag([R_odo, R_vcon, R_vcon]);     % 3维观测噪声
    else
        % 异常：仅y/z速度约束 (2维观测)
        z = [vel(k,2); vel(k,3)];
        % H1, H2 对应2行
        psi = atan2(Cnb(1,2), Cnb(1,1));
        H1 = zeros(2,3);
        H1(1,:) = [-sin(psi), cos(psi), 0];
        H1(2,:) = [0, 0, 1];
        q0 = q(k,1); q1 = q(k,2); q2 = q(k,3); q3 = q(k,4);
        H2 = zeros(2,3);
        H2(1,:) = [0,0,0];
        H2(2,:) = [0,0,0];
        H = [H1, H2, zeros(2,3), zeros(2,3), zeros(2,3)];
        R = diag([R_vcon, R_vcon]);    % 2维观测噪声
    end

    % ====================== 步骤6：EKF 更新 ======================
   f_n = Cnb * f_b;
    F = eye(15);

    % 1. 位置误差 δP
    F(1:3, 4:6) = eye(3)*dt;              %速度误差耦合

    % 2. 速度误差 δV
    F(4:6, 7:9)  = -skew(f_n) * dt;       % 失准角耦合
    F(4:6, 13:15) = -Cnb * dt;            % 加计零偏耦合

    % 3. 失准角 φ
    F(7:9, 7:9)  = -skew([wx, wy, wz])*dt;%失准角自身更新
    F(7:9, 10:12) = -eye(3)*dt;           % 陀螺零偏耦合

    % 4. 陀螺零偏 ε_g
    F(10:12, 10:12) = eye(3);

    % 5. 加计零偏 ∇_a
    F(13:15, 13:15) = eye(3);

    % EKF 预测
    x_ekf = F * x_ekf;
    P = F * P * F' + Q;

    % EKF 更新
    K = P * H' / (H * P * H' + R);
    x_ekf = x_ekf + K * (z - H * x_ekf);
    P = (eye(15) - K * H) * P;

    % ====================== 步骤7：误差补偿 ======================
    pos(k,:) = pos(k,:) - x_ekf(1:3)';
    vel(k,:) = vel(k,:) - x_ekf(4:6)';

    % 姿态误差补偿
    phi = x_ekf(7:9);
    dq = [1; 0.5*phi];
    dq = dq / norm(dq);
    q(k,:) = (quatmultiply(q(k,:), dq')) / norm(quatmultiply(q(k,:), dq'));

    % 保存融合结果
    pos_fusion(k,:) = pos(k,:);
    vel_fusion(k,:) = vel(k,:);

    % 误差归零
    x_ekf = zeros(15,1);
end

figure('Color','w','Position',[100,100,900,600])

% X 轴位置
subplot(3,1,1)
plot(t, pos_true(:,1)*1000, 'k-', 'LineWidth',1.8); hold on;
plot(t, pos_fusion(:,1)*1000,'b-', 'LineWidth',2.0);
title('X 轴位置（前进方向）','FontSize',12)
ylabel('位置 / mm','FontSize',11)
legend('真值','融合定位','Location','best');
grid on;

% Y 轴位置
subplot(3,1,2)
plot(t, pos_true(:,2)*1000, 'k-', 'LineWidth',1.8); hold on;
plot(t, pos_fusion(:,2)*1000,'b-', 'LineWidth',2.0);
title('Y 轴径向位置','FontSize',12)
ylabel('位置 / mm','FontSize',11)
grid on;

% Z 轴位置
subplot(3,1,3)
plot(t, pos_true(:,3)*1000, 'k-', 'LineWidth',1.8); hold on;
plot(t, pos_fusion(:,3)*1000,'b-', 'LineWidth',2.0);
title('Z 轴径向位置','FontSize',12)
xlabel('时间 / s','FontSize',11)
ylabel('位置 / mm','FontSize',11)
grid on;

%% 5.计算闭合误差与定位精度
% 计算每一刻的误差（单位：m）
error_fusion= pos_fusion - pos_true; % 组合定位误差

% ====================== RMSE 均方根误差======================
% 组合定位 RMSE
rmse_fusion_x = sqrt(mean(error_fusion(:,1).^2)) * 1000;
rmse_fusion_y = sqrt(mean(error_fusion(:,2).^2)) * 1000;
rmse_fusion_z = sqrt(mean(error_fusion(:,3).^2)) * 1000;
% （1）提取起点和终点坐标（组合定位结果）
x0 = pos_fusion(1,1); y0 = pos_fusion(1,2); z0 = pos_fusion(1,3);
x1 = pos_fusion(end,1); y1 = pos_fusion(end,2); z1 = pos_fusion(end,3);

% 计算定位误差（单位：m）
x = sqrt( (x1 - x0)^2 + (y1 - y0)^2 + (z1 - z0)^2 );
% 计算总行程 l（单位：m）
l = max(pos_true(:,1)) - min(pos_true(:,1));
x_error=l-x;
% 计算相对定位精度
rho = x_error / l;

fprintf('=========================================================\n');
fprintf('           三轴位置 RMSE 误差（单位：mm）\n');
fprintf('=========================================================\n');
fprintf('组合定位位置 | X轴：%.6f | Y轴：%.6f | Z轴：%.6f \n',rmse_fusion_x,rmse_fusion_y,rmse_fusion_z);
fprintf('=========================================================\n');
fprintf('起点-终点闭合误差 结果\n');
fprintf('=========================================================\n');
fprintf('定位误差 x_error：%.6f mm\n', x_error * 1000);
fprintf('总行程 l = %.2f m\n', l);
fprintf('相对定位精度 ρ = %.6f %% \n', rho * 100);
fprintf('=========================================================\n');
%% 辅助函数：反对称矩阵
function M = skew(v)
    M = [ 0, -v(3),  v(2);
         v(3),   0, -v(1);
        -v(2), v(1),   0];
end