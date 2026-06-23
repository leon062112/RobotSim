clear; clc; close all;

%% ====================== 1. 物理参数 ======================
D_pipe  = 0.130;     % 套管直径 130 mm
D_robot = 0.080;     % 机器人直径 80 mm
D_wheel = 0.015;     % 轮子直径 15 mm
gap = (D_pipe - D_robot)/2; % 径向间隙 25 mm

L_total = 500;       % 总长度 500 m
v_x     = 0.3;       % 前进速度 0.3 m/s
fs      = 100;       % 采样频率 100 Hz
dt      = 1/fs;      % 采样间隔
N       = round(L_total / v_x * fs); % 总点数
time    = (0:N-1)*dt;

g       = 9.81;
fprintf('总仿真时间：%.1f 秒 | 总点数：%d\n', time(end), N);

%% ====================== 2. 基础轨迹生成 ======================
pos_true = zeros(3, N);
vel_true = zeros(3, N);

% 基础 X 轴匀速运动
pos_true(1,:) = v_x * time;
vel_true(1,:) = v_x;

% Y/Z 基础值：极小噪声（几乎为0）
pos_true(2,:) = 1e-4 * randn(1, N);
pos_true(3,:) = 1e-4 * randn(1, N);
vel_true(2,:) = 1e-4 * randn(1, N);
vel_true(3,:) = 1e-4 * randn(1, N);

%% ====================== 3. 随机生成三类障碍 ======================
% 障碍间距：每 20~40 米随机出现一段
% 障碍类型：1=接箍 2=腐蚀 3=结垢
% 高度/深度：1~2 mm 随机

obs_interval_min = 20;  % 最小间距 m
obs_interval_max = 40;  % 最大间距 m
obs_amplitude_min = 0.001; % 1 mm
obs_amplitude_max = 0.002; % 2 mm

obs_positions = [];
obs_types = [];
obs_amplitudes = [];

current_x = 50; % 从 50m 开始生成障碍
while current_x < L_total - 50
    % 随机间距
    current_x = current_x + obs_interval_min + (obs_interval_max-obs_interval_min)*rand;
    if current_x >= L_total, break; end
    
    % 随机类型 & 幅值
    type = randi(3);
    amp = obs_amplitude_min + (obs_amplitude_max-obs_amplitude_min)*rand;
    
    obs_positions = [obs_positions, current_x];
    obs_types = [obs_types, type];
    obs_amplitudes = [obs_amplitudes, amp];
end

fprintf('生成障碍总数：%d 个\n', length(obs_positions));

%% ====================== 4. 给轨迹加入障碍扰动 ======================
obs_width = 0.5; % 障碍作用宽度 m

for o = 1:length(obs_positions)
    x_obs = obs_positions(o);
    type  = obs_types(o);
    amp   = obs_amplitudes(o);
    
    % 找到障碍附近的索引
    idx = find( abs(pos_true(1,:) - x_obs) < obs_width );
    if isempty(idx), continue; end
    
    % 高斯扰动轮廓
    profile = amp * exp( -( (pos_true(1,idx)-x_obs)/0.2 ).^2 );
    
    % 三类障碍
    if type == 1
        % 1：接箍 → Z 正方向
        pos_true(3, idx) = pos_true(3, idx) + profile;
    elseif type == 2
        % 2：腐蚀 → Z 负方向
        pos_true(3, idx) = pos_true(3, idx) - profile;
    else
        % 3：结垢 → Y+Z 组合扰动
        pos_true(2, idx) = pos_true(2, idx) + 0.6*profile;
        pos_true(3, idx) = pos_true(3, idx) + 0.8*profile;
    end
end

% 数值微分求速度（遇障碍产生速度）
vel_true(2,2:end) = diff(pos_true(2,:))/dt;
vel_true(3,2:end) = diff(pos_true(3,:))/dt;

%% ====================== 5. 生成 IMU + 里程计数据 ======================
gyro = 0.005*randn(3, N);   % 陀螺噪声
accel = zeros(3, N);
accel(1,:) = 0.01*randn(1,N);
accel(2,:) = 0.01*randn(1,N);
accel(3,:) = g + 0.01*randn(1,N);

odom1 = v_x + 0.02*randn(1,N);
odom2 = v_x + 0.02*randn(1,N);

%% ====================== 6. 保存 CSV ======================
data_table = table( ...
    time', ...
    pos_true(1,:)', pos_true(2,:)', pos_true(3,:)', ...
    vel_true(1,:)', vel_true(2,:)', vel_true(3,:)', ...
    gyro(1,:)', gyro(2,:)', gyro(3,:)', ...
    accel(1,:)', accel(2,:)', accel(3,:)', ...
    odom1', odom2', ...
    'VariableNames', { ...
    'time_s','pos_x','pos_y','pos_z', ...
    'vel_x','vel_y','vel_z', ...
    'gyro_x','gyro_y','gyro_z', ...
    'accel_x','accel_y','accel_z', ...
    'odom1','odom2'});

writetable(data_table, 'PipeRobot.csv');
fprintf('✅ 轨迹已保存：PipeRobot_Trajectory.csv\n');

%% ====================== 7. 绘图：轨迹可视化 ======================
figure('Position',[100,100,1100,800])

% 子图1：3D轨迹
subplot(2,3,[1,2,4,5]);
plot3(pos_true(1,:), pos_true(2,:), pos_true(3,:), 'b-', 'LineWidth',1.5);
grid on; hold on;
xlabel('X 前进方向 (m)'); ylabel('Y 横向 (m)'); zlabel('Z 径向 (m)');
title('套管机器人 3D 轨迹（含障碍扰动）');
view(45,30);
axis tight;

% 子图2：X 位移
subplot(2,3,3);
plot(time, pos_true(1,:), 'b');
grid on;
xlabel('时间 (s)'); ylabel('X (m)');
title('X 轴前进位移');

% 子图3：Y 扰动
subplot(2,3,6);
plot(time, pos_true(2,:)*1000, 'm');
grid on;
xlabel('时间 (s)'); ylabel('Y (mm)');
title('Y 横向扰动 (mm)');

% 子图4：Z 扰动（障碍最明显）
subplot(2,3,3+3);
plot(time, pos_true(3,:)*1000, 'r');
grid on;
xlabel('时间 (s)'); ylabel('Z (mm)');
title('Z 径向扰动（凸起/凹陷/接箍）');

sgtitle('套管井机器人 500米轨迹仿真（含随机障碍）','FontSize',14);

fprintf('✅ 仿真完成！轨迹图已绘制 ✅\n');