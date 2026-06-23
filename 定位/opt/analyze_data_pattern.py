"""
输入数据 Pattern / Shape / 动态性分析 (TODO #4)
为 Auto-Tuning 与 mega-kernel 设计提供依据。
"""
import numpy as np
import json
import os


def analyze(csv_path='PipeRobot_Trajectory.csv'):
    data = np.loadtxt(csv_path, delimiter=',', skiprows=1)
    n = data.shape[0]
    t = data[:, 0]
    dt = np.diff(t)
    pos_true = data[:, 1:4]
    gyro = data[:, 7:10]
    accel = data[:, 10:13]
    odom1 = data[:, 13]
    odom2 = data[:, 14]

    rep = {}
    rep['n_samples'] = int(n)
    rep['dt_mean'] = float(dt.mean())
    rep['dt_std'] = float(dt.std())
    rep['dt_is_uniform'] = bool(dt.std() / dt.mean() < 1e-6)

    # --- 异常判别分支命中率 (决定固定3维观测的合理性) ---
    g = 9.81
    delta_thresh = 0.01
    # 复现单步 delta_D vs delta_S 需完整 SINS，这里用真值近似 delta_S
    dpos = np.linalg.norm(np.diff(pos_true, axis=0), axis=1)
    delta_D = (odom1[1:] + odom2[1:]) / 2 * dt
    diff = np.abs(delta_D - dpos)
    rep['odom_branch'] = {
        'normal_ratio_approx': float((diff < delta_thresh).mean()),
        'mean_|dD-dS|': float(diff.mean()),
        'max_|dD-dS|': float(diff.max()),
        'note': '近似(用真值位移替代SINS位移); 实测主循环里几乎恒为 normal 分支'
    }

    # --- 矩阵维度/规模 ---
    rep['shapes'] = {
        'state_dim': 15, 'obs_dim_normal': 3, 'obs_dim_abnormal': 2,
        'quat_dim': 4, 'Cnb': '3x3', 'F': '15x15', 'P': '15x15',
        'H_normal': '3x15', 'S': '3x3 (or 2x2)', 'K': '15x3',
        'note': '全部为极小稠密矩阵, 最大 15x15'
    }

    # --- 稠密/稀疏 (F/H 结构) ---
    # F = I15 + 少量子块: 非零率
    F_nnz = 15 + 3*3*4  # 对角 + 5个3x3子块(其中部分)
    rep['sparsity'] = {
        'F_15x15_structural_nnz_approx': '~对角(15)+5个3x3块 → ~30%稠密, 但块状规则',
        'H_3x15_nnz': '前6列有值, 后9列全0 → 列稀疏 (惯性误差状态不被直接观测)',
        'P_15x15': '稠密 (协方差递推后填满)',
        'verdict': '矩阵太小, 稀疏优化(cuSPARSE)无意义; 块结构可手工展开'
    }

    # --- 输入数值范围 (mega-kernel 常数内联 / 精度需求) ---
    rep['ranges'] = {
        'gyro_abs_max': float(np.abs(gyro).max()),
        'accel_col_means': accel.mean(0).tolist(),
        'odom_mean': float((odom1.mean()+odom2.mean())/2),
        'odom_std': float(np.std(np.concatenate([odom1, odom2]))),
    }

    # --- 动态性 ---
    rep['dynamism'] = {
        'shape_dynamic': False,
        'batch_dynamic': False,
        'batch_size': 1,
        'note_compile_cache': 'shape全程静态(单步固定) → torch.compile/CUDAGraph缓存100%命中, 无重编译; '
                              '唯一动态是异常分支shape(2vs3), 已用固定3维+R抑制消除'
    }

    # --- Auto-Tuning 空间评估 ---
    rep['autotuning'] = {
        'tile_size': '矩阵≤15, 单block单warp即可, tile搜索空间≈无',
        'memory_layout': 'state常驻寄存器/共享内存, 无需layout搜索',
        'kernel_fusion': '最大价值: 531微kernel→1; 这是唯一高收益方向',
        'compile_cache': '静态shape, 100%命中, 无需策略',
        'verdict': '传统auto-tuning(tile/layout)空间几乎为0; 收益全在 fusion(mega-kernel) + 精度(fp64→fp32)'
    }

    print(json.dumps(rep, indent=2, ensure_ascii=False))
    os.makedirs('opt/results', exist_ok=True)
    json.dump(rep, open('opt/results/data_pattern_analysis.json', 'w'),
              indent=2, ensure_ascii=False)
    return rep


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)).rsplit('/opt', 1)[0])
    analyze()
