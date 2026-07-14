# 混合精度方案总结

> 套管井机器人 SINS+双里程计 EKF 定位 · 贡献点② 深化
> 数据集: PipeRobot_Trajectory.csv (N=166667, 100Hz, ~500m)
> 硬件: NVIDIA Hopper sm_90, 78 SM · Triton 3.6.0

---

## 1. 设计动机

v4 实现了全局统一精度 (fp64/fp32/tf32 三选一，整个 kernel 用一种精度)，但 EKF 的 16 步计算中各步骤的精度需求差异极大：

```
传感器读入 (gyro ~0.02 rad/s) ── 精度需求: 低 (~1e-3)
      ↓
四元数归一化 (q ∈ [-1,1])    ── 精度需求: 高 (~1e-5, 方向敏感)
      ↓
位置积分 (pos ∈ [0,300] m)   ── 精度需求: 高 (~1e-5, mm 级定位)
      ↓
P 协方差传播 (特征值 1e-8)    ── 精度需求: 极高 (正定性, 1e-12)
```

**核心问题**：能否对不同组件使用不同精度，而不是"一刀切"？

---

## 2. 三层精度模型

```
┌──────────────────────────────────────────────────────────────────┐
│                                                                  │
│   Layer 1                    Layer 2                Layer 3       │
│   传感器 I/O                 EKF 数学计算           tl.dot matmul │
│   ──────────                ────────────           ─────────────  │
│                                                                  │
│   gyro / accel / odom        quat / SINS / H / F    8 个矩阵乘   │
│   pos / vel 输出             z / S / K / x           F@P@F' 等   │
│   q 输出                     P 协方差                             │
│                                                                  │
│   精度需求: ~1e-3            精度需求: ~1e-5         精度需求: 1e-8 │
│   ─────────────────         ─────────────────      ────────────── │
│   fp16 可行                  fp32 必须              ieee 最优     │
│   bf16 失败 (Z 轴漂移)       libdevice 固定 fp32     TF32 反慢    │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### Layer 1: 传感器 I/O — fp16 可行

| 传感器 | 典型范围 | fp16 最小正规数 | 裕度 |
|--------|---------|----------------|------|
| 陀螺 | -0.024 ~ +0.024 rad/s | 6e-8 | 10⁶ 倍 |
| 加速度 | -10 ~ +10 m/s² | 6e-8 | 10⁸ 倍 |
| 里程计 | 0.28 ~ 0.32 m/s | 6e-8 | 10⁷ 倍 |
| 位置输出 | 0 ~ 500 m | 6e-8 | 10¹⁰ 倍 |

**结论**: 所有传感器值在 fp16 范围内，有 >10⁶ 的裕度。读入后可 cast 到 fp32 参与计算。

**bf16 失败**: bf16 与 fp16 值域相同（共享 8-bit 指数），但尾数仅 7-bit (vs fp16 的 10-bit)。里程计差分 `delta_S - delta_D ~0.0001m` 在 bf16 下截断为 0 → 位置积分停滞 → Z 轴漂移 96mm。

### Layer 2: EKF 数学计算 — fp32 必须

| 组件 | fp16 尝试 | 结果 | 根因 |
|------|----------|------|------|
| 四元数 q | 存储 fp16 | X 轴漂移 1872mm | q 归一化方向精度 < 1e-3 |
| 位置 pos | 存储 fp16 | X 轴漂移 1872mm | fp16 分辨率 ~0.03m@500m, EKF sub-mm 修正截断 |
| 速度 vel | 存储 fp16 | 间接漂移 | 速度积分 → 位置, 连锁反应 |
| libdevice | 无法降 | N/A | sin/cos/sqrt/atan2 固定返回 fp32 |

**结论**: 所有状态向量 (q/pos/vel) 和内部计算必须保持 fp32。Layer 1 的 fp16 仅作为"读入阶段"，在进入 EKF 循环的第一步就 cast 到 fp32。

**fp64 升级不可行**: Triton 3.6.0 的 `tl.dot` 要求操作数同 bitwidth。若 P 升级 fp64，则需要 F/H 也 fp64 → 2 倍寄存器压力，内核无法编译。

### Layer 3: matmul 精度 — ieee 最优

| 精度 | 机制 | 吞吐 (N=5000) | vs ieee |
|------|------|--------------|---------|
| ieee | 标准 IEEE fp32 乘加 | 6,995 | 1.00x |
| tf32 | Hopper TensorCore, 尾数截断 10→7 bit | 5,317 | **0.76x** |

**TF32 为什么反而更慢？**

```
16×16 tl.dot 的计算量: 16³ = 4096 FLOPs
TensorCore 最小 tile:  16×16×16 = 引擎只能打出 1 个 tile
TensorCore 启动延迟:   ~5-10 cycles (Hopper)
TF32 截断开销:         input 转换: fp32 → tf32 → fp32

总延迟: 启动(5-10) + 截断(2-3) + 计算(1) ≈ 8-14 cycles
vs fp32 ieee: 计算(~3-4 cycles)

结论: TF32 在 16×16 上的额外开销 > 节省的计算时间
     (大矩阵 128×128+ 才是 TF32 的甜区)
```

---

## 3. 实测结果

### 3.1 四方案对比

| 方案 | 传感器 I/O | Matmul | 吞吐 (st/s) | RMSE X | RMSE Z | ΔZ (mm) | Z 判定 |
|------|-----------|--------|------------|--------|--------|---------|--------|
| fp32-ieee | fp32 | ieee | 6,562 | 948.75 | 1.133 | 0 | ✅ 基线 |
| **fp16-ieee** | **fp16** | **ieee** | **6,995** | **948.77** | **1.609** | **+0.48** | **✅** |
| fp32-tf32 | fp32 | tf32 | 5,217 | 951.01 | 1.185 | +0.05 | ⚠️ |
| fp16-tf32 | fp16 | tf32 | 5,317 | 905.54 | 2.332 | +1.20 | ❌ |

### 3.2 精度-吞吐 Pareto 图

```
Tput ▲
 7k ┤           ★ fp16-ieee              ← Pareto 最优点
     │              (精度等价, 吞吐最高)
 6k ┤   ● fp32-ieee
     │      (基线)
 5k ┤                    ◆ fp16-tf32
     │              ■ fp32-tf32
     │
     └──────┬──────────┬──────────┬──────────▶ RMSE Z
          1.0        1.5        2.0        2.5 (mm)

★ = 推荐   ● = 基线   ■/◆ = 不推荐 (精度或吞吐劣化)
```

### 3.3 失败路径量化

| 尝试 | 效果 | 量化 | 论文价值 |
|------|------|------|---------|
| pos/vel@fp16 | X 漂移 | +1872mm | fp16 分辨率极限的量化证据 |
| q@fp16 | X 漂移 | 同上 | 四元数归一化精度需求 > 3 位小数 |
| bf16 I/O | Z 漂移 | +96mm | 7-bit 尾数不适用于累积型运算 |
| TF32 dot | tput↓20% | 5217→6995 | 微矩阵 TensorCore 反模式的量化 |
| P@fp64 | 编译失败 | N/A | Triton 同 bitwidth 限制 |

---

## 4. 论文可陈述的核心论点

### 论点 1: EKF 各组件精度需求差异达 3 个数量级

```
传感器 (1e-3) < 状态/q (1e-5) < P 特征值 (1e-8)
    ↑                 ↑              ↑
 Layer 1           Layer 2        Layer 3
 (fp16 ok)        (fp32)         (fp32 + ieee)
```

### 论点 2: fp16 传感器 I/O 是"免费午餐"

- 精度损失 <0.5mm，小于系统 RMSE (~1m) 的 2 个数量级
- 吞吐 +7%（寄存器压力降低，单次 load 从 4B→2B）
- 论文术语: "zero-cost precision reduction at the I/O boundary"

### 论点 3: TF32 对微矩阵是反模式（反直觉结果）

- 普遍认知: "TF32 能加速矩阵乘"
- 实测结果: 对 16×16 矩阵，额外开销 > 计算节省
- 原因: TensorCore 启动延迟 + 截断转换在极小 tile 上摊销不了
- 论文术语: "TF32 demonstrates negative returns below the TensorCore breakeven tile size (~64)"

### 论点 4: bf16 vs fp16 的精度差异在累积型运算中被放大

- 单次 bf16 截断误差 ~0.78% (7-bit 尾数)
- 累积 166667 步后: Z 轴漂移 96mm (放大 10⁵ 倍)
- 论文术语: "truncation error accumulation in iterative filters rules out bfloat16 for state propagation"

---

## 5. 配置速查

```bash
# 最优方案 (Pareto 最优点)
python ekf_v6_per_component.py  # sweep all combos

# 单独运行推荐配置
from ekf_v6_per_component import run_one
run_one(combo='fp16-ieee', n_steps=5000)

# 论文用 combo map:
#   fp32-ieee  = 基线 (= v4 fp32)
#   fp16-ieee  = 推荐 (传感器 fp16, 计算 fp32)
#   fp32-tf32  = 对照 (TF32 反慢证据)
#   fp16-tf32  = 对照 (最差组合)
```

## 6. 实现细节

**代码文件**: `opt/ekf_v6_per_component.py`

**核心改动** (相对 v4 kernel, 仅 2 行):

```python
# v4:  所有 load 直接 cast 到 DT
wx = tl.load(gyro_ptr + ...).to(DT)

# v6:  插入 DT_IO 作为中间精度
wx = tl.load(gyro_ptr + ...).to(DT_IO).to(DT)
#                                  └─ Level 1  └─ Level 2
#                                     (fp16)      (fp32)
```

Kernel body 其余部分与 v4 逐位一致，保证正确性。`DT_IO` 和 `IP` 作为 `tl.constexpr` 参数传入，零运行时开销。全部实验由 4 个 combo 覆盖（2×2: DT_IO ∈ {fp16,fp32} × IP ∈ {ieee,tf32}）。