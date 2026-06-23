# EKF 高性能优化 — 最终分析报告

> 套管井机器人 SINS+双里程计 EKF 组合导航定位 · GPU 高性能优化
> 数据集: PipeRobot_Trajectory.csv (N=166667 步, 100Hz, float64)

---

## 0. 执行摘要

| 版本 | 方法 | 设备 | 吞吐 | 相对基线 | 正确性 |
|------|------|------|------|---------|-------|
| v0 | 原始 eager | CPU | 1820 步/s | 1.00x | 金标准 |
| v1 | 消碎片 | CPU | 2060 步/s | 1.13x | 等价 |
| v2 | CUDA Graph | GPU | 2702 步/s | 1.48x | 等价 |
| **v3** | **Triton 单 kernel Mega** | **GPU** | **20534 步/s** | **11.3x** | 亚毫米等价 |

**严格串行 (卡尔曼递归) + 极小矩阵 (≤15×15) + 海量微算子**的
latency-bound 负载, 对朴素 GPU 并行极不友好 (eager GPU 比 CPU 还慢 4 倍)。
唯一有效的优化主线是 **kernel fusion 的三级递进**, 最终单 kernel 实现 11.3x 加速。

---

## 1. EKF 性能瓶颈分析

### 1.1 算法结构
每个时间步串行执行 7 个阶段: 四元数姿态更新 → SINS 速度/位置积分 →
双里程计位移 → 异常判别 → 构建 H/R → EKF 预测+更新 (15维) → 误差补偿。
步 k 强依赖步 k-1 的 (pos, vel, q, P) ⇒ **时间维度无法并行**。

### 1.2 Eager 模式 Profiling (torch.profiler, CPU n=2000)
见 `profiles/v0_cpu_n2000_summary.txt`。

- **每步 ~1026 个算子调用**, 总计 205 万次 op / 2000 步
- 真正的线性代数 (inv / lu_solve / mm / mv) 仅占 ~15-20% 时间
- 时间大头是**碎片化小算子**:
  | 算子 | 调用次数 | 占比 | 来源 |
  |------|---------|------|------|
  | aten::select | 161932 | 14.2% | `q[k]`, `gyro[k-1,0]` 等逐元素索引 |
  | aten::mul | 111964 | 11.8% | 标量乘 |
  | aten::item / _local_scalar_dense | 各 111954 | ~5% | **CPU↔Python 标量同步** |
  | aten::pow / copy_ | 38k / 72k | ~12% | 现场张量构造 |

- **根因**: `skew/quat2dcm/quatmultiply/eul2quat` 及 `theta_mat/H/F` 全部用
  `torch.tensor([[标量,...]])` **现场构造** —— 每个标量都触发 select+item+mul 链。

### 1.3 GPU 瓶颈 (v2 CUDA Graph, 单步 CUDA kernel 统计)
- **每步 531 个 CUDA kernel launch**, 817 us GPU 时间
- mul 75/步, cat 31/步, sub 32/步, neg 27/步; cutlass gemm 仅占 ~20%
- 微 kernel 间 ~0.7us 转换延迟串行累积 ⇒ 单步 latency ≈ 531 × 间隙, **算力几乎闲置**
- 这解释了为何 eager GPU 比 CPU 更慢: launch + .item() 同步开销 >> 微小算力收益

### 1.4 算子库参照 (揭示本质)
- cuBLAS gemm(F@P@Fᵀ): **17.6 us/步** · cuSOLVER inv(3×3): **73.5 us/步**
- 仅 cuSOLVER 单次求逆 (73.5us) 就 > v3 整步 (~48.7us)
- ⇒ 通用库为大矩阵设计, 在 15×15 规模下其固定开销 (launch/同步) 主导, 成为反向瓶颈

---

## 2. Torch Compile 融合效果分析

### 2.1 Graph Break 分析
v0 原始实现无法编译 (3 类 graph break): `.item()` 隐式同步、数据相关 shape
(正常3维/异常2维观测)、python 标量构造 tensor。

v1 重构后用 `torch._dynamo.explain` 实测单步 `ekf_step`:
> **graph_count=1, graph_break_count=0, op_count=241** —— 完整捕获, 零 break。
(FX 图层面 op 从 v0 的 1026/步 降至 241/步, -76%)

### 2.2 融合效果与反直觉结论
| 配置 | 吞吐 | 结论 |
|------|------|------|
| v1 compile(default) CPU | 306 步/s | **比 eager 慢 6.5x** |
| v1 compile(default) GPU | 2233 步/s | 仅 1.23x |
| v1 compile reduce-overhead | 失败 | CUDAGraph 与循环内 state 读写冲突 |

- **torch.compile(default) 对超小单步图收益有限甚至为负**: inductor codegen +
  守卫检查开销, 在 166667 次循环放大, 超过 tiny step 的融合收益。
- reduce-overhead (内置 CUDA Graph) 报 "tensor output overwritten" —— 因 state
  张量循环复用; 需手动管理静态 buffer (即 v2 的做法)。
- **启示**: 自动编译器擅长大 kernel 融合, 但对"极小图 + 超长串行循环"需手动 CUDA Graph / 自定义 kernel。

---

## 3. Mega Kernel 可行性与实现


| 目标 | v0→v3 改善 |
|------|-----------|
| 减少中间 Tensor 读写 | state(pos/vel/q/P) 常驻寄存器, 跨步零 global 往返 |
| 减少 Kernel Launch | N×531 → **1 次** launch |
| 提高数据局部性 | 全部 padded-16 矩阵在 register/SRAM 内计算 |
| 提升 GPU 利用率 | 单 block 串行受限, 但消除了 launch/同步空泡 |

### 3.1 两级 Mega Kernel
- **v2 (CUDA Graph)**: 把"单步图"捕获为 1 个 graph, N 次 replay。消除 CPU 调度开销
  → 1.48x。但 GPU 仍串行执行 531 微 kernel, 未触及本质。unroll 1→50 吞吐不变,
  证明已非 launch-bound 而是 **GPU-execution-bound**。
- **v3 (Triton 单 kernel)**: 整个 N 步 scan 写进 1 个 Triton kernel。531 微 kernel
  →融合为 1 个, 彻底消除 kernel 间转换延迟 → **11.3x**。

### 3.2 关键技术 (踩坑记录)
- 3×3 求逆: cuSOLVER 在 CUDA Graph capture 中触发 host 同步而失败 → 改 **解析 adjugate 逆**
- 矩阵表示: 全部 padded 到 16×16, 用 `tl.dot`(fp64) 做 matmul, mask 构造/抽取标量
- 数学函数: `libdevice.{atan2,sin,cos,sqrt,abs}` (非 tl.math)
- 异常分支: 固定 3 维观测 + 异常时 `R_odo→1e12` 抑制里程计通道 (K≈0), 与原"删观测行"等价, 消除动态 shape
- GPU 标量索引在 capture 中触发同步 → 用 `index_select` + 1-elem idx 张量

---

## 4. Auto-Tuning 与输入特征分析

见 `results/data_pattern_analysis.json`。

### 4.1 数据 Pattern
- dt 严格均匀 (std/mean < 1e-6), 单 batch
- 异常判别命中率: normal ≈ 100% (|ΔD-ΔS| max=0.0007 ≪ 阈值 0.01) → 固定3维观测合理
- 矩阵: 全部 ≤15×15 稠密小矩阵; H 后 9 列恒 0 (列稀疏) 但规模太小, cuSPARSE 无意义

### 4.2 Shape / 动态性
- **shape 全程静态, batch 静态 (=1)** → torch.compile / CUDA Graph 缓存 100% 命中, 无重编译
- 唯一动态点 (异常分支 2维/3维) 已被"固定3维+R抑制"消除

### 4.3 Auto-Tuning 方向评估
| 方向 | 空间 | 结论 |
|------|------|------|
| Tile Size | ≈0 | 矩阵≤15, 单 warp 足够 |
| Memory Layout | ≈0 | state 驻寄存器, 无需搜索 |
| Kernel Fusion | **极大** | 531→1, 唯一高收益方向 (已由 v3 实现) |
| 编译缓存策略 | ≈0 | 静态 shape, 天然 100% 命中 |

**结论**: 传统 auto-tuning (tile/layout/搜索) 在此问题空间几乎为 0;
收益全部集中在 **kernel fusion** (已落地) 和 **精度选择 fp64→fp32** (未来方向)。

---

## 5. Baseline 对比实验 (TODO #5)

见 `results/benchmark_summary.json` (统一 RMSE 校验 + 吞吐)。完整表格见第 0 节。

- 调研定位: 本工作对应学界 "GPU 上的递归滤波 / 串行 scan 融合" 方向。通用
  高性能库 (cuBLAS/cuSOLVER) 针对大矩阵, 在本场景的微矩阵规模下因固定开销
  反成瓶颈 (cuSOLVER 单次求逆 73.5us > v3 整步)。自定义融合 kernel 是正解。
- 正确性: v0/v1/v2 与金标准逐位一致; v3 因 fp64 累加顺序不同有亚毫米偏差
  (ΔZ≈0.13mm), 远小于 RMSE 本身量级, 算法等价。

RMSE (mm): X=1017.838 Y=16.835 Z=8.141 · ρ=0.1755%

---

## 6. 优化路线总结与后续建议

### 已完成路线
```
v0 eager CPU (1820)  ──消碎片/固定shape──▶  v1 (2060, 0 graph break)
     │                                          │
     │                                   ──手动CUDA Graph──▶ v2 GPU (2702, 1.48x)
     │                                          │
     └──────────────单kernel融合──────────────▶ v3 Triton Mega (20534, 11.3x)
```

### 后续改进
1. **fp32**: 定位精度需求 ~mm 级, fp32 足够; (fp64→fp32 + tensor core)
2. **多轨迹batch并行**: 当前单 block 串行受限。多条轨迹 / 蒙特卡洛时, 每个
   CUDA block 跑一条独立 scan, 填满 SM, 吞吐可随轨迹数近线性扩展
3. **持久化kernel + 在线流式**: 100Hz 实时场景下, kernel 常驻、数据流式喂入,
   消除每帧launch

