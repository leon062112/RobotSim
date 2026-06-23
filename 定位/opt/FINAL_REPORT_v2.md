# 套管井机器人多传感融合定位 — GPU 高性能优化最终报告 (v2)

> SINS + 双里程计 EKF 组合导航 · 从朴素 PyTorch 到三级融合 + 多轨迹并行 + 混合精度
> 数据集: PipeRobot_Trajectory.csv (N=166667 步, 100Hz, 500m 管道)
> 硬件: NVIDIA Hopper sm_90, 78 SM, 150GB HBM · torch 2.11.0+cu129 · Triton 3.6.0
> 本报告是 6 月 22 日 TODO 推进后的总报告, 接续早期 FINAL_REPORT.md (v0–v3)

---

## 0. 执行摘要

本工作研究一个对 GPU **极不友好**的负载:严格串行的卡尔曼递归 + 极小矩阵 (≤15×15)
+ 海量碎片化微算子,典型的 **latency-bound** 问题。朴素 GPU 实现甚至比单核 CPU 还慢 4 倍。
我们沿三条正交主线优化,形成论文的三个贡献点:

| 贡献点 | 思路 | 代表版本 | 关键结果 |
|--------|------|---------|---------|
| **① 算子级 (Kernel Fusion)** | 531 微 kernel → 1 个 Triton mega-kernel | v1→v2→v3 | 单条 fp64 **8.8×** vs CPU |
| **② 算法级 (Mixed Precision)** | fp64 → fp32 / TF32 | v4 | fp32 **133×** vs CPU,精度亚毫米 |
| **③ 系统级 (Batch Parallel)** | 轨迹维映射到 grid,每 block 一条 scan | v5 | B≤78 **完美线性**,B=256 聚合 34M 步/s |

全版本吞吐 (full N=166667):

| 版本 | 方法 | 设备/精度 | 吞吐 (步/s) | 相对 v0 | 正确性 |
|------|------|----------|------------|--------|--------|
| v0 | 原始 eager | CPU fp64 | 1866 | 1.00× | 金标准 |
| v1 | 消碎片 | CPU fp64 | 2098 | 1.12× | 逐位等价 |
| v0 | 原始 eager | GPU fp64 | 263* | 0.14× | (慢于CPU) |
| v2 | CUDA Graph | GPU fp64 | 2640 | 1.41× | 逐位等价 |
| v3 | Triton 单 kernel | GPU fp64 | 16491 | 8.8× | 亚毫米 |
| **v4** | **+ fp32** | **GPU fp32** | **242737** | **130×** | **亚毫米** |
| v4 | + TF32 | GPU tf32 | 209524 | 112× | 亚厘米 |
| **v5** | **+ 多轨迹 batch** | **GPU fp32 ×256** | **33.98M** | **18210×** | 逐位一致 |

(* v0 GPU 为 20001 步外推,全量需 ~400s)

**一句话结论**:此问题的加速不来自传统 auto-tuning (tile/layout 搜索空间≈0),而来自三件事的乘积——
**kernel fusion (8.8×) × 混合精度 (14.3×) × 多轨迹并行 (≤78× 线性)**。

---

## 1. Background — 为什么 GPU 不擅长这个任务

(论文 Background 章节素材,数据见 `results/small_matrix_characterization.json`)

### 1.1 算法结构与串行依赖
每个时间步串行执行 7 个阶段:四元数姿态更新 → SINS 速度/位置积分 → 双里程计位移
→ 异常判别 → 构建 H/R → EKF 预测+更新 (15 维) → 误差补偿。
步 k 强依赖步 k-1 的 (pos, vel, q, P) ⇒ **时间维度零并行度**。

### 1.2 小矩阵特性 (微观)
- 单步真实计算量 ≈ **25092 FLOPs**(8 个 ≤15×15 GEMM + 一个 3×3 解析逆 + 标量三角函数)
- 涉及矩阵:最大 15×15 (padded 16×16),最小 3×3
- 单个 `tl.dot` 在 16×16 上只用到一个 warp 的极小部分 ⇒ **算力利用率 < 1%**

### 1.3 四重 GPU 不友好性 (量化)
ncu 实测单条轨迹 (`results/sm_profiling.json`):

| 指标 | 单 block (v4) | 解读 |
|------|--------------|------|
| `sm__throughput` | **0.27%** | 计算单元几乎全空闲 |
| `sm__warps_active` | **6.25%** | 64 warp/SM 只用 4 个 |
| 占用 SM 数 | **1 / 78** | 单条轨迹只用 1.3% 的 GPU |
| `waves_per_multiprocessor` | 0.00 | 不足一个 wave |

→ ① 矩阵太小用不满 warp;② 串行递归零时间并行;③ 微矩阵+串行 = latency-bound,
occupancy 天然 ≤ 6.25%;④ 朴素逐算子实现的 launch/同步开销 **远大于** 真实算力。
**这是本工作所有优化的出发点。**

---

## 2. 性能瓶颈分析 (Profiling)

### 2.1 Eager CPU (torch.profiler, n=2000)
- **每步 ~1026 个算子调用**,共 205 万次 op / 2000 步
- 真正线代 (inv/lu_solve/mm/mv) 仅占 ~15-20%;大头是碎片小算子:
  `aten::select` 14.2%、`aten::mul` 11.8%、`aten::item`/`_local_scalar_dense` 各 ~5%
- 根因:`skew/quat2dcm/quatmultiply/eul2quat` 和 `theta_mat/H/F` 全部用
  `torch.tensor([[标量...]])` **现场构造**,每个标量都触发 select+item+mul 链。

### 2.2 Eager GPU 为何更慢
- v0 GPU 仅 263 步/s,**比 CPU 慢**:`.item()` 隐式同步 + 海量 kernel launch
  在微小算力上完全主导。

### 2.3 单步 CUDA kernel 统计 (v2)
- **每步 531 个 CUDA kernel launch**,817us GPU 时间
- cutlass gemm 仅占 ~20%,其余是 mul/cat/sub/neg 微 kernel 链
- 微 kernel 间 ~0.7us 转换延迟串行累积 ⇒ 单步 latency ≈ 531 × 间隙
- unroll 1→50 步吞吐不变 ⇒ **已非 launch-bound,而是 GPU-execution-bound**

### 2.4 算子库参照 (揭示本质)
- cuBLAS GEMM (F@P@Fᵀ): **17.5 us/步** · cuSOLVER inv(3×3): **134 us/步**
- 仅 cuSOLVER 单次 3×3 求逆就 > v3 整步 (~48us) ⇒ **通用库为大矩阵设计,
  在 15×15 规模下其固定开销反成瓶颈**。这是"为何不用现成库"的硬核论据。

---

## 3. 贡献点 ① 算子级优化:Kernel Fusion 三级递进

### 3.1 v1 — compile-friendly 重构 (消碎片)
- `stack/cat` 在张量上构造 theta_mat/H/F,杜绝 python 标量与 `.item()`
- 固定 3 维观测:异常时 `R_odo→1e12` 抑制里程计通道 (K≈0),与原"删观测行"数学等价
- `torch._dynamo.explain`: **graph_count=1, graph_break=0, op_count=241** (vs v0 的 1026/步)
- 结果:CPU eager 1.12×;但 **torch.compile(default) 反而慢** (超小图 codegen/守卫开销 > 收益)

### 3.2 v2 — CUDA Graph capture/replay
- 单步图捕获为 1 个 CUDA Graph,N 次 replay,消除 CPU 调度开销 → **1.41×**
- 踩坑:cuSOLVER 在 capture 中触发 host 同步 → 改 **3×3 解析 adjugate 逆**;
  GPU 标量索引同步 → `index_select` + 1-elem idx
- unroll 无效 ⇒ 证明瓶颈在 GPU 执行而非 launch

### 3.3 v3 — Triton 单 kernel Mega Kernel (本贡献点核心)
- **整个 N 步串行 scan 写进 1 个 Triton kernel**:531 微 kernel → **1 次 launch**
- state (pos/vel/q/P, padded 16×16) 常驻寄存器,跨步零全局往返
- 矩阵全 padded-16 + `tl.dot`(fp64);3×3 解析逆;`libdevice.{atan2,sin,cos,sqrt,abs}`
- 踩坑:`F[6:9,6:9]=-skew(w)` 是**覆盖**对角块,padded eye 的对角 1 需先 -1 抵消
- 结果:**16491 步/s = 8.8× vs CPU,5.5× vs CUDA Graph**;vs 金标准偏差 ≤9 微米 (fp64 重排)

> 小结:fusion 是 latency-bound 串行问题的正解。三级递进
> (算子级 → 图级 → 单 kernel) 把"每步 531 个有依赖的微 kernel"压成"1 个 kernel"。

---

## 4. 贡献点 ② 算法级优化:Mixed Precision (v4)

(代码 `ekf_v4.py`,数据 `results/v4_precision_full.json`)

### 4.1 方法
把数据类型 `DT` 与 `tl.dot` 的 `input_precision IP` 提为 **constexpr 模板参数**,
一份 kernel 编译出 fp64(ieee) / fp32(ieee) / tf32(TensorCore) 三档。
真值始终 fp64,输出 cast 回 fp64 计算 RMSE,保证精度评估公平。

### 4.2 精度 vs 吞吐 (full N=166667)

| 精度 | 吞吐 (步/s) | vs fp64 | RMSE X/Y/Z (mm) | vs CPU 金标准偏差 |
|------|------------|---------|------------------|------------------|
| fp64 | 17090 | 1.0× | 1017.842 / 16.836 / 8.269 | ΔZ≈0.13mm |
| **fp32** | **243587** | **14.3×** | 1017.730 / 16.837 / 8.270 | **ΔX≈0.11mm,全轴亚毫米** ✅ |
| tf32 | 209472 | 12.3× | 1022.550 / 17.236 / 8.577 | ΔX≈4.7mm,亚厘米 |

### 4.3 两个关键发现

**(1) fp32 是"免费午餐",且收益远超预期。**
原预期 fp32 仅快 1.5-2×,实测 **14.3×**。原因:
- Hopper 这张卡 fp64 ALU 吞吐远低于 fp32;
- 串行 latency-bound 下,fp64 的 transcendental (libdevice sin/cos/sqrt/atan2)、
  双倍寄存器压力、指令延迟被**逐步放大**,每一项都拖慢关键路径。
- 精度代价仅 ΔX≈0.11mm,远小于 RMSE 本身 (X≈1m,主要来自 X 轴系统误差而非数值),
  **mm 级定位需求下 fp32 完全够用**。

**(2) TF32 / TensorCore 在此场景不划算。**
tf32 比 fp32 **更慢且精度更差**:矩阵 ≤15,TensorCore 的收益被 padding 到 16 的浪费
和尾数截断抵消。**印证了 auto-tuning 结论**——微矩阵场景 TC 无用武之地。

---

## 5. 贡献点 ③ 系统级优化:多轨迹 Batch 并行 (v5)

(代码 `ekf_v5.py`,数据 `results/v5_batch_scaling.json` + `sm_profiling.json`)

### 5.1 动机与方法
单条轨迹的卡尔曼递归无法并行 (v3/v4 已达单条 latency 下限)。但 **蒙特卡洛 / 多传感器
/ 多目标跟踪** 场景存在大量**相互独立**的轨迹。把"轨迹维"映射到 grid:
`grid=(B,)`,每个 CUDA block 跑一条独立 scan。kernel body 与 v4 完全一致,仅指针加
`pid × batch_stride` 偏移。默认 fp32 (§4 已证最优)。

### 5.2 吞吐扩展 (full N=166667, fp32)

| B | 吞吐 (步/s) | traj/s | speedup vs B=1 |
|---|------------|--------|----------------|
| 1 | 229k | 1.4 | 1.0× |
| 16 | 3.67M | 22.0 | **16.0× (完美线性)** |
| 32 | 7.35M | 44.1 | **32.0×** |
| 64 | 14.66M | 88.0 | **64.0×** |
| 78 (=SM 数) | 17.86M | 107.2 | 77.9× |
| 128 | 24.07M | 144.4 | 105× |
| 256 | **34.08M** | 204.5 | 148.7× |

### 5.3 SM 利用率证据 (ncu)
| 配置 | sm__throughput | 解读 |
|------|---------------|------|
| 单 block | 0.27% | 占 1/78 SM |
| B=78 | **19.91%** (**~74×**) | 78 block 铺满 78 SM |

- 寄存器 110 regs/thread → `occupancy_limit_registers = 4 block/SM` →
  理论 ~312 条轨迹饱和,解释了 B≤256 仍近线性。
- B≤78 **speedup ≈ B** 的完美线性:轨迹维映射是填满 SM 的正解。
- 各轨迹结果 `traj_spread = 0` (逐位一致),确认 block 间无串扰,正确性无损。

> 注意:v5 的 34M 步/s 是**集群吞吐**,非单条加速;单条 latency 仍受 §3 串行下限约束。
> v3(降单条延迟)与 v5(升集群吞吐)是**正交**的两个维度。

---

## 6. Auto-Tuning 与输入特征分析

(数据 `results/data_pattern_analysis.json`)

- dt 严格均匀 (std/mean < 1e-6),单 batch,**shape 全程静态** → compile/CUDA Graph 缓存 100% 命中
- 异常判别命中率 normal ≈ 100% (|ΔD-ΔS| max=0.0007 ≪ 阈值 0.01) → 固定 3 维观测合理
- 唯一动态点 (2维/3维观测) 已被"固定 3 维 + R 抑制"消除

| Auto-Tuning 方向 | 搜索空间 | 结论 |
|------------------|---------|------|
| Tile Size | ≈0 | 矩阵 ≤15,单 warp 足够 |
| Memory Layout | ≈0 | state 驻寄存器,无需搜索 |
| **Kernel Fusion** | **极大** | 531→1,唯一高收益 (v3 已实现) |
| **精度选择** | **大** | fp64→fp32 = 14.3× (v4 已实现) |
| TensorCore | 负 | 微矩阵下 TF32 反而更慢 (v4 已验证) |

**结论**:传统 tile/layout auto-tuning 在此问题几乎为 0;收益全在 **fusion + 精度 + 并行映射**。

---

## 7. 高级 GPU Kernel 手段适用性分析 (Negative Result)

(对应 6.22 TODO §4 "Tensor Core 优化" 等条目;本节是一个有价值的**负结果**)

§6 表明传统 tile/layout auto-tuning 收益≈0。本节进一步论证:近年面向**吞吐受限大矩阵**
(GEMM / Attention / Conv) 的 SOTA kernel 手段,与本问题画像
(**latency-bound + 微矩阵 ≤15 + 串行递归**) 是**结构性错配**;其中 TC/tf32 已在 v4 实测为负收益。

### 7.1 逐手段裁决

| 手段 | 设计目标 | 适用? | 依据 |
|------|---------|-------|------|
| **Tensor Core / tf32** | 大矩阵独立 MAC 吞吐 | ❌ 实测负收益 | v4:tf32 比 fp32 更慢 + ΔX 4.7mm。15×15 padding 到 16 后 TC tile 大半在算补零;SM throughput 仅 0.27% 说明算力单元在**等依赖**而非缺算力 |
| **TMA** (Tensor Memory Accelerator) | 128×128 级 global↔shared 异步大块搬运 | ❌ 无杠杆 | 单步工作集仅几 KB,全程驻留寄存器/shared,**无大块 DMA 可 overlap**;descriptor 建立开销 > 收益 |
| **Warp Specialization** | producer(访存)/consumer(计算) 流水 | ❌ 无杠杆 | 单轨迹仅 4 warp,工作集驻留**无访存阶段可隐藏**;单步内部 skew→F→P→K→update 强依赖链,拆 warp 同步开销 > 算力 |
| **Persistent Kernel** | 消除 per-launch 调度开销 | ✅ **已实现** | v3/v5 即单次 launch 跑完整 scan(nsys 实测 1 launch);在线流式场景的持久化属未来方向(§10) |
| **Multi-Stream Streaming** | 计算与 H2D/D2H overlap | ❌ 无杠杆 | 瓶颈是 compute 非 transfer,数据全程 on-device |
| **小矩阵合并 / 对角堆叠** | 多个小 GEMM 批量化 | 🟡 部分=v5 | 见 §7.2 |

### 7.2 "小矩阵合并 / 对角堆叠" 的两种语义

- **块对角堆叠成单个大矩阵做一次 matmul** —— ❌ **反模式**:B 条独立小矩阵堆成 block-diagonal,
  off-diagonal 全零,白算 B× FLOPs,严格劣于 batched 路线。
- **跨 batch 维堆成 grid/bmm** —— ✅ 这**正是 v5 已做的**(每条轨迹一个 block 并行 scan)。
  未走 cuBLAS batched-GEMM,是因为单步近一半算子是 sin/cos/atan2 等 transcendental,
  cuBLAS 不覆盖,自写 scan 反而能把整步融合。

### 7.3 唯一有边际价值的方向:利用 F 的块稀疏结构

F 矩阵本质是 `I + 少数 3×3 off-diagonal (skew) 块`,现按稠密 15×15 计算 `F·P·Fᵀ`。
改为只算非零 3×3 块(块稀疏)的影响:

- 对**单轨迹**:❌ 无用。FLOPs 从 ~6750 降到 ~2000,救不了 latency-bound(算力本就空闲)。
- 对 **batch 吞吐**:🟡 有**间接**价值。当前 110 regs → 4 block/SM → ~312 轨迹饱和;
  15×15 fp64 驻留是寄存器压力主因。块稀疏可降寄存器 → 提 occupancy → 推高饱和点、
  拉长 batch 线性区间。属吞吐侧二阶优化,复杂度不低,列为后续候选(§10)。

### 7.4 底线

这些手段解决的是"算力喂不饱",而本问题的病是"**算力在等依赖**"(SM throughput 0.27%)。
三大贡献点已把 latency-bound 串行递归里能拿的都拿了。若要再榨,**唯一有头部空间的轴是
batch 吞吐侧的 occupancy(走 F 块稀疏降寄存器压力),而非 TC / TMA / warp-specialization**。
这一负结果本身体现了对问题本质的判断——**不盲目堆 SOTA 技术**。

---

## 8. Baseline 对比与相关工作

### 8.1 对比对象 (本工作内部)
原始 eager (v0) / torch.compile (v1) / 手工 CUDA Graph (v2) / 手工 Triton fusion (v3-v5)
/ 算子库微基准 (cuBLAS gemm 17.5us, cuSOLVER inv 134us)。完整表见 §0。

### 8.2 相关工作定位 (按贡献点)
- **算子级 fusion**:与 SSM 并行 scan 的 Triton 实现 (Mamba selective-scan、S5、FlashAttention
  系列的单 kernel 融合思想) 同源——把有依赖的逐步计算驻留片上、单次 launch。
- **系统级 batch**:主流 GPU 卡尔曼库 (如 batched cuBLAS/cuSOLVER 路线) 几乎都是
  **多轨迹 batched 设计**,而**缺少单轨迹串行 latency-bound 的高性能实现**——正是本工作 v3 的空白点。
- **算法级并行滤波 (未来方向)**:Särkkä & García-Fernández "Temporal Parallelization of
  Bayesian Smoothers" 把 KF 递归改写为满足结合律的算子 + 并行前缀和,串行深度 O(N)→O(log N)。
  这是打破单轨迹串行本质的唯一"质变级"方向,与 v3(把串行做到极致)正交。

> ⚠️ 上述论文卷期/arXiv 编号/开源库 URL 因本机公网受限未能联网复核,
> 引用前需核实 (详见 `RESEARCH_TODO5_baseline.md` D 节)。

---

## 9. 论文结构建议 (对应 6.22 TODO)

```
1. Background        小矩阵 + 串行递归的 GPU 挑战 (§1, ncu 证据)
2. Method
   2.1 Operator-level Fusion         (贡献点①, v1→v3)
   2.2 Mixed-Precision Execution     (贡献点③→论文算法级, v4)
   2.3 Multi-Trajectory Batch Parallelism (贡献点②→论文系统级, v5)
3. Evaluation
   Accuracy (亚毫米) / Throughput (§0) / SM Utilization (§5.3) / Scaling (§5.2)
4. Related Work      (§8.2);Negative Result 章节可引 §7
```

---

## 10. 总结与后续方向

### 已完成路线
```
v0 eager CPU (1866)
   │ 消碎片/固定shape
   ▼
v1 (2098, 0 graph break)
   │ 手工 CUDA Graph
   ▼
v2 GPU (2640, 1.41×)
   │ 单 kernel 融合  ← 贡献点①
   ▼
v3 Triton fp64 (16491, 8.8×)
   │ 混合精度 fp32   ← 贡献点③
   ▼
v4 fp32 (242737, 130×)
   │ 多轨迹 batch    ← 贡献点②
   ▼
v5 ×256 (34M 步/s, 集群吞吐)
```

### 后续 (尚未实现)
1. **Associative-scan 并行 EKF**:O(N)→O(log N),打破单轨迹串行本质 (需自写 Blelloch scan
   或借 JAX 验证;PyTorch 无 `associative_scan` 原语是主要障碍)
2. **F 块稀疏 → 提 occupancy** (§7.3):只算 F 的非零 3×3 块,降寄存器压力,把 batch 饱和点
   从 ~312 推高,拉长 v5 线性区间。**当前唯一有头部空间的吞吐侧方向**
3. **cuBLASDx + cuSOLVERDx device 端融合**:官方支持单 kernel 内 GEMM+分解 (需 CUDA 13.0+,
   现 12.9 需升级)
4. **持久化 kernel + 流式**:100Hz 在线场景,kernel 常驻 + ring buffer 喂数,消除每帧 launch
5. **极小块 FMA 展开**:3×3/4×4 避开 padded-16 的 tl.dot 浪费 (微调)

> 注:TC / TMA / warp-specialization 经 §7 分析为结构性错配,**不列入**后续方向。

### 一句话总结
对这个串行 + 微算子 + latency-bound 的负载,GPU 的胜利不靠堆并行,而靠
**融合消除 launch (8.8×) × 降精度匹配硬件 (14.3×) × 用独立轨迹填满 SM (≤78× 线性)** 三者相乘。
