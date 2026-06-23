# EKF 优化日志 (Optimization Log)

数据集: PipeRobot_Trajectory.csv, N=166667 步, float64
正确性金标准 (full CPU baseline, 原始 ekf.py 等价):
  RMSE mm: X=1017.838305  Y=16.834549  Z=8.140964 | rho=0.175540%

---

## v0 — Eager baseline (原始实现, 参数化 device/n_steps)
代码: opt/ekf_baseline.py

| device | n_steps | elapsed | throughput | 备注 |
|--------|---------|---------|-----------|------|
| CPU    | 166667  | 91.09 s | 1830 步/s | 与原 ekf.py 完全一致 = 金标准 |
| CPU    | 5000    | 2.95 s  | 1693 步/s | |
| CUDA   | 2000    | 4.77 s  | 419 步/s  | **比CPU慢4倍** (小矩阵, kernel launch + .item() 同步主导) |

### Profiling (CPU, n=2000, torch.profiler)
profiles/v0_cpu_n2000_summary.txt + _trace.json

关键发现 (**每步 ~1026 个算子调用!**):
- 总算子调用 2,051,411 次 / 1999 步 = 1026 op/步
- 真正的线代 (linalg_inv/lu_solve/mm/mv) 只占 ~15-20% 时间
- 大头是**碎片化小算子开销**:
  - `aten::select` 161932次 (14.2%) — 来自 q[k]、gyro[k-1,0] 等逐元素索引
  - `aten::mul` 111964次 (11.8%)
  - `aten::item` + `_local_scalar_dense` 各 111954次 — **CPU<->py 标量同步, GPU 上是致命伤**
  - `aten::pow` 37987次, `aten::copy_` 71986次
- 根因: skew/quat2dcm/quatmultiply/eul2quat + theta_mat/H/F 全部用
  `torch.tensor([[标量,...]])` 现场构造 → 每个标量都触发 select/item/mul

### 结论 (指导优化方向)
1. 卡尔曼递归天然串行 (步k依赖k-1) → **无法跨时间步并行**, mega-kernel=融合"单步"为一个无 graph-break 的图
2. GPU 慢是因为 launch 开销 + .item() 同步, 而非算力不足 → 必须消除 .item()/python标量
3. torch.compile 会因 3 处 graph break 失败:
   - `.item()` 隐式同步 (theta_norm>1e-10, abs(delta_D-delta_S)<thresh)
   - 数据相关控制流 (正常3维/异常2维观测, H 形状变化)
   - `torch.tensor([...])` 从 python 标量构造
4. 优化手段:
   - 用 stack/index_put 在 GPU 张量上原位构造 theta_mat/H/F, 杜绝 python 标量
   - 统一观测模型为固定 3 维: 异常时把里程计通道的 R 置极大 (K≈0 等效拒绝) → 消除分支与变形
   - theta_norm 分支用 torch.where + clamp 替代

---

## v1 — compile-friendly 重构 (单步纯张量函数 ekf_step)
代码: opt/ekf_v1.py
改造: stack/cat 构造所有矩阵; 固定3维观测(R_odo→1e12抑制异常通道); torch.where替代if; 无.item()

### 正确性 ✅ 完全等价金标准
- full CPU: X=1017.838305 Y=16.834549 Z=8.140964 rho=0.175540% (逐位匹配 v0)
- n=5000: X=948.751759 Y=2.758031 Z=1.128291 (匹配 v0)
- R_odo=1e12 抑制异常里程计通道, 与 v0 "删除观测行" 数学等价 ✓

### torch.compile 分析 (dynamo.explain on ekf_step)
- **graph_count=1, graph_break_count=0, op_count=241** → 单步完整捕获, 无 graph break (达成 TODO#1 目标)
- 对比 v0 每步 1026 op → v1 单步 241 op (FX图层面, 减少~76%)

### 性能 (关键反直觉结果)
| 配置 | n | throughput | vs v0 |
|------|---|-----------|-------|
| v1 eager CPU full | 166667 | 2079 步/s | **1.14x** (80.2s vs 91.1s) |
| v1 eager CPU 5000 | 5000 | 1994 步/s | 1.18x |
| v1 compile(default) CPU 5000 | 5000 | 306 步/s | **0.18x 更慢!** |
| v1 eager GPU 2000 | 2000 | 599 步/s | (v0 GPU 419) 1.43x |
| v1 compile(default) GPU 2000 | 2000 | 428 步/s | 比eager慢 |

### 核心结论 (决定 v2 方向)
- **torch.compile(default) 在此处反而更慢**: 单步图太小(241 op→几十个kernel), 而循环要执行166667次python迭代+图调用, overhead 主导。inductor 的 codegen 开销/守卫检查 > tiny step 收益
- eager 提速来自消除 .item()/select 碎片, 而非编译
- **reduce-overhead (CUDA Graphs) 是正解**: 报错 "accessing tensor output of CUDAGraphs overwritten" — 因为 state 张量在循环里反复读写。需用静态输入/输出 buffer + cudagraph_mark_step_begin, 或手动 CUDA graph capture/replay
- 真正的 "Mega Kernel" 对**串行 scan** = 把单步固化成一个 CUDA Graph, 每步只 replay 一次 (1 launch 替代几十个) → v2 方向

---

## v2 — CUDA Graph capture/replay "Mega Kernel"
代码: opt/ekf_v2.py
思路: 串行scan每步图固定 → CUDA Graph 捕获单步, 之后纯 replay; state常驻GPU静态buffer, idx用GPU long张量自增

### 踩坑记录 (重要)
1. `torch.linalg.inv(S)` 在 graph capture 中报 `cudaErrorStreamCaptureInvalidated` (cuSOLVER 内部有 host 同步/动态分配) → 改用 **3x3 解析逆 inv3x3()** (adjugate/det, 纯 elementwise)
2. GPU 标量索引 `gyro[idx]` (idx 为 GPU 标量张量) 在 capture 中触发 host 同步而失败 → 改用 `torch.index_select(x, 0, idx_1elem)` (纯 device gather)
3. idx 必须是 1-elem 张量 (`torch.tensor([1])`) 才能被 index_select 接受

### 正确性 ✅ 完全等价金标准
- full: X=1017.838305 Y=16.834549 Z=8.140964 rho=0.175540% (逐位匹配)

### 性能
| 配置 | n | throughput | vs v0(1830) | vs v1eager(2079) |
|------|---|-----------|------|------|
| v2 CUDAGraph full | 166667 | **2700 步/s** (61.7s) | **1.48x** | 1.30x |

unroll 扫描 (graph 内循环 1/5/10/20/50 步): throughput 恒定 ~2690 步/s, 无变化
→ 证明**已不是 launch-bound, 而是 GPU-execution-bound**

### Profiling (eager step, 单步 CUDA kernel 统计)
profiles: per-step **531 个 CUDA kernel launch**, 817us GPU 时间
- aten::mul 75/步, aten::cat 31/步, aten::sub 32/步, aten::neg 27/步, aten::add 23/步
- 元凶: torch.stack/cat 现场构造 theta_mat/Cnb/H/F/sk_fn/sk_w → 每个小矩阵几十个 1.3us 微 kernel
- mm/cutlass gemm 仅占 ~20% → **真正算力极少, 几乎全是微 kernel 链的延迟**

### 关键结论 (决定 v3 方向)
- CUDA Graph 消除了 CPU 端 launch 调度开销, 但 GPU 仍需**串行执行 531 个依赖微 kernel**, 每个 kernel 间有 ~0.7us 转换延迟 → 单步 latency ≈ 531 × kernel间隙, 算力闲置
- 因 EKF 单步矩阵极小 (≤15×15), GPU 完全无法被填满
- **唯一能让 GPU 发挥的路径**: 把整个单步写成 1 个 kernel (Triton/CUDA), 531→1, 消除微 kernel 链延迟 → 这才是真正的 "Mega Kernel"
- 也解释了为何 GPU 仅比 CPU 快 1.48x: 此问题本质 latency-bound + 微算子, 对 GPU 极不友好

---

## 数据 Pattern 分析 (TODO #4)
代码: opt/analyze_data_pattern.py → results/data_pattern_analysis.json

- N=166667, dt 严格均匀 (std/mean<1e-6), 单 batch, **shape 全程静态**
- 异常分支命中率: normal≈100% (|dD-dS| max=0.0007 << thresh=0.01) → 固定3维观测完全合理
- 矩阵规模: 全部 ≤15×15 稠密小矩阵; H 后9列恒0(列稀疏)但太小, cuSPARSE 无意义
- **动态性结论**: shape/batch 均静态 → compile/CUDAGraph 缓存 100% 命中, 无重编译
- **Auto-Tuning 结论**: tile/layout 搜索空间≈0 (矩阵太小, 单 warp 足够);
  收益全部集中在 **kernel fusion (531→1)** 和 **精度 (fp64→fp32)** 两个方向
- Triton fp64 可行性验证: tl.dot 支持 16×16 fp64 ✓; libdevice.atan2/sin/cos/sqrt fp64 ✓;
  kernel 内 runtime-loop 携带矩阵 state 做 scan ✓ → v3 单 kernel mega-kernel 技术可行

---

## v3 — Triton 单 kernel Mega Kernel (终极, TODO #2)
代码: opt/ekf_v3.py (kernel building blocks 验证: triton_blocks.py / triton_features.py / triton_probe.py)
思路: **整个 N 步串行 scan 在 1 个 Triton kernel 内完成**
- 1 次 kernel launch (vs v0 的 N×531; vs v2 的 N 次 replay)
- 单 program(1 block), state(pos/vel/q/P 16×16 padded) 常驻寄存器, 跨步零 global 往返
- 每步只读 gyro/accel/odom, 写 pos/vel; 矩阵全 padded-16 + tl.dot(fp64); 3×3 解析逆; where 分支

### 踩坑
- libdevice 无 fabs → 用 libdevice.abs
- F[6:9,6:9] = -skew(w)*dt 是**覆盖**对角块(非叠加): padded eye 的对角1需先减掉 → `where(r==c & 6<=r<9, -1, 0)` 抵消
- @triton.jit 函数必须写在 .py 文件 (不能 python -c)

### 正确性 ✅ (fp64, 累加顺序不同, 亚毫米级偏差)
- full: X=1017.842 Y=16.836 Z=8.269 rho=0.175539% (golden X=1017.838 Y=16.835 Z=8.141)
- ΔZ≈0.13mm, ΔX≈0.004mm → 浮点重排导致, 量级远小于 RMSE 本身, 算法等价 ✓

### 性能 🚀 (重大突破)
| 配置 | n | throughput | vs v0(1830) | vs v2(2700) |
|------|---|-----------|------|------|
| v3 Triton Mega full | 166667 | **14815 步/s** (11.25s) | **8.1x** | **5.5x** |
| v3 5000 | 5000 | 3185 步/s | (含编译warmup, 已预热) | |

### 结论
- **531 微 kernel → 1 kernel**: 彻底消除 kernel 间转换延迟, 这是 latency-bound 串行问题的正解
- 8.1x vs CPU eager, 5.5x vs CUDA Graph → 验证 TODO#2 "Mega Kernel" 假设成立
- 仍是单 block(串行依赖无法并行), 但寄存器内计算 + 单次 launch 把 overhead 压到极致
- 进一步空间: fp32(精度换速度)、多轨迹 batch(每个 block 跑一条, 填满 SM)

---

## 最终 Benchmark 汇总 (TODO #5)
代码: opt/benchmark.py → results/benchmark_summary.json

| 版本 | 设备 | n | throughput | 相对v0 | 正确性 |
|------|------|---|-----------|-------|-------|
| v0 eager (golden) | CPU | 166667 | 1820 步/s | 1.00x | 基准 |
| v1 eager | CPU | 166667 | 2060 步/s | 1.13x | ✅等价 |
| v0 eager | GPU | 20001* | 439 步/s | 0.24x | (partial) |
| v1 eager | GPU | 20001* | 651 步/s | 0.36x | (partial) |
| v1 compile(default) | GPU | 20001* | 2233 步/s | 1.23x | (partial) |
| v2 CUDA Graph | GPU | 166661 | 2702 步/s | 1.48x | ✅等价 |
| **v3 Triton Mega** | GPU | 166667 | **20534 步/s** | **11.3x** | ✅亚毫米 |

(* eager GPU 全量需 ~400s, 限 20k 步; partial 行 RMSE 非全量金标准, 仅测吞吐)

### 算子库参照 (单步 eager, 揭示瓶颈本质)
- cuBLAS gemm (F@P@F'): **17.6 us/步**
- cuSOLVER inv 3×3: **73.5 us/步** ← 单个求逆就比 v3 整步还慢!
- v3 整步 (含姿态/SINS/F/P/K/补偿 全部): **~48.7 us/步** (1/20534)
- → 结论: eager GPU 每步光 cuSOLVER 求逆就 73us, 累积 166667 步 = 12s 纯求逆开销;
  v3 把"整步"压到 ~49us, 比单次 cuSOLVER 调用还快 → 融合 + 解析逆 + 寄存器驻留的胜利

### 最终结论 (TODO 全部完成)
1. 此 EKF 是**串行 + 微算子 + latency-bound** 问题, 对朴素 GPU 极不友好 (eager GPU 比 CPU 还慢)
2. torch.compile(default) 对超小图收益有限 (CPU 反而更慢; GPU 1.23x)
3. **真正有效的是 fusion 三级递进**: 算子级(v1) → 图级 CUDA Graph(v2 1.48x) → 单 kernel(v3 11.3x)
4. Auto-tuning(tile/layout) 在此问题空间≈0; 收益全在 kernel fusion
5. v3 已逼近单 block 串行上限; 再提速需 fp32 或多轨迹 batch 并行

### v3 正确性硬验证 (逐步轨迹对比, n=5000)
golden(v0 CPU) vs v3 每步位置绝对偏差:
- max: X=3.8e-7m Y=1.6e-8m Z=9.0e-6m (≤9微米)
- mean: X=1.9e-7m Y=6.1e-9m Z=4.4e-6m
→ 偏差纯属 fp64 累加顺序重排, 比 RMSE(mm级)小3个数量级, 算法严格等价 ✓

---

## TODO #5 Baseline 与对比实验调研 (报告: opt/RESEARCH_TODO5_baseline.md)
公网被墙(arxiv/github ECONNREFUSED), 结论来自模型知识 + 本机源码核实(triton3.6.0/torch2.11/sm90/CUDA12.9)

### 现状定位 (v0~v3 已完成, fusion 三级递进已验证)
- 工程优化路线已走完: v1 算子级 → v2 CUDA Graph(1.48x) → **v3 Triton 单 kernel(11.3x) 已达单 block 串行上限**
- 即 TODO#2 "Mega Kernel" 假设已被 v3 证实. 本次调研回答"还能往哪走 + 跟谁比 + related work"

### 核心发现 (按对本项目价值排序)
1. **Associative-scan 并行滤波 = 唯一剩下的"质变级"方向 (算法级, 非工程级)**
   - Särkkä & García-Fernández "Temporal Parallelization of Bayesian Smoothers"
     arXiv:1905.13002 / IEEE TAC 2021. 把 KF 递归改成满足结合律的算子 → 并行前缀和(Blelloch scan),
     **串行深度 O(N)→O(log N)**. N=166667 → ~36 层. 并行的是**时间维**(不需多轨迹), 正中单轨迹串行本质.
   - v3 是"把串行做到极致"(单 block 串行上限); 此方向是"打破串行本身", 二者正交.
   - 非线性 EKF 扩展: Yaghoobi et al. ICASSP 2021 (arXiv:2102.00514) = 外层迭代线性化 + 内层时间并行.
     需先固定观测结构(v1 已用 R=1e12 抑制法). 配 Cholesky 平方根(TFP scale_tril)保数值.
   - 开源参考: dynamax(JAX) parallel_lgssm_filter / TFP parallel_filter. **PyTorch 无 associative_scan 原语**=主要障碍.
   - 代价: 需物化 N 个 15×15 消息(数 GB 显存); 算子内 solve/slogdet 单元素工作量 > 朴素一步.
2. **v3 已验证 mega-kernel; 注意一个 tl.dot 细节[源码核实]**:
   nvidia min_dot_size 对 ≥16bit 返回 K≥16. v3 把矩阵 padded 到 16 用 tl.dot — **能跑且拿到 11.3x**,
   但极小块(3×3/4×4)的 padding 浪费仍在; 若想再压, 极小块可改 **FMA 显式展开**(潜在微调, 非必须).
3. **device 端融合上限方案 cuBLASDx+cuSOLVERDx**: 唯一官方支持单 kernel 内 GEMM+Cholesky/LU/solve 融合,
   fp64 支持. **但要 CUDA Toolkit 13.0+** (现 12.9 → 需升级). 自带稳健 solver, 免手写分解. = 潜在 v5.
4. **batched 库(cuBLAS/cuSOLVER/MAGMA) 对本项目不适用**: 解决"多独立小矩阵", 我们 batch=1 串行用不上.
   唯一翻盘场景=多 seed 蒙特卡洛(batch=轨迹数). cuSPARSE 无意义(矩阵太小; H 列稀疏应在 kernel 内裁剪而非引库).
5. **开源 baseline / related work**:
   - baseline 候选: torchfilter(Stanford, PyTorch, 技术栈一致首选) / dynamax(JAX, 含并行KF) /
     KalmanNet / filterpy(CPU 数值金标准交叉验证).
   - **立论点**: 主流 GPU KF 库几乎全是 batched 多轨迹设计, 缺单轨迹串行 latency-bound 实现 = 本工作空白点.
   - INS/GNSS EKF 上 GPU 的专门工作稀少(工程主流 FPGA/DSP), GPU 切入点多是粒子滤波. 本身是有力 related-work 陈述.
   - 相邻领域: SSM parallel scan (S5 ICLR2023 / Mamba 2023 selective-scan Triton / S4 ICLR2022) 与并行KF 数学同源, 可借鉴 Mamba 的 Triton scan 实现.

### benchmark.py 补充建议 (非现在改)
- 加 CPU 单/多线程上限行(让"GPU 是否值得"有量化答案); 加 fp32 对照(精度换吞吐, 最低成本真实加速);
  把 batched 库 + cuSPARSE 明确列为"评估后排除"(附理由). 可选: torchfilter 单轨迹验证主流库同样 latency-bound.

### 后续路线 (v3 之后)
- v4: associative-scan 并行 EKF (算法级质变, O(N)→O(logN), 需自写 Blelloch scan 或迁 JAX 验证)
- v5: cuBLASDx+cuSOLVERDx device 融合 (上限最高, 需升级 CUDA 13)
- 微调: v3 极小块 FMA 展开 / fp32 / 多轨迹 batch(填满 SM)

### ⚠️ 公网核实清单 (报告 D 节): 论文卷期页/GitHub URL/arXiv 标题均待联网复核, 勿直接引用

---

## v4 — 精度可配置 Mega Kernel (fp64/fp32/tf32) — 6.22 TODO#1,#5 (算法/精度优化, 贡献点3)
代码: opt/ekf_v4.py (DT/IP 做成 constexpr 模板, 一份 kernel 跑三档精度), 结果: results/v4_precision_full.json

### 做法
- 在 v3 单 kernel 基础上, 把 dtype `DT` 与 `tl.dot(input_precision=IP)` 提为 constexpr
- fp64(ieee) / fp32(ieee) / tf32(TensorCore) 三档; truth 始终 fp64, 输出 cast 回 fp64 算 RMSE
- 探针验证: tl.dot 支持 input_precision ∈ {ieee, tf32}, 16×16 三档均可编译运行

### 正确性 (full N=166667, vs CPU golden X=1017.838 Y=16.835 Z=8.141)
| 精度 | RMSE X mm | Y mm | Z mm | vs golden 偏差 |
|------|-----------|------|------|------|
| fp64 | 1017.842 | 16.836 | 8.269 | ΔZ≈0.13mm (与 v3 同, fp64 重排) |
| **fp32** | 1017.730 | 16.837 | 8.270 | **ΔX≈0.11mm, 全轴亚毫米, 满足 mm 级** ✅ |
| tf32 | 1022.550 | 17.236 | 8.577 | ΔX≈4.7mm, ΔZ≈0.44mm (TC 截断, 仍亚 cm 但明显劣化) |

### 性能 🚀 (重大发现: fp32 收益远超预期 1.5-2x)
| 精度 | throughput | vs fp64 | vs v0 CPU |
|------|-----------|---------|-----------|
| fp64 | 17090 步/s | 1.0x | 9.3x |
| **fp32** | **243587 步/s** | **14.3x** | **133x** |
| tf32 | 209472 步/s | 12.3x | 114x |

### 关键结论
- **fp32 单精度带来 14x 加速 (非预期的 1.5-2x)**: 此 Hopper 卡 fp64 ALU 吞吐远低于 fp32, 且串行 latency-bound 下 fp64 transcendental(sin/cos/sqrt/atan2 via libdevice)、寄存器压力(双倍)、指令延迟全部放大 → fp32 多重受益
- **fp32 精度完全够用**: 与 fp64-triton 偏差 <0.12mm, 远小于 RMSE 本身(X≈1m, 来自 X 轴系统误差非数值). mm 级定位需求下 fp32 是免费午餐
- **tf32 不划算**: 矩阵太小(≤15), TensorCore 收益被 padding/截断抵消, 比 fp32 还慢且精度更差 → 微矩阵场景 TC 无用 (印证 auto-tuning 结论)
- ⇒ 后续 batch 并行默认采用 fp32

---

## v5 — 多轨迹 Batch 并行 Mega Kernel — 6.22 TODO#2 (系统级并行, 贡献点2)
代码: opt/ekf_v5.py (grid=(B,), 每 block 一条独立 scan), 结果: results/v5_batch_scaling.json

### 做法
- v4 kernel body 不变, 指针加 `pid*batch_stride` 偏移 → 每个 CUDA block 独立处理一条轨迹
- 场景: 蒙特卡洛 / 多传感器 / 多目标 (B 条轨迹相互独立, 天然并行)
- 复制同一轨迹 B 份验证 SM 扩展; **逐位一致 (traj_spread=0)** 确认 block 间无串扰

### 吞吐扩展 (full N=166667, fp32, GPU=78 SM)
| B | steps/s | traj/s | speedup vs B=1 | 备注 |
|---|---------|--------|------|------|
| 1 | 229k | 1.4 | 1.0x | =单条 (≈v4 fp32) |
| 16 | 3.67M | 22.0 | 16.0x | **完美线性** |
| 32 | 7.35M | 44.1 | 32.0x | 完美线性 |
| 64 | 14.66M | 88.0 | 64.0x | 完美线性 |
| 78 | 17.86M | 107.2 | 77.9x | =SM 数, 每 SM 一个 block |
| 128 | 24.07M | 144.4 | 105x | 进入第二波 (occupancy) |
| 256 | **34.08M** | 204.5 | **148.7x** | 接近寄存器 occupancy 上限 |

### SM Profiling (ncu/nsys, results/sm_profiling.json)
- **单 block(v4)**: SM throughput **0.27%**, warps_active **6.25%** (4/64 warp), 只占 1/78 SM → 单轨迹 latency-bound 的硬证据
- **B=78(v5)**: SM throughput **0.27%→19.91% (~74x)**, 78 block 铺满 78 SM
- 寄存器 110 regs/thread → occupancy_limit_registers=4 block/SM → 理论 ~312 条轨迹饱和, 解释 B≤256 仍近线性
- nsys: 整段 N 步 scan = **1 个 kernel launch** (mega-kernel 已无 launch 可省)

### 关键结论
- **B≤78 完美线性扩展 (speedup=B)**: 把"轨迹维"映射到 grid 是填满 SM 的正解; 单条串行无法并行, 但多条独立轨迹可
- v3(单条延迟极致) 与 v5(多条吞吐) **正交**: 前者降单条 latency, 后者升集群 throughput
- B=256 聚合 **34M 步/s = 相对 v0 CPU 18210x** (但这是吞吐量, 非单条加速; 单条仍受串行下限约束)

---

## 小矩阵特性分析 (Background 素材, results/small_matrix_characterization.json)
- 单步真实 FLOPs ≈ **25092** (8 个 ≤15×15 gemm + 3×3 解析逆 + 标量), 朴素实现却需 531 kernel launch
- 全局访存: 读 8 + 写 6 = 14 fp32/步 (56 B); state(P/q/pos/vel) 常驻寄存器零往返
- **GPU 不友好的本质** (论文 Background): ① 矩阵 ≤15, 单 tl.dot 用不满 1 warp, 算力利用 <1%; ② 卡尔曼递归严格串行, 时间维零并行; ③ 微矩阵+串行 = latency-bound, occupancy 天然 ≤6.25%/SM; ④ 朴素逐算子 launch/同步开销 >> 真实算力

---

## 三贡献点最终汇总 (6.22 TODO#7,#8)
| 贡献点 | 版本 | 核心结果 |
|--------|------|---------|
| ① 算子级 (Kernel fusion) | v1→v2→v3 | 531 微 kernel → 1 Triton kernel, 单条 fp64 8.8x |
| ② 系统级 (多轨迹 batch) | v5 | grid 映射轨迹维, B≤78 线性, B=256 聚合 34M 步/s |
| ③ 算法级 (混合精度) | v4 | fp32 单条 133x (fp32 vs fp64 = 14x), 精度亚毫米满足需求 |
最终报告: opt/FINAL_REPORT_v2.md
