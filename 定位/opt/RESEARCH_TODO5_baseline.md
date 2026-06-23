# TODO #5 调研报告：Baseline 与对比实验设计

> 对应 TODO.md 第 5 点。本报告分两部分：
> (A) **对比对象**——系统化 Benchmark 该跟谁比，怎么比；
> (B) **调研方向**——GPU 加速相关工作、高性能线代库、近几年论文与开源实现。
>
> **数据来源与可信度声明**：调研时公网被防火墙阻断（arxiv/github/google 均
> `ECONNREFUSED`，仅厂内 pip 镜像可达）。因此：
> - 标注 **[源码已核实]** = 从本机实际安装的包（triton 3.6.0 等）逐行验证；
> - 标注 **[知识/待联网核实]** = 来自模型知识，URL/卷期为高置信记忆但本次未能二次核验，**写入正式论文前请联网复核**；
> - 文末附「待核实清单」与「可复制检索式」。

---

## 0. 本项目问题画像（决定一切对比的前提）

| 维度 | 取值 | 来源 |
|------|------|------|
| 状态维度 | 15 (SINS 误差态) | ekf.py |
| 观测维度 | 固定 3 (双里程计+约束, 异常时 R→1e12 抑制) | v1/data_pattern |
| 时间步 N | 166667, dt 严格均匀, shape 全程静态 | data_pattern_analysis.json |
| batch | **1 (单条轨迹)** | — |
| 矩阵规模 | 全部 ≤15×15 稠密小矩阵 (含 3×3/4×4) | — |
| 依赖结构 | **严格串行 scan** (步 k 依赖步 k-1) | — |
| 瓶颈性质 | **latency-bound**: 单步 531 个微 kernel, mm 仅占~20% | v2 profile |
| 环境 | torch 2.11.0+cu129, **CUDA 12.9**, GPU **sm_90 (Hopper)** fp64 | 本机核实 |

**一句话定性**：这是一个「单轨迹、串行、极小矩阵、fp64」的滤波递归，对 GPU 的批吞吐
与算力极不友好；GPU 仅 1.48x（v2）正是这个本质的体现。这个画像同时决定了——
**大多数主流 GPU 滤波库（batched 设计）并不能直接解决我们的痛点**，这本身就是
立论点。

---

## (A) 对比对象（5 类，对应 TODO 清单）

### A.1 原始 PyTorch Eager —— ✅ 已有 (v0)
- `opt/ekf_baseline.py`，CPU full = **金标准** (1830 步/s, 91.09s)。
- RMSE 金标准: X=1017.838 Y=16.835 Z=8.141 ρ=0.1755%。所有版本以此为正确性验收。
- 同代码 `device='cuda'` = GPU eager 对照 (419 步/s, 反而比 CPU 慢 4x)。

### A.2 torch.compile —— ✅ 已有 (v1)
- `opt/ekf_v1.py` + `compile_mode='default'`。
- 结论已确认：**单步图太小，compile 反而更慢** (CPU 0.18x, GPU 比 eager 慢)。
- 作为 benchmark 的一行，用来**证伪**「compile 万能」的直觉。

### A.3 手工优化版本 —— ✅ 已完成 (v1 / v2 / **v3**)
- v1 eager (消除 .item()/碎片化): CPU 2079 步/s (1.14x)。
- v2 CUDA Graph (capture/replay): 2700 步/s (1.48x)。
- **v3 Triton 单 kernel mega-kernel: 20534 步/s, 11.3x (已完成, 单 block 串行上限)**。
  531 微 kernel → 1 kernel，正确性亚毫米级等价金标准。即 TODO#2 "Mega Kernel" 假设已被证实。

### A.4 相关高性能算子库 —— ⚠️ 已有微基准，但需重新定位
- 现有 `benchmark.py::lib_reference_bench()` 测了「单独 15×15 gemm / cuSOLVER inv3」
  的 GPU 单步耗时，作为 **launch-bound 下界参照**——这是对的，**保留**。
- **但要明确写清**：cuBLAS/cuSOLVER/MAGMA 的 **batched API 对本项目不适用**
  （理由见 B.3）。它们解决的是「大量独立小矩阵」，而我们是「单轨迹串行 batch=1」。
  把它们列为「评估后排除」的对象，而非可跑的 baseline。**这是审稿人会问的点，主动答。**

### A.5 学术界代表性实现 —— 🆕 本次调研补足
- 见 (B.1)：以 **dynamax (JAX) 的 parallel KF** 作为「associative-scan 跨时间步并行」
  范式的可读参考实现；以 **torchfilter (PyTorch)** 作为「主流 GPU 滤波库在单轨迹场景
  表现」的对照（预期它在 batch=1 下同样 latency-bound）。

### A.6 建议补充的两个对照轴（强化实验说服力）
1. **fp32 vs fp64**：数据 pattern 分析已指出收益集中在 fusion + 精度。加一行 fp32
   版（在金标准 tol 内）量化「精度换吞吐」，是最低成本的真实加速来源。
2. **CPU 多线程 / 单线程**：小矩阵串行在 CPU 上常常就是最优解之一。给出 CPU 上限，
   让「GPU 是否值得」这个问题有量化答案（目前 v1 CPU 2079 vs v2 GPU 2700，差距并不大，
   这个对比本身极有价值）。

---

## (B) 调研方向

### B.1 ⭐ GPU 上打破卡尔曼串行递归的根本方法：Associative-Scan 并行滤波

这是本次调研**最重要**的发现，直接对应 TODO「GPU 上的相关加速工作」+「近几年论文」。

#### 奠基论文
- **Särkkä & García-Fernández, "Temporal Parallelization of Bayesian Smoothers"**
  - arXiv:1905.13002；期刊版 **IEEE TAC 2021, 66(1):299-306** [知识/待联网核实卷期页]
  - arXiv 编号经 dynamax / TFP 源码内嵌引用确认 [源码已核实其被引用]
  - **核心思想**：把 KF 滤波/平滑递归改写成满足**结合律**的二元算子，再用并行前缀和
    (associative scan / Blelloch scan) 执行。串行深度 **O(N) → O(log N)**。
  - 对我们 N=166667：串行深度 166667 → `2·ceil(log2 N) ≈ 36` 层。**这是质变**，
    而非 v1/v2 的常数倍 launch 摊销。
  - 代价：(a) 算子内含多个小矩阵 solve/slogdet，单元素工作量 > 朴素一步；
    (b) 需一次性物化 N 个 15×15 消息（每个量约 166667×15×15×8B≈0.3GB，多个量数 GB，
    需估显存)；(c) PyTorch **无内置 associative_scan 原语**（JAX 有 `lax.associative_scan`）。

#### 非线性 (EKF) 扩展 —— 与我们直接相关
- **Yaghoobi, Corenflos, Hassan, Särkkä, "Parallel Iterated Extended and
  Sigma-Point Kalman Smoothers", ICASSP 2021**；arXiv:2102.00514 [知识/待联网核实]
  - **做法**：在标称轨迹上**线性化** → 得分段线性高斯模型 → 套用并行 scan；
    外层对「重新线性化」做迭代。即 **外层串行/少数几轮 + 内层时间并行 O(log N)**。
  - **限制（回答"EKF 是否适用"）**：结合律只在「给定一组固定 Jacobian F_k,H_k」后成立，
    所以非线性带来的轨迹依赖被推到外层迭代。我们的**数据相关分支**(|ΔD−ΔS|<thresh
    切 2/3 维) 必须先固定结构（沿用 v1 的 R=1e12 抑制法），否则算子无法向量化。
  - dynamax 现状 [源码已核实]：其 EKF/UKF (`inference_ekf.py`) **仍是串行 lax.scan**，
    **并行 scan 只覆盖线性 LGSSM**。非线性并行需按本文自行封装算子。

#### 可参考的开源实现
| 项目 | 出处 [核实状态] | 用途 |
|------|------|------|
| **dynamax** | github.com/probml/dynamax [源码已核实其 API] | `parallel_lgssm_filter/smoother`，最干净的 associative-scan KF 参考，可移植 |
| **TFP** `tfp.experimental.parallel_filter` | `parallel_kalman_filter_lib.py` [源码已核实] | 线性并行 KF，**原生支持 Cholesky 因子 scale_tril** (见 B.4) |
| **jax.lax.associative_scan** | jax 文档 [源码已核实签名] | 底层原语；要求结合律；支持 reverse(反向平滑) |

#### 相关性评估：**高**
并行的是**时间维**（我们有 16 万步），不要求多轨迹，正中「单轨迹串行」痛点。
**这是 v3 之后最值得尝试的算法级方向**，也是论文里区分「工程优化(v1/v2)」与
「算法级并行(associative scan)」的关键叙事。

---

### B.2 ⭐ Mega-Kernel 融合：把单步 531 微 kernel → 1（v3 已实现 11.3x）

对应 TODO #2，与 B.1 **正交可叠加**（理想终局：scan 压成 ~36 层，每层算子再用融合 kernel）。
**本项目 v3 已用 Triton 单 kernel 走通此路（20534 步/s，11.3x，单 block 串行上限）**，
以下是对该方向的完整调研，含「为何 v3 选 Triton 而非其他」与「再往上的 device 融合方案」。

#### 方案一：Triton 自研单 kernel（v3 已采用，当前环境最优解）
- **fp64 `tl.dot` 支持** [源码已核实 triton 3.6.0]：`semantic.py` dot() 允许 float64
  lhs/rhs 与 fp64 累加器。
- **关键约束** [源码已核实 `backends/nvidia/compiler.py:min_dot_size`]：
  ≥16bit 类型返回 `(M=1,N=1,K=16)`，即 **收缩维 K 必须 ≥16**。我们的 3×3/4×4/15×15
  全部 K<16 → 用 `tl.dot` 会被 padding 到 tensor-core 形状。
  → **v3 现状**：v3 把矩阵 padded 到 16 用 `tl.dot(fp64)`，**已跑通并拿到 11.3x**，
  说明 padding 路径在本规模下仍是净收益。**潜在微调**（非必须）：极小块 (3×3/4×4)
  改 **显式 FMA 展开** 可能再省 padding 浪费，但 v3 已逼近单 block 串行上限，收益有限。
- **跨时间步**：Triton 有 `tl.associative_scan` (`core.py`)，但卡尔曼递归非结合性
  (除非按 B.1 改写)，所以 Triton 的现实用法是「单 kernel 算一步消除 launch」+
  「kernel 内 `for` 串行推进多步（状态留寄存器，避免每步落盘）」，可把 16 万步压成极少 launch。
- 坑 [memory 已记]：`@triton.jit` 必须写在 `.py` 文件。

#### 方案二：cuBLASDx + cuSOLVERDx (NVIDIA MathDx, device-side)
- **唯一官方支持「单 kernel 内 GEMM + Cholesky/LU/solve 融合」**的方案，专为
  「减少 global memory 往返」设计。
  - cuBLASDx: device GEMM/TRSM，fp64 支持，sm_90 上 fp64 GEMM 尺寸上限约 196-240（我们远低于）。
    docs.nvidia.com/cuda/cublasdx
  - cuSOLVERDx: device POTRF/POSV/GETRF/GESV/QR，fp64 支持，可与 cuBLASDx 同 kernel 组合。
    docs.nvidia.com/cuda/cusolverdx
  - 求逆建议用 POSV/GESV 求解而非显式逆（数值更稳，与 v2 自写 inv3x3 方向一致）。
- **⚠️ 版本陷阱（决定性）**：cuSOLVERDx 0.4.0+ 与 cuBLASDx 0.6.0 **要求 CUDA Toolkit 13.0+**，
  当前是 **CUDA 12.9**。用 device-solver 基本需升级 CUDA 13.x。Python 可经
  nvmath-python / NVIDIA Warp 接入。
- 相关性：**高（上限最高）**，但有 CUDA 升级成本。

#### 方案三：CUTLASS / CuTe DSL —— 中
- 可在 fused kernel 内做小 GEMM（fp64 tensor-core），CUTLASS 4 有 Python `nvidia-cutlass-dsl`。
- **局限**：只有 GEMM，**无 solver**；3×3/4×4 极小矩阵 tensor-core 收益有限。性价比 < 前两者。

---

### B.3 高性能线代库逐一结论（对应 TODO「求解器/cuBLAS/cuSOLVER/cuSPARSE」）

| 库 | fp64 | in-kernel 融合 | 对本项目相关性 | 结论 |
|----|------|----------------|----------------|------|
| **cuBLASDx / cuSOLVERDx** | ✅ | ✅ 唯一官方支持 | **高** | 上限最高，但需 CUDA 13 |
| **Triton** | ✅ (tl.dot, 但 K≥16) | ✅ 自研 | **高** | 当前环境最务实，极小矩阵用 FMA 展开 |
| **CUTLASS / CuTe** | ✅ | 仅 GEMM | 中 | 缺 solver，极小矩阵收益低 |
| **cuBLAS/cuSOLVER batched** (`gemmStridedBatched`/`getrfBatched`/`potrfBatched`) | ✅ | ❌ host-launched | **低** | 解决「多独立小矩阵」，我们 batch=1 串行用不上 |
| **MAGMA batched** | ✅ | ❌ host-launched | **低** | 同上，且是额外第三方依赖 |
| **cuSPARSE** | ✅ | — | **低** | H 后9列恒0 但矩阵太小；稀疏开销 > 收益。**正确做法是手写 kernel 里裁剪掉这9列乘加（常量折叠），不引入 cuSPARSE** |

> **batched API 为何不适用（需在论文明确）**：batched 的价值是一次 launch 算完 N 个
> *相互独立* 的小矩阵。我们是单轨迹严格串行（步 k 依赖 k-1），每步 batch=1，
> 既不能跨时间步并行（递归依赖），单步内几十个操作也仍是独立 kernel——无法解决
> latency-bound 根因。**唯一翻盘场景：跑大量独立轨迹/多 seed 蒙特卡洛**，那时 batch
> 维 = 轨迹数，strided-batched / potrfBatched 才合适（但那改变了问题定义）。

---

### B.4 数值稳定：Square-root / Cholesky 形式（配套保险）

- 并行 scan 算子内出现 `(I+CJ)` 求解与 `slogdet`，协方差正定性比串行更脆弱。
- 我们的协方差跨 6 个量级（X 轴 ~1m vs Z 轴 ~8mm），属病态场景。
- **TFP parallel filter 已示范** [源码已核实]：接受 `scale_tril`（三角因子）输入，
  采样过程原生用 Cholesky 因子——是平方根 + 并行 scan 结合的现成工程例子。
- 平方根并行 SLR 论文（Yaghoobi 等，约 2022）[知识/待联网核实标题年份]。
- 相关性：**中→高**（并行版若数值崩溃，这是必备手段，确保 RMSE 仍对齐金标准）。

---

### B.5 开源滤波库总览（候选 baseline / related work）

| 项目 | URL [待核实] | 真 GPU | batched | EKF | 相关性 | 用途 |
|------|------|--------|---------|-----|--------|------|
| **torchfilter** (Stanford IPRL) | github.com/stanford-iprl-lab/torchfilter | ✅PyTorch | ✅ | ✅ | 高 | **首选 baseline**，技术栈一致 |
| **dynamax** (probml) | github.com/probml/dynamax | ✅JAX | ✅vmap | ✅+并行KF | 高 | associative-scan 参考实现 |
| **KalmanNet** | github.com/KalmanNet/KalmanNet_TSP | ✅PyTorch | ✅ | KF变体 | 中-高 | 学习型增益对照 |
| **filterpy** (R.Labbe) | github.com/rlabbe/filterpy | ❌CPU | ❌ | ✅ | 中 | 数值金标准交叉验证 |
| **pykalman** | github.com/pykalman/pykalman | ❌CPU | ❌ | ✅ | 低-中 | 同上 |
| **simdkalman** | github.com/oseiskar/simdkalman | ❌(numpy向量化) | ✅多轨迹 | ❌线性 | 中 | 向量化思路 |
| **TFP** `parallel_filter` | tensorflow_probability | ✅ | ✅ | KF(EKF需手搭) | 中 | 线性并行+Cholesky 参考 |

> **重要观察（立论点）**：**几乎所有 "GPU Kalman" 开源实现都走 batched 多轨迹路线，
> 缺少专门面向"单轨迹串行 latency-bound"的 CUDA/Triton EKF 实现**。这个空白本身就是
> 本工作的 related-work 立足点（请联网确认有无新近小项目）。

---

### B.6 INS/GNSS 组合导航的 GPU 加速现状

- **诚实结论**：EKF 形式的 INS/GNSS 直接上 GPU 的代表性论文**很稀少**——正因状态维小、
  串行递归，GPU 收益有限。工程主流是 **FPGA/DSP/MCU** 流水线并行，而非 GPU。
- GPU 在导航里的常见切入点是**粒子滤波**（成百上千粒子天然 batch 并行），属 PF 非 EKF。
- 「单轨迹 INS/GNSS EKF 鲜有专门 GPU 工作」本身是有力的 related-work 陈述。
  [此结论为知识判断，务必联网核实有无反例]

---

### B.7 相邻领域：SSM 的 parallel scan（方法同源，工程标杆）

- **S5** (Smith, Warrington, Linderman, ICLR 2023) — 显式用 parallel associative scan
  处理线性 SSM，与并行 KF 数学同源。
- **S4** (Gu, Goel, Ré, ICLR 2022) / **Mamba** (Gu & Dao, 2023) — Mamba 的
  hardware-aware **selective scan** 用 Triton/CUDA 实现，是「序列递归在 GPU 上高效
  并行扫描」的工程标杆，可借鉴其 Triton scan 实现。
- **Blelloch (1990)** parallel prefix scan — 所有 associative scan 的理论源头。
- 相关性：中-高（B.1 的工程实现可借鉴 Mamba 的 Triton scan）。

---

## (C) 最终建议

### C.1 Benchmark 该怎么搭（更新 benchmark.py 的方向，不是现在改）
现有 `benchmark.py` 已覆盖 v0/v1/v2 + lib_reference，骨架良好。建议补三行对照：
1. **CPU 单/多线程上限**（让「GPU 是否值得」有量化答案——目前 v1 CPU 2079 vs v2 GPU 2700）；
2. **fp32 版**（精度换吞吐，最低成本真实加速）；
3. **（可选）torchfilter 单轨迹**（验证主流 GPU 库在 batch=1 下同样 latency-bound）。
并把 cuBLAS/MAGMA batched / cuSPARSE 明确列为「评估后排除」（附理由 B.3）。

### C.2 优化路线（按投入产出排序，v0~v3 已完成）
**已完成**：v1 算子级 → v2 CUDA Graph (1.48x) → **v3 Triton 单 kernel (11.3x，单 block 串行上限)**。
TODO#2 "Mega Kernel" 已被 v3 证实。后续方向：
1. **Associative-scan 并行 EKF (v4, 算法级质变)** — O(N)→O(log N)，是 v3 之后**唯一剩下的质变方向**
   （v3 是把串行做到极致，v4 是打破串行本身，二者正交）。外层迭代线性化 + 内层时间并行；
   先固定观测结构(R=1e12)；PyTorch 无原语 → 自写 Blelloch scan / 借 Triton / 或迁 JAX 验证。
   配 Cholesky 平方根保数值。代价：物化 N 个 15×15 消息（数 GB 显存）。
2. **cuBLASDx+cuSOLVERDx (v5, device 融合上限)** — 仅当愿升级 CUDA 13；GEMM+solver 全融合，自带稳健分解。
3. **微调** — v3 极小块 FMA 展开 / fp32 精度换吞吐 / 多轨迹 batch 填满 SM。

### C.3 论文必引 related work
1. Särkkä & García-Fernández 2021 (IEEE TAC) — 并行贝叶斯滤波，理论基石。
2. Yaghoobi et al. 2021 (ICASSP) — 并行 EKF/sigma-point，与我们 EKF 直接对应。
3. S5 / Mamba / S4 — SSM parallel/selective scan，序列递归 GPU 并行现代标杆。
4. MAGMA/cuBLAS batched + CUDA Graphs — 小矩阵 latency-bound 方法论与工程优化引用支撑。

---

## (D) 待联网核实清单（公网被墙，写论文前务必复核）

| # | 待核实项 | 当前状态 |
|---|----------|----------|
| 1 | arXiv:1905.13002 TAC 2021 卷期页 66(1):299-306 | arXiv号经源码引用确认，卷期待核 |
| 2 | arXiv:2102.00514 (Parallel Iterated EKF/Sigma-Point) 标题/作者/年 | 待核 |
| 3 | 平方根并行 SLR 论文 (Yaghoobi 等 ~2022) 标题年份 | 待核 |
| 4 | torchfilter / dynamax / KalmanNet 等 GitHub URL 与现状 | URL 为记忆形式，待核 |
| 5 | 是否存在「单轨迹 INS/GNSS EKF GPU」反例工作 | 当前判断为空白，待核 |
| 6 | cuBLASDx/cuSOLVERDx 对 CUDA 13 的硬要求版本号 | 待核 |

### 可复制检索式
```
Sarkka "Temporal Parallelization of Bayesian Smoothers" IEEE TAC 2021
Yaghoobi Corenflos Sarkka "Parallel Iterated Extended" Kalman ICASSP 2021
"torchfilter" pytorch differentiable filter site:github.com
"dynamax" probml jax parallel kalman site:github.com
Mamba "Selective State Spaces" Gu Dao selective scan triton
MAGMA "batched" small matrix GEMM LU Dongarra Haidar
cuBLASDx cuSOLVERDx device BLAS solver fuse kernel CUDA 13
INS GNSS EKF GPU CUDA acceleration integrated navigation
```
