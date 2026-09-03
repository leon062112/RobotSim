# Baseline 技术调研备忘：torch ↔ 算子库、torch.compile、CUTLASS 与输入约定

> 日期：2026-09-01 · 目标机器 A100-SXM4-80GB / 108 SM（torch 2.9.1+cu128 / triton 3.5.1 / CUDA 12.8）
> 本文固化 Plan（`docs/paper-planning/baseline_experiment_design.md` 的补充）中四个技术点的调研结论，
> 供撰文与后续实验引用。关键结论均在本机实测或核对到代码/数据。

---

## 1. torch 是否与 cuBLAS / cuSOLVER 重叠？——**是，且 torch.compile 同样重叠**

`torch.matmul`/`@` 在 CUDA 下底层调度到 **cuBLAS**；`torch.linalg.inv`/`solve` 底层调度到
**cuSOLVER**（batch 情形走 MAGMA/getrf）。现有代码 `code/optimization/benchmark.py::lib_reference_bench()`
里的 "cuSOLVER inv3" 就是直接计时 `torch.linalg.inv(S)`（`benchmark.py:50`），说明该"算子库微基准"
与 torch eager 是**同一条代码路径**。

实测 torch.compile 的 Inductor 输出进一步印证：fp64 的 `F@P`、`P@Hᵀ`、`K@S` 被降级为
`extern_kernels.mm` / `extern_kernels.addmm`（= cuBLAS），3×3 逆走 `torch.ops.aten.linalg_inv_ex`（= cuSOLVER），
**不生成 `tl.dot`**。即 torch.compile 也**没有逃离 cuBLAS/cuSOLVER 依赖**，只是把点式算子重排在其周围。

**结论**：「torch 内嵌算子库」不能算独立基线。真正独立的库级基线是 **batched** 路径：
`gemmBatched` / `potrfBatched`（或 `torch.linalg`/`torch.matmul` 的 batched 张量形式），它剥掉
Python dispatch / 逐 op launch 开销、保留库的固定代价。这与
`baseline_experiment_design.md` 已写明的「batched cuBLAS/cuSOLVER 替代单个 getrf」一致。

---

## 2. torch.compile 是否生成 Triton、与 whole-scan 的差异？——**生成，但仅覆盖点式碎片**

在本机 A100 对 `ekf_v1.py::ekf_step` dump Inductor 输出，单步 EKF 编译结果为：

- **18 个 dispatch** = 17 个 `triton_poi_fused_*`（pointwise kernel）+ 若干 `extern_kernels` 调用；
- 17 个 Triton kernel 只覆盖 elementwise 碎片（quaternion/skew/norm/diag_embed/stack/cat），
  每个 `xnumel` 仅 4~12600，彼此通过全局内存缓冲 `buf0..buf33` 来回传递；
- fp64 矩阵乘与求逆**未被融合**，仍落回 cuBLAS/cuSOLVER（见 §1）。

对照 Trident whole-scan（v3/v4），差异有三重，比「step-level vs trajectory-level」更具体：

| 维度 | torch.compile（v1） | whole-scan（v3/v4） |
|---|---|---|
| dispatch 次数/步 | 18（17 Triton + cuBLAS/cuSOLVER） | 1（全程单 launch） |
| 矩阵乘路径 | `extern_kernels.mm/addmm`（cuBLAS 库） | `tl.dot`（融合，寄存器内） |
| 递归状态位置 | 全局内存 `buf*` 跨 kernel 往返 | 寄存器常驻，零全局往返 |

> 注：`main.tex` 的 `\Cref{subsec:related-fusion}` 已讨论 torch.compile / Welder / Chimera 等
> graph-level fusion，可在此补充「torch.compile 对 fp64 微矩阵的 extern 降级」作为 step-level
> auto-fusion 边界的硬证据。

---

## 3. CUTLASS 能否作为 baseline？——**技术上能，但冗余，不单开**

- CUTLASS 面向大 tile 吞吐（128×128+、warp specialization、tensor core），对 15×15（pad 到 16）
  无法摊销开销；且表达不了 sin/cos/atan2 等 transcendental、3×3 解析逆、串行 scan，只能覆盖 8 个 matmul。
- 更关键：**cuBLAS 内部多处即 CUTLASS** 实现，单写 CUTLASS kernel ≈ 重写 cuBLAS 已做的路径。
- 本机 `nvidia-cutlass 4.7.0` / `cutlass` 已装，若需「手写 CUDA whole-scan 对照」可用 cupy `RawKernel`
  或 cutlass-dsl（回应"加速来自 Triton 还是融合本身"），属**可选**，不列入核心基线。

**结论**：库级 GEMM 轴用 **batched cuBLAS**（= CUTLASS 的生产前端）覆盖即够。

---

## 4. 基线是否太少？——**否，完整族 4 类约 9 个对照点**

cuBLAS/cuSOLVER 在两类里出现、但角色不同：「torch 内嵌」是隐式底层，「batched 库组合」是
显式工程策略（把 8 个 matmul + 1 个逆按批批量调用），后者即使复用 cuBLAS 仍是独立合法基线，
因为它测的是「批量化库调用 vs 单 kernel 融合」这条**工程路线**。

| 类别 | 对照点 | 服务贡献点 |
|---|---|---|
| 同算法·实现阶梯 | CPU eager / GPU eager / torch.compile / CUDA Graph / **batched 库组合** / 手写 CUDA whole-scan(可选) / Trident whole-scan | ① fusion |
| 替代算法 | Särkkä prefix-sum 并行滤波（O(N log N)、log 深度） | 路线选择 |
| 替代系统 | torch-kf / batched eager / FastTrack·Ai et al. 的 batched-KF | ③ batch 轴 |
| 技术消融（负结果） | TF32/TC、TMA、warp-spec、cuBLASDx/cuSOLVERDx | "非堆 SOTA" |

**真正不单开**：单独 `getrf` 微基准（launch 下界，用于标注 `potrfBatched` 对比）；独立 CUTLASS kernel。

---

## 5. 横坐标 / 输入约定（回应"展示性能数据时横坐标是什么"）

现状三张性能图的**自变量（x 轴）**，与各自底层**数据输入**：

| 图 | x 轴（自变量） | 底层数据输入 | 备注 |
|---|---|---|---|
| 版本阶梯图 | 方法名（v0..v7） | 单条 seed=42 轨迹（N=166667） | `benchmark_summary.json` |
| 精度权衡图 | 精度方案（fp64/fp32/io-fp16/dot-tf32/io-bf16/state-fp16） | 单条轨迹 | 吞吐 bar + 相对 ΔX/ΔZ line |
| batch 扩展图 | B（轨迹数） | **同一条轨迹复制 B 份** | `v5_batch_scaling.json` 的 `max_traj_spread_m=0.0` 证实 |

**关键改造**：batch 轴当前 `replicate_input=True`（复制），
`v5_batch_scaling.json` 里 `max_traj_spread_m=0.0` 证明 B 条完全相同——这不是蒙特卡洛。
本轮改为 **B 条独立 seed 轨迹**（`fangzhen.py::generate_trajectory(seed)` 天然支持），
使 batch 图的 x=B 真正代表"B 条独立蒙特卡洛输入"。
改后三元图各有清晰自变量与输入：精度图(x=精度, 输入固定单条)、batch图(x=B, 输入=B 条独立轨迹)、
版本图(x=方法, 输入=单条)。

---

## 6. A100 vs H20 环境差异（影响实验与写作）

| 项 | H20（论文现数字） | A100（本轮） |
|---|---|---|
| SM 数 | 78 | **108** |
| batch 饱和点 `B_sat≈S·C_reg` | 78×4 = 312 | **108×4 ≈ 432** |
| fp64:fp32 CUDA-core 比 | 弱 fp64（Hopper） | **≈1:2**（A100 fp64 较强） |
| fp64 tensor core | 无实际 tl.dot fp64 加速 | **有 fp64 MMA** |
| torch/triton | 2.11.0 / 3.6.0 | 2.9.1 / 3.5.1 |

**影响**：v3 fp64 与 v4 fp32 的倍数**很可能不同于 H20 的 14.3×**（H20 的 14.3× 主因 fp64
transcendental latency + 寄存器压力，A100 上 fp64 相对更强，倍数可能收窄），**以本轮实测为准**；
batch 扫描须到 ~432 才覆盖饱和点。报告须标注 "A100 / 108 SM"，不与 H20 旧数字混淆。