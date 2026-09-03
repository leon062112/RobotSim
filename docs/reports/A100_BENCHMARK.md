# A100 全量复跑报告（TRIDENT 三轴框架跨硬件验证）

> 日期：2026-09-01 · 机器：NVIDIA A100-SXM4-80GB / **108 SM**（compute 8.0）
> 软件：torch 2.9.1+cu128 · triton 3.5.1 · CUDA 12.8 · driver 470.82.01
> 数据源：`data/results/a100_benchmark_summary.json`（全量 N=166667；Triton/CUDA Graph 各 config 先 warmup 再 ≥3 次计时）
> 本报告**只覆盖 A100**，与 H20（78 SM）旧数字并存，不互相覆盖。

## 0. Golden 复现核对

`fangzhen.py::generate_trajectory(seed=42)` 在 A100 机（torch 2.9.1，128 线程默认）逐位复现金标准：

| 指标 | 本机实测 | GOLDEN 常量 | 结果 |
|---|---|---|---|
| RMSE X (mm) | 1017.838305 | 1017.838305 | ✅ 逐位一致 |
| RMSE Y (mm) | 16.834549 | 16.834549 | ✅ |
| RMSE Z (mm) | 8.140964 | 8.140964 | ✅ |
| rho (%) | 0.175540 | 0.175540 | ✅ |

---

## 1. 单轨迹阶梯（A100 全量 N=166667）

| 版本 | 方法 | Steps/s | vs CPU | 备注 |
|---|---|---|---|---|
| v0 | CPU eager fp64 | 1,004 | 1.0× | 金标准 |
| v0 | GPU eager fp64 | 306 | **0.30×** | partial n=20001，GPU 比 CPU 慢 3.3× |
| v1 | GPU eager（消碎片） | 490 | 0.49× | partial |
| v1 | torch.compile(default) | 1,343 | 1.34× | partial；18 dispatch/步 |
| v2 | CUDA Graph fp64 | 2,476 | 2.47× | 全量 |
| v3 | Triton 单 kernel fp64 | **102,575** | **102.2×** | vs GPU eager 335×；RMSE 与金标准 Δ<0.01mm |
| v4 | Triton 单 kernel fp32 | **181,028** | **180.4×** | vs fp64 **1.76×**；ΔX≈0.11mm |
| v4 | Triton 单 kernel tf32 | 181,599 | 180.9× | tf32 吞吐≈fp32，精度下降（见下） |

**三轴分解（A100）**：

- **融合轴**（v0 GPU eager → v3 fp64）：**335×**（消除 531 kernel/步的 launch 链）；或 vs CPU 102×。
- **精度轴**（fp64 → fp32）：**1.76×**（远小于 H20 的 14.3×，见 §4）。
- **并行轴**（B=1 → B=108）：**~108× 完美线性**（填满 108 SM，见 §2）。

**tf32 负结果（A100 版）**：tf32 吞吐与 fp32 基本持平（1.003×），但 X 轴 RMSE 增大 0.74mm、
Z 轴增大 0.27mm —— 即 **无吞吐收益 + 有精度损失**。对比 H20 版（tf32 更慢且 ΔX≈4.7mm），
结论方向一致、量级不同：对 ≤15×15 微矩阵，TensorCore/tf32 无利可图的具体表现与硬件相关。

---

## 2. Batch 扩展（A100，fp32，全量）

### 2.1 replicate 列（同一轨迹复制 B 份，测 SM 扩展）

| B | Steps/s | vs B=1 | per-traj steps/s |
|---|---|---|---|
| 1 | 181,093 | 1.0× | 181,093 |
| 16 | 2,895,064 | **16.0×** | 180,942 |
| 32 | 5,790,817 | **32.0×** | 180,963 |
| 64 | 11,579,744 | **63.9×** | 180,933 |
| **108** | **19,539,160** | **107.9×** | 180,918 |
| 128 | 19,603,255 | 108.2× | 153,150 |
| 256 | 32,088,946 | 177.2× | 125,348 |
| **432** | **44,445,029** | 245.4× | 102,882 |
| 512 | 44,064,992 | 243.3× | 86,065 |

**三区扩展模型在 A100 上验证成立**：

- **线性区** B ≤ 108（= SM 数）：per-traj 稳定在 ~181k steps/s，B 扫 → 16×/32×/64×/108× **完美线性**。
- **次线性区** 108 < B ≤ 432：per-traj 从 181k 降到 ~103k，寄存器资源竞争开始。
- **饱和区** B ≈ 432：峰值 44.4M steps/s；B=512 绝对值下降（44.1M）。

**关键验证**：`B_sat ≈ S·C_reg = 108 × 4 = 432` 精确命中峰值点——与 H20 的 `78 × 4 = 312` 同源
（110 regs/thread → 每 SM 4 block），证明该批扩展模型**跨硬件可移植**。

### 2.2 independent 列（B 条不同 seed 的蒙特卡洛轨迹）

| B | Steps/s | RMSE X（mean±std, mm） | 与 replicate 差异 |
|---|---|---|---|
| 1 | 181,110 | 1017.7±— | 逐位一致（seed=42 金标准） |
| 16 | 2,895,064 | 987.6±101.8 | 吞吐一致 |
| 32 | 5,596,757 | 1008.0±112.9 | 吞吐一致 |
| 64 | 11,244,987 | 1022.3±102.7 | 吞吐一致 |
| 108 | 19,001,293 | 1008.9±97.0 | 吞吐一致 |

**结论**：independent 与 replicate 吞吐**几乎相同**（kernel 无数据相关分支、定长定工作），
但独立轨迹的跨轨迹 RMSE 有真实分布（X ~ ±100mm 来自不同障碍实现），证实这是**真正的蒙特卡洛**，
而非同一轨迹的 B 份拷贝（replicate 的 `traj_spread=0`）。

---

## 3. batched 库微基准（cuBLAS / cuSOLVER）

| 指标 | 值 | 含义 |
|---|---|---|
| single_inv3_us | 110.3 µs | 单次 3×3 逆（cuSOLVER getrf）的 launch 下界 |
| single_gemm15_us | 11.4 µs | 单次 15×15 gemm（cuBLAS）的 launch 下界 |
| batched_inv3_us_per_op_B256 | 0.35 µs | B=256 批量摊薄后单条 3×3 逆 |
| batched_gemm15_us_per_op_B256 | 0.063 µs | B=256 批量摊薄后单条 15×15 gemm |

**解读**：单次库调用 launch-bound（110µs/11µs），batch 摊薄后 per-op 降到 ~0.06µs。单步 EKF ≈ 8 gemm + 1 inv
⇒ 批量化库只算线代部分 ≈ 0.8µs/步（对比 fused kernel 整步 5.5µs = 1/181k）。但 batched 库**无法表达整步**：
sin/cos/atan2 等 transcendental、四元数归一化、反馈、协方差递推仍需 531 个微 kernel 或手写；且单条轨迹的
串行依赖无法跨步 batch —— 这正是「批量化库调用 vs 单 kernel 融合」的工程分界。

---

## 4. ⚠️ 跨硬件重大发现：fp32「免费午餐」是 Hopper 特例

| 指标 | H20（论文现数字） | A100（本报告） |
|---|---|---|
| SM / fp64 能力 | 78 / 弱 fp64 | 108 / fp64 强（≈1:2 vs fp32） |
| v3 fp64 (steps/s) | 16,491 | **102,575**（6.2× 更快） |
| v4 fp32 (steps/s) | 242,737 | 181,028 |
| **fp32 vs fp64** | **14.3×** | **1.76×** |
| v3 vs CPU | 8.8× | 102.2× |

**结论**：论文 headline 的 **「fp32 免费午餐 14.3×」是 Hopper（弱 fp64）特有现象**。A100 的 fp64 ALU
吞吐约为 fp32 的 1/2（远强于 H100），串行 latency-bound 下 fp64 的 transcendental/寄存器压力没被同等放大，
因此 fp32 只带来 1.76× 增量。

相反，**融合轴在 A100 上更强**（A100 fp64 快 → v3 fp64 = 102k 步/s = 102× vs CPU，而 H20 仅 8.8×）。
三轴乘积在 A100 ≈ 102× × 1.76× × 108× ≈ 19,000×，与 H20 的 ~9,800–18,000× 同一量级，
**但分解不同**：A100 融合主导、H20 精度主导。

**写作建议**（供作者决策）：

1. 若 A100 作主硬件：§4 精度章主数字 14.3× 改为 1.76×，标题由「free lunch」改为「组件感知精度的泛化收益」；
   tf32/bf16 负结果方向不变、量级改为 A100 实测。
2. 若 H20 为主：A100 作为 **cross-hardware portability** 一节，证明 (a) 三轴框架与 `B_sat≈S·C_reg`
   模型跨卡成立；(b) 精度轴收益量级**硬件相关**（fp64:fp32 比决定）——这本身强化「组件感知·不该一刀切」论点。

---

## 5. §5 Evaluation 两表草稿（A100 版，可直接回填 main.tex）

### 表 A：端到端单轨迹（全量，A100 / 108 SM）

| Version | Method | Steps/s | Speedup vs CPU | Accuracy |
|---|---|---|---|---|
| v0 | CPU eager fp64 (golden) | 1,004 | 1.0× | RMSE X=1017.8 mm |
| v0 | GPU eager fp64 | 306¹ | 0.30× | — |
| v1 | torch.compile (Inductor) | 1,343¹ | 1.34× | — |
| v2 | CUDA Graph fp64 | 2,476 | 2.47× | 逐位等价 |
| v3 | Triton fused fp64 | 102,575 | 102× | Δ<0.01 mm |
| v4 | Triton fused fp32 | 181,028 | 180× | 亚毫米（ΔX≈0.11 mm） |
| v4 | Triton fused tf32 | 181,599 | 181× | ΔX≈0.74 mm（无吞吐收益） |

¹ v0 GPU eager 与 v1 用 partial n=20001（全量 eager GPU 代价过高）；其余全量。

### 表 B：batch 扩展（fp32，全量；replicate / independent 吞吐一致，列 replicate）

| B | 1 | 16 | 32 | 64 | 108 | 256 | 432 | 512 |
|---|---|---|---|---|---|---|---|---|
| Steps/s (M) | 0.18 | 2.90 | 5.79 | 11.6 | 19.5 | 32.1 | 44.4 | 44.1 |
| 三区 | 线性 | 线性 | 线性 | 线性 | 线性(SM) | 次线性 | 饱和 | 饱和(降) |

> B_sat = 432 = 108 SM × 4 blocks/SM（register 约束），与 H20 78×4=312 同模型。

---

## 6. 附带结论

- **torch.compile 三重差异实锤**：单步 EKF → 18 dispatch（17 Triton pointwise + cuBLAS/cuSOLVER），
  matmul 走 `extern_kernels.mm/addmm`、逆走 `aten.linalg_inv_ex`，**不生成 tl.dot**。
  详见 `docs/paper-planning/baseline_tech_memo.md` §2。
- **CUTLASS 不单开**：cuBLAS 内嵌 CUTLASS，batched cuBLAS 覆盖该轴即够（memo §3）。
- **输入约定**：batch 轴已同时提供 replicate（SM 扩展）与 independent（真实蒙特卡洛）两列
  （memo §5）。

## 附：数据文件

- `data/results/a100_benchmark_summary.json` — 唯一数据源（含 full hardware 块 + 每 config 的 reps）
- `data/trajectory/traj_batch_fp64.npy` — 108 条独立轨迹 (108,166667,15) fp64
- `code/optimization/benchmark_a100.py` — 复现脚本（`--quick` 冒烟，`--skip-cpu-full` 跳过 3min CPU）
- `docs/paper-planning/baseline_tech_memo.md` — 四点技术调研结论