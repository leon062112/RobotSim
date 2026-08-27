# Baseline 对比实验设计

> 状态：方案已确定（2026-08-27），待 H20 / A100 环境就位后执行。
> 当前环境不具备 H20 / A100，本文件只固化设计，不包含实测数字。

## 1. 本轮决策结论

| 候选 baseline | 结论 | 理由 |
|---|---|---|
| torch eager | **保留**，端到端核心 baseline | 全文加速比的根基；"naive GPU port 慢于 CPU"是关键动机证据 |
| prefix-sum 并行滤波（Särkkä 2025） | **新增**，核心学术 baseline | 唯一能在同一 SINS/EKF workload 上 head-to-head 的学术方法 |
| batched cuBLAS / cuSOLVER | **替代**单个 `cuSOLVER getrf` | 单 getrf 测的是 launch 开销而非求解器；batched 路径才是库的正确打开方式 |
| CAKE | **移除** | 与本文 workload 无关（compiler-agent，B200 大算力 kernel） |

## 2. Baseline 清单与定位

每个 baseline 与论文三大贡献点的对应关系：

| 类别 | Baseline | 对标贡献点 | 论证什么 |
|---|---|---|---|
| 端到端（单轨迹） | v0 eager CPU / GPU；v1 eager；torch.compile；CUDA Graph；Triton fused fp64 / fp32 | ① fusion | 融合把 531 kernel/步降到 1 launch；naive port 反被 CPU 吊打 |
| 学术方法 | Särkkä 2025 prefix-sum 并行滤波（JAX `associative_scan` 或手写 2×2 算子） | ① 路线选择 | 融合（保留串行语义）vs 并行-in-time（对数深度），本 workload 选前者 |
| 库级组件 | batched cuBLAS `gemmBatched`；cuSOLVER `potrfBatched` | ① 组件选型 | 解析 3×3 逆 + 融合优于库调用；库 batched 路径仍受 launch/调度限制 |
| batch 扩展 | `torch-kf`（batched KF 库）/ batched eager | ③ 多轨迹并行 | 你的 v5/v7 相较通用批库的吞吐优势 |

## 3. 各 baseline 实现与测量协议

### 3.1 torch eager（补全）

- **现状**：GPU eager 仅跑了 `n=20001`（partial），rho=1.72% 与全量 0.17% 不可比，RMSE 不可跨步数比较。
- **动作**：H20 / A100 上跑**全量 166667 步**（v0 eager GPU，代价约 6–7 min/机）。v1 eager、torch.compile 同步补全或统一 partial 标注。
- **指标**：`throughput_steps_per_s`、RMSE x/y/z、`rho_pct`，对照 fp64 CPU 金标准（`GOLDEN` 常量）。

### 3.2 prefix-sum 并行滤波（核心学术 baseline）

- **复现路径**：Särkkä et al. 2025, "On The Performance of Prefix-Sum Parallel Kalman Filters"（arXiv:2511.10363）。实现可取两条路：
  1. JAX `jax.lax.associative_scan` + 手写 2×2 组合算子（推荐，改动小、可复现）；
  2. 直接手写 CUDA/Triton 的 2×2 块 prefix-scan。
- **关键限制（必须在论文中如实表述）**：prefix-sum 需要转移/观测矩阵**预先固定**，而本 workload 的 `F_k`、`H_k` 依赖当前姿态与量测有效性决策（数据相关分支）。两条可行路径：
  1. 先用 nominal/EKF 线性化点冻结 `F_k, H_k`（**线性化近似**，RMSE 会偏离金标准）——这是可执行的 head-to-head；
  2. 仅在"无需反馈的线性化 EKF"子问题上对比吞吐。
- **指标**：吞吐（steps/s）+ RMSE x/y/z（标注"线性化近似，非严格同位语义"）。
- **预判**：组合矩阵从 15×15 膨胀到 30×30 块，单 prefix 元素的乘法成本远大于单步 EKF，且 work 由 O(N) 增至 O(N log N)；本 workload 是 latency-bound + 微矩阵，并行-in-time 很可能**不划算**——这恰好支撑"融合 + 多轨迹并行"的路线选择。

### 3.3 库级组件（替代 cuSOLVER）

- **正确路径**：`cublasDgemmBatched`（批量 15×15）、`cusolverDnDpotrfBatched`（批量 3×3 Cholesky，S 为对称正定）。
- **对照**：解析 3×3 逆（adjugate/det）vs `potrfBatched` vs `getrfBatched`。
- **结论点**：论证"解析逆 + 融合"优于库调用；同时指出单个 `getrf` 的 134μs 是 launch-bound 下界而非算法本身（避免审稿人反咬）。

### 3.4 batch 扩展库级

- **`torch-kf`**（github raphaelreme/torch-kf）：batched Kalman filter/smoother 的 PyTorch 库，测 batch 吞吐与你的 v5/v7 对比。
- **batched eager**：v0/v1 CPU/GPU 的多轨迹 batch 版本作为实现层参照。

## 4. 表格模板（对应 main.tex `sec:evaluation` 三/四张表）

**表 1 端到端单轨迹（fp64 金标准对照）**

| 版本 | 方法 | Steps/s | 加速比 | RMSE X/Y/Z | rho% |
|---|---|---|---|---|---|
| v0 | CPU eager fp64 | | 1.0× | | |
| v0 | GPU eager fp64（全量） | | | | |
| v1 | torch.compile GPU | | | | |
| v2 | CUDA Graph fp64 | | | | |
| v3 | Triton fused fp64 | | | | |
| v4 | Triton fused fp32 | | | | |

**表 2 prefix-sum 并行滤波 vs 本文融合（单轨迹，同 166667 步）**

| 方法 | 实现 | Steps/s | 加速比 | RMSE 偏离 | 语义备注 |
|---|---|---|---|---|---|
| prefix-sum 并行滤波 | JAX associative_scan | | | | 线性化近似 |
| 本文 fused scan | Triton fp32 | | | | 原始顺序 |

**表 3 库级组件（微基准，per-step 或 per-batch）**

| 算子 | 实现 | 延迟/吞吐 | 备注 |
|---|---|---|---|
| 15×15 gemm | cublasDgemmBatched | | |
| 3×3 求逆 | cusolver potrfBatched / getrfBatched / 解析逆 | | 对称正定用 potrf |

**表 4 batch 扩展**

| 方法 | B=78 | B=128 | B=256 | B=312 | 备注 |
|---|---|---|---|---|---|
| torch-kf | | | | | |
| batched eager GPU | | | | | |
| 本文 v5/v7 | | | | | |

## 5. 硬件矩阵：H20 vs A100

- **H20**：78 SM（与论文现有数字一致，确认型号即可）；
- **A100**：108 SM（H20 的 ~1.38× SM 数，用于验证批扩展模型 `B_sat ≈ S·C_reg` 的可迁移性）。

每机必须报告（对应 main.tex line 398 的 TODO）：
GPU 具体型号、显存、驱动版本、CUDA 版本、PyTorch / Triton 版本、CPU 型号、OS、编译旗标、测量重复次数、冷/热启动策略、是否排除编译与 autotuning（论文已声明排除，保持一致）。

## 6. 数值正确性协议

1. **统一金标准**：CPU eager fp64 的 `rmse_{x,y,z}` 与 `rho_pct`（`benchmark.py` 的 `GOLDEN`）。
2. **partial 与 full 不混比**：任何 `n<160000` 的 run 必须标 `PART`，不进最终加速比表。
3. **语义区分**：prefix-sum 属"线性化近似"，RMSE 偏离预期内，须在表脚注声明，不得与金标准 golden_match 强绑定。
4. **跨机一致性**：同一 baseline 在 H20/A100 各跑 ≥3 次，报告 mean ± std（论文现标注"final manuscript will report means over repeated runs"）。

## 7. prefix-sum 的已知退化风险（预判）

1. 组合矩阵 2×2 块维度膨胀（15→30），单算子乘法成本上升；
2. fp32 下长期程组合可能**失去协方差正定性/对称性**（与本文 `sec:motivation` 精度分析同源，可互相印证）；
3. work 为 O(N log N)，latency-bound 场景收益存疑；
4. 需冻结线性化点，无法严格复现非线性 SINS + 闭环反馈 + 量测选择。

以上四点均**支持**而非削弱本文论点：它们解释了为何本文选择"融合 + 多轨迹并行"而非"并行-in-time"。

## 8. 下一步 TODO

- [ ] 读 Särkkä 2025 全文，确认 2×2 组合算子细节与可复现接口
- [ ] 确认 H20 / A100 环境可用性，记录各机完整环境清单
- [ ] 跑 3.1 torch eager 全量
- [ ] 实现 3.2 prefix-sum（优先 JAX 路径）
- [ ] 按第 4 节模板出四张表，替换 main.tex `sec:evaluation` 的占位段
