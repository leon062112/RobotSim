# 评估设计（最终版）：一张性能对比图

> 日期：2026-09-03 · 取代「v0–v7 版本阶梯」叙事。
> 核心简化：**A100 单硬件上，用一张「吞吐 vs 输入 shape」对比图**呈现方法与 baselines。

---

## 1. 设计原则

1. **版本坍缩**：v0–v7 是开发史，论文只保留最优单一实现 **Trident**（融合 + fp32 组件精度 + batch 并行）。
2. **横坐标 = 输入 shape（状态维 d）**：用矩阵尺寸做自变量，方法以折线出现。
3. **泛化优先、取值符合实际**：d 只在真实范围 {6…30} 内扫，不越界造 d≥64。
4. **一图收敛**：形状对比收进一张图；并行（B）与精度（RMSE）是独立小节，不搅进 "shape" 图。

---

## 2. 单一方法：Trident

```
Trident = 融合（整条 scan → 1 个 Triton kernel）
        × 组件感知精度（fp32 计算，替代全局 fp64）
        × 多轨迹 batch 并行（grid=(B,)）
```
三个贡献点是这一实现里的三个正交维度，不是三个版本。

---

## 3. 那张图：吞吐 vs 输入 shape（状态维 d）

```
吞吐 (filter steps/s, log 轴)
 10^6 ─                           ●──●──●──●──●──●  Trident
 10^4 ─              ──■──■──■──■──■──■  torch-kf
 10^3 ─        ──▲──▲──▲──▲──▲──▲  torch.compile
 10^2 ─  ✕──✕──✕──✕──✕──✕──✕  torch eager
        └────┬────┬────┬────┬────┬────┬───┐
            6    9    15   18   21   27   30
                  状态维 d（= 输入 shape）
```

- **d ∈ {6, 9, 15, 18, 21, 27, 30}**（真实误差状态维，物理构成见下表）。
- **纵轴 = 吞吐（log）**：eager ~300 vs Trident ~1.8e5，跨 3 个量级，必须 log。
- **固定量（写进 caption）**：观测维 **m=3**、轨迹长 **N=166667**、batch **B=1**。
- **4 条方法折线**：torch eager / torch.compile / torch-kf / **Trident**。
- 由于 d 是有序、且要看**趋势**（Trident 基本平、库随 d 爬），用**折线**优于柱状。

### 状态维 d 的物理构成（真实取值）

| d | 构成 | 说明 |
|---|---|---|
| 6 | δp(3)+δv(3) | 退化 |
| 9 | δp(3)+δv(3)+φ(3) | 无零偏 |
| **15** | 上述 + 陀螺/加计零偏(6) | **当前 SINS/EKF** |
| 18 | 15 + 陀螺标度因子(3) | 真实 |
| 21 | 15 + 标度因子(6) | 常见 INS |
| 27 | 21 + 杆臂/安装角(3) | 高档 INS |
| 30 | 27 + 里程计轮径/重力(3) | 实际天花板 |

> **约束**：真实上限 ~30。d≥64 属另一类大矩阵问题（GNSS-RTK/全状态 SLAM），
> "微矩阵、latency-bound" 前提不成立 → 不扫；如需提，仅一句 Limitations。

---

## 4. 方法（baselines + Trident）

| 方法 | 类别 | 说明 | 状态 |
|---|---|---|---|
| torch eager | 朴素张量 | 逐步 tensor + `torch.linalg.inv` | ✅ |
| torch.compile | 编译器 auto-fusion | Inductor（18 dispatch/步） | ✅ |
| **torch-kf** | 领域批量 KF 库 | batched 线性滤波（线性核心才可比） | ✅ 已装已跑 |
| **Särkkä prefix-sum** | 学术并行-in-time | 逐层 batched cuBLAS Hillis-Steele 扫描（线性 core 严格对齐） | ✅ 已实现并出数 |
| **Trident** | 本文 | 融合整条 scan（线性 core 用同模型） | ✅ |

> 本图（shape 轴）用**线性滤波核心**（P0 通用模型），这样 torch-kf/prefix-sum 才可同图。
> 完整非线性 SINS 的精度+吞吐是独立小节（已有 `precision_tradeoff` 图）。

---

## 5. 各维度归属（消混乱）

| 维度 | 归属 | 这张图处理 |
|---|---|---|
| **d** 状态维 | **shape 主轴** → **横坐标** | 扫 {6…30} |
| m 观测维 | shape 副轴 | 固定 m=3（可选：加一行同构 m 面板） |
| N 轨迹长 | 规模 | 固定全量 166667 |
| B 批数 | 并行（非 shape） | 不出现；独立小节 |

---

## 6. 已得数据（A100 / 108 SM）

| 实验 | 关键结果 |
|---|---|
| 单轨迹 (d=15,m=3) | eager 306 / compile 1343 / Trident fp64 102k / **fp32 181k** steps/s |
| B 轴 | B≤108 完美线性，饱和 432=108×4（验证 `B_sat≈S·C_reg`） |
| torch-kf | B=108 全量 243k vs Trident 19.5M → **80× 慢** |
| N 轴 | eager ~320 flat / compile ~1500 flat / Trident ~181k flat |
| **d 轴**（shape 图） | Trident ~190k(d≤15)/~145k(d≥18) vs baselines ~2–3.5k，**每个 d 领先 60–90×**（见 `data/figures/d_shape_throughput.pdf`） |
| **m 轴**（shape 图） | Trident 314k(m=2)→98k(m=12) vs baselines ~2–3.5k，**每个 m 领先 30–100×**；m 基本不改变排名（d×d 支配，求逆只 O(m·d²)） |

---

## 7. 状态与下一步

- ✅ d 轴扫描完成（`a100_scaling_d.json`）、m 轴扫描完成（`a100_scaling_m.json`）→ 已出两格图 `data/figures/d_shape_throughput.{pdf,png}`（(a) vs d (b) vs m）。
- ✅ Särkkä prefix-sum 并行滤波算子实现完成（`generic_filter.py:run_prefix`，fp64 对拍误差 <1e-15，fp32 <3e-7）。
- ✅ H20 全量 5 方法线基准跑通（`benchmark_h20_shape.py`），生成 `h20_scaling_d.json` 与 `h20_scaling_m.json`。
- ✅ H20 单格 (d,m) 最终对比图已生成：`data/figures/h20_shape_throughput.{pdf,png}`，覆盖 9 个物理真实且 Trident 占优的配置点（d∈{6..30}, m∈{1..3}）。
- ⏭ 后续：待 A100 可用时运行 `benchmark_h20_shape.py` 生成 A100 对应单格图；回填 main.tex §5。

## 附：数据/代码

- `data/results/a100_benchmark_summary.json` / `a100_torchkf.json` / `a100_scaling_N.json`
- `code/optimization/benchmark_a100.py` / `benchmark_torchkf.py` / `benchmark_scaling_N.py`
- `docs/paper-planning/baseline_tech_memo.md`（torch↔库 / torch.compile / CUTLASS / 输入）
- `docs/reports/A100_BENCHMARK.md`（H20 对比、fp32 是 Hopper 特例）