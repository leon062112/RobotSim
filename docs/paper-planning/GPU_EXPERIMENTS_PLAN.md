# GPU 补实验计划（TRIDENT §5 实验闭环）

> 日期：2026-09-08 · 状态：等待 GPU 环境执行
> 本文档是**唯一执行入口**：列出所有待补实验、精确到命令的 runbook、产出文件、验收标准、
> 以及实验跑完后论文要回填的位置。无 GPU 环境下不要执行本文档任何命令。

---

## 0. 背景：为什么需要补实验

论文 §5 当前结构：5.1 Setup（Table 5 平台表）/ 5.2 End-to-End（占位 TODO）/
**5.3 Accuracy and Precision（Table 7 已写）** / 5.4 Batch（占位）/ 5.5 Ablation（占位）。

Table 7 的 eager / torch.compile 两行**缺失**：归档数据里这两个 baseline 只有
`n_steps=20001` 的 partial run（RMSE X 994.717…，与全量 166,667 步的 golden 1017.838
不是同一评估窗口），**不能与全量行同表**。正文已如实标注
"full-length timing and accuracy runs for these two baselines are pending"。
补齐后即可删除该 pending 声明，并顺带闭环 5.2 / 5.4 / 5.5 的数据支撑。

数据事实源（source of truth）：`data/results/*.json`。本次所有新结果一律写入新文件，
**不覆盖旧文件**；同一指标若与旧轮冲突，以本文档新跑一轮为准（正文/表注写明轮次）。

---

## 1. 环境准备与校验（约 20 分钟）

### 1.1 依赖

```bash
pip install torch triton numpy matplotlib torch_kf   # torch_kf 为批量 KF 库 baseline
```

### 1.2 硬件/软件记录（每台机器开头先跑，结果进 JSON）

```bash
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python -c "import torch,triton;print(torch.__version__, torch.version.cuda, triton.__version__)"
```

### 1.3 Golden 校验（必做，失败即停）

```bash
python - <<'EOF'
import sys, os, json
sys.path.insert(0, 'code/optimization')
from ekf_baseline import run_ekf_baseline

# (a) 短窗数值栈自检：与归档 partial 值逐位比对（同一确定性数值栈，设备无关）
arch = [r for r in json.load(open('data/results/benchmark_summary.json'))['results']
        if r['label'] == 'v0 eager GPU'][0]
m = run_ekf_baseline('data/trajectory/PipeRobot_Trajectory.csv', device='cpu',
                     n_steps=20001, verbose=False)
for k in ('rmse_x_mm', 'rmse_y_mm', 'rmse_z_mm', 'rho_pct'):
    assert abs(m[k] - arch[k]) < 1e-9, (k, m[k], arch[k])
print('short-window stack OK')

# (b) 全量 golden 逐位校验（约 1.5 min CPU）
GOLDEN = dict(rmse_x_mm=1017.838305, rmse_y_mm=16.834549,
              rmse_z_mm=8.140964, rho_pct=0.175540)
m = run_ekf_baseline('data/trajectory/PipeRobot_Trajectory.csv', device='cpu', verbose=False)
for k, v in GOLDEN.items():
    assert abs(m[k] - v) < 1e-6, (k, m[k], v)
print('golden OK')
EOF
```

(a) 或 (b) 失败 → 检查 torch 版本 / CPU 线程数（历史差异来源），不要继续。

### 1.4 论文声明环境核对

Table 5（`tab:experimental-platform`，§5.1）声明 PyTorch 2.11.0 / CUDA 12.9 / Triton 3.6.0。
若 GPU 机实际版本不同：要么装到声明版本，要么**改 Table 5 使其与实际一致**——二选一，不允许表与机器不符。

---

## 2. 任务总表

| ID | 优先级 | 内容 | 预计耗时/机 | 产出 | 回填位置 |
|---|---|---|---|---|---|
| **G1** | P0 | eager + torch.compile 全量 fp64：timing + RMSE | ~35 min | `full_tensor_baselines_{tag}.json` | Table 7 两行；删 pending 声明 |
| **G2** | P1 | 线性核 5 方法全量精度对拍（d,m ∈ {(15,3),(30,3),(6,3),(15,1)}） | ~30 min（eager-fp32 全量是主要耗时） | `linear_accuracy_{tag}.json` | §5.3 已有结论升级为实测表 |
| **G3** | P1 | PrefixScan 逐 d fp32 精度 | 并入 G2 | 并入 G2 文件 | 同上 |
| **G4** | P2 | 端到端阶梯复跑（v0/v2/v3/v4/v5/v7，3 reps） | ~20 min | `{tag}_benchmark_summary.json` | §5.2 正文+表 |
| **G5** | P2 | batch 三区扫描 + spread 校验 + tf32 两机 | ~15 min | `batch_{tag}.json` | §5.4 正文 |
| **G6** | P2 | 全程漂移曲线（fp32/TF32/bf16-io） | ~5 min | `trajectory/{tag}_drift_*.npy` | §5.3 可选图 |
| **G7** | P3 | 负结果 ablation（streams/TF32/TMA/warp-spec/autotune） | ~30 min | `ablation_negative_{tag}.json` | §5.5 正文 |

`{tag}` = `a100` / `h20`，按 `torch.cuda.get_device_name(0)` 自动命名（沿用现有约定）。

---

## 3. 详细 Runbook

> 所有脚本放 `code/optimization/`，从仓库根目录执行；沿用现有写法
> （`os.chdir` 到仓库根、warmup 后计时、`torch.cuda.synchronize()`）。

### G1（P0）eager / torch.compile 全量 fp64

新建 `code/optimization/full_tensor_baselines.py`，复用现有 runner（勿重写滤波逻辑）：

```python
import json, time, os, sys
import numpy as np, torch
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here); os.chdir(os.path.join(_here, '..', '..'))
from ekf_baseline import run_ekf_baseline          # eager
from ekf_v1 import run_ekf_v1                      # eager(v1) / compile via compile_mode
from benchmark_a100 import bench_repeat, get_hardware, check_golden  # 计时+元数据+校验

CSV = 'data/trajectory/PipeRobot_Trajectory.csv'    # 全量 166,667 步
tag = 'a100' if 'A100' in torch.cuda.get_device_name(0) else 'h20'

out = {'hardware': get_hardware()}
# eager GPU fp64 全量
m = bench_repeat(lambda: run_ekf_baseline(CSV, device='cuda'), n=3)
out['v0_gpu_eager_full'] = m
# torch.compile 全量（compile_mode='default'）
m = bench_repeat(lambda: run_ekf_v1(CSV, device='cuda', compile_mode='default'), n=3)
out['v1_gpu_compile_full'] = m
# 精度：对 golden 逐轴 RMSE 已由 runner 返回（rmse_x_mm 等），check_golden 校验
json.dump(out, open(f"data/results/full_tensor_baselines_{tag}.json", 'w'), indent=2)
```

要点：
- **fp64 全程**（默认 dtype 即 fp64，勿转 fp32）；
- runner 已返回逐轴 RMSE 与 `golden_match`，直接入表；
- 耗时外推（按归档 partial 速率）：eager ≈300 steps/s → 全量单次 ≈9–10 min；
  compile ≈1.4–2.2k steps/s → 全量单次 ≈1.3–2 min；每配置 3 reps 合计 ≈35 min/机。
  **不要**用 partial window 代替，也不要把 reps 省到 3 以下。

**验收**：`golden_match == True`（容差 0.2 mm，同 `benchmark_a100.py:check_golden`）；
throughput mean±std 的 std/mean < 5%。预期量级：eager ~300 steps/s、compile ~1.3–2.2k steps/s
（H20/A100 partial 轮的外推值，仅作 sanity，不作为验收线）。

### G2+G3（P1）线性核精度对拍

扩展 `code/optimization/benchmark_h20_shape.py`，加 `--accuracy` 模式（性能路径不动）：

- 配置点：`(d,m) ∈ {(6,3), (15,3), (30,3), (15,1)}`，`N=166667`（全量，替换该脚本默认 N=2000，
  命令行参数 `--N`）；
- 参考：`run_eager` 加 `dtype=torch.float64` 参数，CPU 跑一次存 `x_ref, P_ref`（分钟级）；
- 每个 GPU 方法（eager-fp32 / compile-fp32 / torch-kf / prefix / triton）输出终态
  `mean_err = max|x - x_ref|`、`cov_err = max|P - P_ref|`（prefix/torch-kf 返回 mean,cov 的取对应元；
  现 `run_triton` 只存了 trace/x0/x1 三个标量 → 给 `lin_scan_kernel` 增加可选输出 buffer
  存完整 (x,P)，仅在 `--accuracy` 时启用，性能路径保持原样）；
- 输出 `data/results/linear_accuracy_{tag}.json`：每 (d,m) × 每方法 × (mean_err, cov_err,
  顺带 steps/s)。

**验收**：torch-kf / prefix / triton 的 fp32 结果 vs fp64 参考 ≤ **1e-5**（历史对拍水平：
prefix fp64<1e-15、fp32<3e-7；留 2 个量级余量）；eager/compile fp32 同量级。
若某方法超线 → 查其求逆/累加路径，记为 finding 而非静默放行。

### G4（P2）端到端阶梯复跑

```bash
python code/optimization/benchmark_a100.py --skip-cpu-full
```

- 产出 `a100_benchmark_summary.json` 同构文件（脚本已按 GPU 名自适应；注意该脚本目前写死
  输出名 `a100_benchmark_summary.json` —— 在 H20 上跑前先改成 `{tag}_benchmark_summary.json`，
  **防止覆盖** A100 归档）；
- CPU fp64 行用 `--skip-cpu-full` 跳过（已有），或机器空闲时去掉该 flag 补 CPU 行（~3 min）；
- v5/v7 的 batch 部分只跑到 SM×2 即可（G5 做完整扫描）。

**验收**：与归档轮同配置偏差 <10%（轮间噪声历史约 ±3%）；`golden_match` 全部 True。
回填 §5.2：正文按"单轨迹延迟阶梯 + batch 聚合吞吐"两段写，表保留 v0–v7 行但用新数，
并删除 "will reconcile…archived JSON" 的 TODO 句。

### G5（P2）batch 三区 + 数值无耦合

扩展 `code/optimization/ekf_v7_concurrency.py` 或用 `benchmark_a100.py` 已有 batch 逻辑：

- replicate 列：`B ∈ 1,2,…,S,2S,3S,4S,4S+64`（S=SM 数；A100→108/432，H20→78/312），3 reps，
  记录 steps/s、traj/s、`max_traj_spread_m`；
- independent 列（不同 seed，`data/trajectory/traj_batch_fp64.npy`）：`B ∈ 1,16,32,64,S`，
  记录逐轨迹 RMSE 的 mean±std；
- tf32 单轨迹行（H20 与 A100 各一）：v4 precision='tf32'，3 reps + RMSE。

**验收**：线性区 efficiency ≥0.98；B_sat 处 efficiency 拐点与 `S×C_reg`（C_reg=4）预测相符 ±1 档；
replicate 的 spread 恒为 0.0；tf32 偏差方向与归档一致（H20 ~4.7mm / A100 ~0.6mm 级）。
回填 §5.4：三区模型叙述 + B_sat 命中 + spread/independent 两句。

### G6（P2，可选图）全程漂移曲线

新增 `code/optimization/trajectory_drift.py`：以 v4 的 fp64 输出为参考轨迹，分别跑
`fp32 / tf32 / bf16-io`，每 1000 步记录 `max|Δpos| (mm)`，存
`data/trajectory/{tag}_drift_{plan}.npy`；绘图追加到 `code/visualization/plot_precision_tradeoff.py`
（新增第三 panel：drift vs step）。用途：把"bf16 的 Z 轴在全程才爆炸"从表格论断变成曲线
（预期 bf16-io 曲线在 10^4 步后陡升，fp32 平直）。

### G7（P3）负结果 ablation

每项记录 方案 / 吞吐 / 与 fp32 基线比 / 结论一句，写入 `data/results/ablation_negative_{tag}.json`：

1. **CUDA streams**：B=S 下 1/2/4/8 流（已有 `v7_concurrency.json:stream_overlap` 旧值，复跑确认）；
2. **TF32**：并入 G5 输出，本节只写结论句；
3. **TMA**：Triton `tl.make_tensor_descriptor` 加载传感器流 vs 现有 masked load，单轨迹 3 reps；
4. **warp specialization**：block 128→256/512 线程、每轴状态拆分到 warp 的两种尝试，单轨迹 3 reps；
5. **autotune**：`@triton.autotune` 扫 num_warps∈{2,4,8}、num_stages∈{1,2,3}，报告选中配置与
   手工配置的吞吐差（预期 ≈0，支撑"微矩阵递推不吃常规 autotune"）。

**验收**：无硬性数值线；要求每项都有"为什么不work"的一句机理（进正文）。

---

## 4. 论文回填清单（实验跑完后）

| 位置 | 动作 |
|---|---|
| Table 7（`tab:precision-fidelity`） | 填 eager / compile 两行的 Steps/s 与 RMSE X（Max Δ 取 max\|RMSE−ref\| 逐轴）；删除 caption 与正文的 pending 句 |
| Table 5 | 若 1.4 核对发现版本不符，更新声明版本 |
| §5.2 | 用 G4 数据写成文（两段 + 更新 tab:end-to-end 数值，删 TODO 句）；headline（摘要/引言）若与最终轮不一致需同步 |
| §5.3 | G2/G3 结果：把"reproduce to within 1e-14 / 3e-7"一句升级为带配置点的实测表述或小表 |
| §5.4 | G5：三区叙述、B_sat 命中、spread=0、independent mean±std、tf32 两机 |
| §5.5 | G7 五段负结果；G6 若出图则加 `\Cref{fig:drift}` |
| 摘要/结论 | 若 G4 复跑的 headline 数字（8.8×/130×/4.63e7）变化 >10%，同步更新 abstract/conclusion |

回填顺序建议：G1 → Table 7（P0 闭环）→ G4 → §5.2 → G5 → §5.4 → G2/G3 → §5.3 → G7 → §5.5。

---

## 5. 风险与陷阱清单

1. **partial vs full 混用**：任何 RMSE 与 1017.838 不同窗口的数（994.717…）**禁止**入全量表。
2. **覆盖归档**：H20 上跑 `benchmark_a100.py` 前必须改输出文件名（见 G4）。
3. **只看 X 轴**：精度判定三件套 = 逐轴 RMSE + max Δ vs ref + ρ（bf16-io 的 X 看似正常、祸在 Z）。
4. **tf32 偏差硬件相关**：H20 ≈4.7mm、A100 ≈0.6mm 量级，两机都报、不平均、不跨机引用。
5. **吞吐轮间噪声 ±3%**：一表一来源；同表内不得混轮次。
6. **fp64 路径别开 tf32**：跑 fp64/IEEE 配置时确认 `torch.backends.cuda.matmul.allow_tf32=False`
   且 Triton `IP='ieee'`。
7. **环境版本漂移**：torch/triton 版本影响寄存器分配 → 影响 B_sat（110 regs/thread→4 blocks/SM
   是归档环境的值）；版本变了则 B_sat 预测按新 profiling 重算，不沿用旧值。

---

## 6. GPU 会话快速执行序（复制即用）

```bash
# 0) 环境校验 + golden
bash docs/paper-planning/gpu_smoke.sh            # （可选：把 §1 命令固化成脚本）
# 1) P0
python code/optimization/full_tensor_baselines.py
# 2) P1
python code/optimization/benchmark_h20_shape.py --accuracy --N 166667
# 3) P2
python code/optimization/benchmark_a100.py --skip-cpu-full
python code/optimization/ekf_v7_concurrency.py --full-sweep
python code/optimization/trajectory_drift.py
# 4) P3
python code/optimization/ablation_negative.py
# 5) 同步结果回仓库后，按 §4 回填论文并重编译
cd paper/latex && latexmk -pdf main.tex
```

（标 `--accuracy` / `--full-sweep` 的 flag 为本次计划新增，脚本未改前会报 unknown argument——
先按 §3 落地脚本改动再执行。）
