按照「论文贡献点 → 技术实现 → 实验验证」的逻辑整理。

## 一、确定论文主线与贡献点（最高优先级）

### 1. 明确论文三个贡献点

初步考虑：
#### Contribution 1：算子级优化（Kernel Optimization）
针对 Scan / Kalman Filter / Trajectory Propagation 中的小矩阵计算特点：
* mega kernel
* fp64 → fp32
* Tensor Core 加速
* TMA
* Warp Specialization
* Persistent Kernel
目标：
> 提出面向小矩阵轨迹推理的高性能 CUDA Kernel
---
#### Contribution 2：系统级并行化（System Optimization）
解决单轨迹并行度不足的问题：
* 多轨迹 Batch 并行
* Monte-Carlo Batch 并行
* Persistent Kernel
* Streaming Pipeline
目标：
> 将轨迹级并行映射到 GPU，实现高吞吐实时处理
---
#### Contribution 3：算法级优化（Algorithm Optimization）
探索：
* 降精度对定位误差影响
* Scan 过程重排
* 稀疏化/近似化
* 减少同步与访存
目标：
> 在保持 mm 级精度的前提下进一步提升性能
---

# 二、Kernel优化方向

## 3. 精度分析

目标：
验证 fp32 是否满足定位需求。
任务：
* 分析定位误差需求（mm级）
* fp64 baseline
* fp32结果对比
* 误差统计
预期：
> fp32足够满足精度需求
进一步：
* fp32 + Tensor Core
---

## 4. Tensor Core优化
尝试：
* WMMA
* MMA
* Tensor Core GEMM
验证：
* 小矩阵场景收益
* 精度损失
---
## 5. TMA / Warp Specialization
参考：
* Hopper架构优化方式
* FlashAttention-3 Triton实现
重点：
* 数据搬运隐藏
* Pipeline overlap
---

# 三、系统级优化方向

## 6. 多轨迹 Batch 并行

现状：
* 单 block 扫描一条轨迹
* SM 利用率低
优化：
* 一个 block → 一条轨迹
* 多 block → 多轨迹
场景：
* Monte-Carlo
* 多目标跟踪
* 多传感器轨迹
验证：
* 吞吐量随 Batch 增长情况
* SM Occupancy
目标：
> 吞吐量接近线性扩展

---
## 8. Persistent Kernel + Streaming
现状：
每帧：

```text
CPU
 ↓
Launch Kernel
 ↓
GPU计算
```
问题：
* launch overhead
优化：
```text
Persistent Kernel
     ↓
Ring Buffer
     ↓
持续处理数据流
```
目标场景：
* 100Hz实时定位
* 在线轨迹更新
验证：
* latency
* jitter
* throughput
---

# 四、Profiling与性能分析

## 9. Profiling SM利用率

工具：

* Nsight Compute
* Nsight Systems

关注：

* SM Occupancy
* Warp Efficiency
* Tensor Core Utilization
* Memory Throughput
输出：
形成性能瓶颈分析图
---

## 10. 小矩阵特性分析

论文背景部分需要补充：

为什么 GPU 不擅长当前任务？

特点：

* Matrix Size 很小
* Arithmetic Intensity 低
* Launch Overhead 占比高
* Occupancy 不足

进一步分析：

* 对角线堆叠（Diagonal Packing）
* Batched GEMM
* 数据布局优化

输出：

Background章节素材

---

# 五、预期执行顺序

建议按以下顺序推进：

```text
① 确定论文贡献点
        ↓
② 查Related Work
        ↓
③ fp32精度验证
        ↓
④ 多Batch并行实现
        ↓
⑤ Profiling分析
        ↓
⑥ Tensor Core优化
        ↓
⑦ Persistent Kernel
        ↓
⑧ TMA/Warp Specialization
        ↓
⑨ 整理论文实验结果
```

这样最后论文结构会比较自然：

```text
1. Background
    小矩阵定位计算的GPU挑战

2. Method
    2.1 Mixed Precision Optimization
    2.2 Multi-Trajectory Batch Parallelism
    2.3 Persistent Streaming Execution

3. Implementation
    Tensor Core
    TMA
    Warp Specialization

4. Evaluation
    Accuracy
    Throughput
    Latency
    SM Utilization
```

从你目前列出的事项看，**最值得优先做的是 fp32精度验证 + 多轨迹Batch并行**，因为这两项最有希望直接形成论文的核心贡献和实验结果。
