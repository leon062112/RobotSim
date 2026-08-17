### EKF 优化与性能分析任务

#### 1. Torch Compile 融合效果验证

* 基于当前EKF实现，使用 `torch.compile` 进行编译优化。
* 通过 Profiling 工具（PyTorch Profiler、Nsight Systems、Nsight Compute 等）分析编译后生成的执行图：
  * 检查算子融合（Fusion）情况；
  * 统计 Kernel 数量变化；
  * 观察是否存在频繁的小 Kernel 启动开销；
  * 分析瓶颈算子及其耗时占比。

#### 2. Mega Kernel 可行性探索

* 评估 EKF 计算流程是否能够进一步重构为类似 Mega Kernel 的执行模式：

  * 减少中间 Tensor 读写；
  * 减少 Kernel Launch 开销；
  * 提高数据局部性；
  * 提升 GPU 利用率。

#### 3. 优化过程管理

* 每完成一个优化步骤：
  * 保存对应代码版本；
  * 保存 Profiling 数据；
  * 记录关键性能指标（Latency、Throughput、Kernel 数量、显存占用等）。
* 建立优化日志，形成完整的性能演进轨迹，方便后续回溯和分析。

#### 4. Auto-Tuning 与输入特征分析

分析实际输入数据特征，评估 Auto-Tuning 的优化空间：
##### 数据 Pattern 分析
* 稠密（Dense）与稀疏（Sparse）程度；
* 输入矩阵结构特征；
* Batch Size 分布；
* 状态维度和观测维度规模。
##### Shape 特征分析


##### 动态性分析

* Shape 是否动态变化；
* Batch 是否动态变化；
* 不同场景下输入模式分布；
* 动态 Shape 对编译缓存命中率的影响。

##### Auto-Tuning 方向

* Kernel 配置搜索；
* Tile Size 选择；
* Memory Layout 优化；
* Kernel Fusion 策略选择；
* 编译缓存策略分析。

#### 5. Baseline 与对比实验设计

构建系统化 Benchmark，对优化效果进行量化评估。

##### 对比对象

1. 原始 PyTorch 实现（Eager Mode）
2. `torch.compile` 实现
3. 手工优化版本
4. 相关高性能算子库
5. 学术界代表性实现

##### 调研方向

* GPU 上的相关加速工作；
* 高性能线性代数库：
  * 求解器
  * cuBLAS
  * cuSOLVER
  * cuSPARSE（若涉及稀疏场景）
* 近几年相关论文与开源实现。

#### 6. 输出成果

最终形成：
1. EKF 性能瓶颈分析报告；
2. Torch Compile 融合效果分析；
3. Auto-Tuning 可行性分析；
4. Baseline 对比实验结果；
5. 优化路线总结与后续改进建议。


#### 6月22号 新TODO
1. fp32: 定位精度需求 ~mm 级, fp32 足够; (fp64→fp32 + tensor core)
2. 多轨迹batch并行: 当前单 block 串行受限。多条轨迹 / 蒙特卡洛时, 每个CUDA block 跑一条独立 scan, 填满 SM, 吞吐可随轨迹数近线性扩展
3. 持久化kernel + 在线流式: 100Hz 实时场景下, kernel 常驻、数据流式喂入,消除每帧launch
4. 合并小矩阵  profiling SM利用率 对角线堆叠，同时bg里说明：小矩阵特性
5. 尝试降精度/利用Tensor core/TMA/warp specialization
6. 参考FA3triton版本中使用的优化技巧
7. 论文三个贡献点确定(算子/系统(多batch并行)/算法优化?)
8. 相关的学术工作，按照贡献点找相关工作