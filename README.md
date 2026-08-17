# RobotSim

面向井下牵引机器人多传感融合定位仿真的 GPU 加速研究项目。

## 目录结构

```text
RobotSim/
├── paper/                 # LaTeX 论文、模板、插图和编译结果
│   └── latex/
├── code/                  # 定位仿真、GPU 优化和可视化代码
│   ├── positioning/
│   ├── optimization/
│   └── visualization/
├── data/                  # 输入轨迹、实验结果、profiling 和结果图
│   ├── trajectory/
│   ├── results/
│   ├── profiles/
│   └── figures/
└── docs/                  # 项目申报书、论文规划、研究报告和参考资料
    ├── project/
    ├── paper-planning/
    ├── reports/
    ├── references/
    ├── development/
    ├── assets/
    └── archive/
```

## 常用入口

- 论文源码：`paper/latex/main.tex`
- 编译论文：`cd paper/latex && latexmk -pdf main.tex`
- 轨迹数据：`data/trajectory/PipeRobot_Trajectory.csv`
- 基础定位实现：`code/positioning/`
- Triton/GPU 优化实现：`code/optimization/`
- 实验结果：`data/results/`

代码中的默认数据和输出路径均相对于仓库根目录。运行 Python 脚本时，请先进入仓库根目录，例如：

```bash
python code/optimization/benchmark.py
```
