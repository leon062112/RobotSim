"""
规模轴之「轨迹长度 N」扫描 — 评估重构第一步（x 轴 = 输入规模，而非版本）。

方法线（单一 Trident 及其对照）：
  - torch eager GPU（naive 张量）
  - torch.compile（auto-fusion）
  - v3 fused fp64（Trident 融合轴）
  - v4 fused fp32（Trident 融合+精度）

自变量 = 轨迹长度 N。eager/compile 是逐步 launch 的，吞吐 ~300-1300 步/s，与 N 无关；
fused 一次 launch 跑完整个 scan，吞吐高且随 N 摊薄 launch 后趋于平台。

用法：python code/optimization/benchmark_scaling_N.py
"""
import os
import sys
import json
import time
import argparse
import numpy as np
import torch

_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)
os.chdir(os.path.abspath(os.path.join(_here, '..', '..')))

from ekf_baseline import run_ekf_baseline
from ekf_v1 import run_ekf_v1
from ekf_v3 import run_ekf_v3
from ekf_v4 import run_ekf_v4

N_FAST = [500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 166667]
N_SLOW = [500, 1000, 2000, 5000]   # eager/compile 逐步 launch，大 N 太慢


def throughput(fn, **kw):
    m = fn(**kw)
    if isinstance(m, tuple):
        m = m[-1]
    return m['throughput_steps_per_s']


def sweep():
    results = {'N': N_FAST, 'methods': {}}

    # 快方法：全 N 范围，先 warmup 再测
    for label, fn in [('triton_fp64', run_ekf_v3), ('triton_fp32', run_ekf_v4)]:
        kw = dict(n_steps=166667)  # warmup compile
        fn(**kw)  # JIT warmup
        torch.cuda.synchronize()
        pts = []
        for N in N_FAST:
            if label.endswith('fp32'):
                m = fn(n_steps=N, precision='fp32', verbose=False)[-1]
            else:
                m = fn(n_steps=N, verbose=False)[-1]
            pts.append({'n': N, 'throughput': m['throughput_steps_per_s']})
            print(f"  {label:14s} N={N:7d}: {m['throughput_steps_per_s']:11.0f} steps/s")
        results['methods'][label] = pts

    # 慢方法：小 N
    for label, fn, kw in [('eager_gpu', run_ekf_baseline, dict(device='cuda')),
                          ('torch_compile', run_ekf_v1, dict(device='cuda', compile_mode='default'))]:
        pts = []
        for N in N_SLOW:
            m = fn(n_steps=N, **kw)
            if isinstance(m, tuple):
                m = m[-1]
            pts.append({'n': N, 'throughput': m['throughput_steps_per_s']})
            print(f"  {label:14s} N={N:7d}: {m['throughput_steps_per_s']:11.0f} steps/s")
        results['methods'][label] = pts

    out = 'data/results/a100_scaling_N.json'
    json.dump(results, open(out, 'w'), indent=2, ensure_ascii=False)
    print(f'\nsaved -> {out}')


if __name__ == '__main__':
    sweep()