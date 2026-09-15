"""
消融图: 单轴隔离 + 完整组合, 端到端 SINS/EKF 工作负载, A100。

五条线 (吞吐 vs 轨迹长度 N, 双 log):
  baseline       v1 eager fp64, B=1            —— 三个轴都不加 (FP64 GPU eager baseline)
  fusion         v3 fused fp64, B=1            —— 只加 fusion
  mix-precision  v1 eager fp32, B=1            —— 只加 mix-precision (component-aware fp32 计划)
  batch          v1 eager fp64, B=108          —— 只加 batch trajectory (每 SM 一条)
  Trident        v5 fused fp32, B=108          —— 三个全加

launch-bound 的 eager 变体吞吐与 N 无关 (平坦), 只测小 N; 为保持各线等长,
所有线统一截断到公共 horizon N=20,000。

数据: data/results/a100_ablation_shape.json (benchmark_ablation_shape.py)
输出: data/figures/ablation_shape.{png,pdf} -> paper/latex/figures/5-eval-ablation-a100.pdf
"""
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
FIGURE_DIR = REPO_ROOT / 'data' / 'figures'
PAPER_FIG_DIR = REPO_ROOT / 'paper' / 'latex' / 'figures'

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False

# CVD-safe 分类色 (与 plot_d_shape.py 相同 palette)
BLUE, AQUA, YELLOW, GREEN = '#2a78d6', '#1baf7a', '#eda100', '#008300'
INK, MUTED = '#0b0b0b', '#898781'

# (json key, label, color, marker, linestyle, linewidth, zorder)
# baseline 与 mix-precision 两条线在对数轴上只差 ~10%, 用强烈区分的视觉通道
# (灰虚线+倒三角 vs 蓝实线+圆点) 并辅以数值标注。
SERIES = [
    ('none',             'baseline',         MUTED,  'v', (0, (4, 3)), 1.2, 2),
    ('fusion_only',      'fusion',           AQUA,   '^', '-',  1.8, 3),
    ('precision_only',   'mix-precision',    BLUE,   'o', '-',  1.5, 4),
    ('batch_only',       'batch trajectory', YELLOW, 'D', '-',  1.6, 3),
    ('all',              'Trident',          GREEN,  'o', '-',  2.6, 4),
]


def main():
    d = json.load(open(REPO_ROOT / 'data/results/a100_ablation_shape.json'))
    combos = d['combos']
    missing = [k for k, *_ in SERIES if k not in combos]
    if missing:
        raise SystemExit(f'missing combos in json: {missing} — '
                         f'run benchmark_ablation_shape.py {" ".join(missing)}')

    # Launch-bound variants are per-step invariant in N (they are flat), so
    # they were timed only up to 20,000 steps; truncating the flat fused
    # curves to the same horizon keeps every series the same length and
    # avoids implying a dependence on N that the fused data does not have.
    max_n = max(p['n'] for pts in combos.values() for p in pts
                if all(q['n'] <= 20000 for q in pts))  # -> 20000

    fig, ax = plt.subplots(figsize=(7.6, 4.4), dpi=200)

    for key, label, color, mkr, ls, lw, z in SERIES:
        pts = [p for p in combos[key] if p['n'] <= max_n]
        x = [p['n'] for p in pts]
        y = [p['throughput'] for p in pts]
        ax.plot(x, y, linestyle=ls, marker=mkr, color=color, linewidth=lw,
                markersize=6, markeredgecolor=color, markeredgewidth=1.0,
                markerfacecolor=color, label=label, zorder=z)

    # 底部两条 launch-bound 曲线相差 ~10%, 在对数轴上几乎重叠, 标注数值区分
    end = {k: [p for p in combos[k] if p['n'] <= max_n][-1] for k, *_ in SERIES}
    for key, (dx, dy) in [('none', (-30, -13)), ('precision_only', (-16, 8))]:
        p = end[key]
        ax.annotate(f"{p['throughput']:,.0f}", xy=(p['n'], p['throughput']),
                    xytext=(dx, dy), textcoords='offset points',
                    fontsize=10, color=MUTED)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xticks([2000, 5000, 20000])
    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda v, _: f'{int(v):,}'))
    ax.set_yticks([1e2, 1e3, 1e4, 1e5, 1e6, 1e7])
    ax.get_yaxis().set_major_formatter(
        plt.FuncFormatter(lambda v, _: f'$10^{{{int(np.log10(v))}}}$'))
    ax.set_xlabel('Trajectory length $N$ (filter steps, log)', fontsize=13, color=INK)
    ax.set_ylabel('Throughput (filter steps/s, log)', fontsize=13, color=INK)
    ax.grid(axis='y', which='major', linestyle='--', linewidth=0.7, color='#e1e0d9', zorder=0)
    ax.tick_params(colors=MUTED, labelsize=11)
    for sp in ax.spines.values():
        sp.set_color(INK)

    # 共享图例: 图上方, 三列两行 5 项, 无边框, 字体加大
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc='upper center',
               bbox_to_anchor=(0.5, 0.99), ncol=3, fontsize=12,
               handlelength=1.6, columnspacing=1.8, handletextpad=0.6)

    # 显式边距 (tight_layout 与 fig.legend 顶部图例不兼容): 顶部留图例行
    fig.subplots_adjust(left=0.115, right=0.99, bottom=0.115, top=0.85)
    for ext in ('png', 'pdf'):
        out = FIGURE_DIR / f'ablation_shape.{ext}'
        fig.savefig(out)
        print('saved ->', out)
    dst = PAPER_FIG_DIR / '5-eval-ablation-a100.pdf'
    shutil.copy(FIGURE_DIR / 'ablation_shape.pdf', dst)
    print('copied ->', dst)


if __name__ == '__main__':
    main()
