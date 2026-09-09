"""
消融图: 贡献组合 x 轨迹长度 N (端到端 SINS/EKF 工作负载, A100)。

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
# The two bottom curves overlap on a log axis (eager fp64 ~449 vs eager fp32
# ~492 steps/s is a ~10% band), so they are given strongly separated visual
# channels: a muted dashed line with downward triangles vs. a solid blue line
# with filled circles.
SERIES = [
    ('none',             'baseline (eager fp64)',           MUTED,  'v', (0, (4, 3)), 1.2, 2),
    ('precision_only',   '+ precision (eager fp32)',        BLUE,   'o', '-',  1.5, 4),
    ('batch_only',       '+ batch (eager fp64, $B{=}108$)', YELLOW, 'D', '-',  1.6, 3),
    ('fusion_only',      '+ fusion (fused fp64)',           AQUA,   '^', '-',  1.8, 3),
    ('fusion_precision', '+ fusion + precision (fused fp32)', GREEN, 's', '--', 1.6, 3),
    ('all',              'Trident (all three, $B{=}108$)', GREEN, 'o', '-', 2.6, 4),
]


def main():
    d = json.load(open(REPO_ROOT / 'data/results/a100_ablation_shape.json'))
    combos = d['combos']
    B = d['batch']

    cpu = json.load(open(REPO_ROOT / 'data/results/a100_benchmark_summary.json'))
    cpu_fp64 = cpu['v0_cpu']['throughput_steps_per_s']

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
        mfc = 'white' if key == 'fusion_precision' else color
        ax.plot(x, y, linestyle=ls, marker=mkr, color=color, linewidth=lw,
                markersize=6, markeredgecolor=color, markeredgewidth=1.0,
                markerfacecolor=mfc, label=label, zorder=z)

    ax.axhline(cpu_fp64, color=MUTED, linestyle=':', linewidth=1.4, zorder=1)
    ax.text(2200, cpu_fp64 * 1.3, f'host CPU fp64 ({cpu_fp64:,.0f} steps/s)',
            fontsize=8.5, color=MUTED)

    # The two launch-bound curves sit within ~10% of each other and overlap;
    # call them out so the reader does not have to separate them by eye.
    none_end = combos['none'][-1]['throughput']
    prec_end = combos['precision_only'][-1]['throughput']
    ax.annotate(f'{none_end:,.0f}', xy=(max_n, none_end), xytext=(-30, -13),
                textcoords='offset points', fontsize=8, color=MUTED)
    ax.annotate(f'{prec_end:,.0f}', xy=(max_n, prec_end), xytext=(-16, 8),
                textcoords='offset points', fontsize=8, color=BLUE)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xticks([2000, 5000, 20000])
    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda v, _: f'{int(v):,}'))
    ax.set_yticks([1e2, 1e3, 1e4, 1e5, 1e6, 1e7])
    ax.get_yaxis().set_major_formatter(
        plt.FuncFormatter(lambda v, _: f'$10^{{{int(np.log10(v))}}}$'))
    ax.set_xlabel('Trajectory length $N$ (filter steps, log)', fontsize=10.5, color=INK)
    ax.set_ylabel('Throughput (filter steps/s, log)', fontsize=10.5, color=INK)
    ax.grid(axis='y', which='major', linestyle='--', linewidth=0.7, color='#e1e0d9', zorder=0)
    ax.tick_params(colors=MUTED, labelsize=8.5)
    for sp in ax.spines.values():
        sp.set_color('#c3c2b7')

    leg = ax.legend(fontsize=8.5, loc='upper left', framealpha=0.95,
                    edgecolor='#c3c2b7', labelcolor=INK)
    leg.set_zorder(5)

    fig.tight_layout()
    for ext in ('png', 'pdf'):
        out = FIGURE_DIR / f'ablation_shape.{ext}'
        fig.savefig(out)
        print('saved ->', out)
    dst = PAPER_FIG_DIR / '5-eval-ablation-a100.pdf'
    shutil.copy(FIGURE_DIR / 'ablation_shape.pdf', dst)
    print('copied ->', dst)


if __name__ == '__main__':
    main()
