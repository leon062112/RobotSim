import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import LogLocator, NullFormatter
from matplotlib.transforms import blended_transform_factory


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = REPO_ROOT / "data" / "results"
FIGURE_DIR = REPO_ROOT / "data" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        "axes.linewidth": 0.8,
        "font.size": 10.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

BLUE = "#4C78A8"
BLUE_LIGHT = "#B8C5D3"
RED = "#D94F4F"
AMBER = "#E57C12"
GREEN = "#3A923A"
GREY = "#666666"
GRID = "#D9D9D9"


def load_results():
    with (RESULT_DIR / "v4_precision_full.json").open(encoding="utf-8") as stream:
        full_precision = json.load(stream)
    with (RESULT_DIR / "precision_isolation.json").open(encoding="utf-8") as stream:
        isolation_rows = {row["combo"]: row for row in json.load(stream)}

    fp32 = isolation_rows["baseline"]
    return [
        {
            "label": "FP64",
            "throughput": full_precision["fp64"]["throughput_steps_per_s"],
            "rmse_x": full_precision["fp64"]["rmse_x_mm"],
            "rmse_z": full_precision["fp64"]["rmse_z_mm"],
        },
        {
            "label": "FP32",
            "throughput": fp32["tput"],
            "rmse_x": fp32["rmse_x"],
            "rmse_z": fp32["rmse_z"],
        },
        {
            "label": "FP16\nsensor\nI/O",
            "throughput": isolation_rows["io-fp16"]["tput"],
            "rmse_x": isolation_rows["io-fp16"]["rmse_x"],
            "rmse_z": isolation_rows["io-fp16"]["rmse_z"],
        },
        {
            "label": "TF32\ndot\nkernel",
            "throughput": isolation_rows["dot-tf32"]["tput"],
            "rmse_x": isolation_rows["dot-tf32"]["rmse_x"],
            "rmse_z": isolation_rows["dot-tf32"]["rmse_z"],
        },
        {
            "label": "FP16\nstate\nkernel",
            "throughput": isolation_rows["state-fp16"]["tput"],
            "rmse_x": isolation_rows["state-fp16"]["rmse_x"],
            "rmse_z": isolation_rows["state-fp16"]["rmse_z"],
        },
    ]


def compact_rate(value):
    return f"{value / 1000:.1f}k"


def scientific_label(value):
    exponent = int(np.floor(np.log10(value)))
    mantissa = value / 10**exponent
    return rf"${mantissa:.1f}\times10^{{{exponent}}}$"


results = load_results()
labels = [row["label"] for row in results]
throughput = np.array([row["throughput"] for row in results])
rmse_x = np.array([row["rmse_x"] for row in results])
rmse_z = np.array([row["rmse_z"] for row in results])

# Use FP64 as the numerical reference so the plot reads as the deviation
# introduced when precision is reduced. The zero reference is not plotted on
# the logarithmic deviation axis.
fp64_index = 0
fp32_index = 1
relative_dx = np.abs(rmse_x - rmse_x[fp64_index]) / rmse_x[fp64_index]
relative_dz = np.abs(rmse_z - rmse_z[fp64_index]) / rmse_z[fp64_index]
deviation_indices = np.array([1, 2, 3, 4])

fig, ax_perf = plt.subplots(figsize=(5.2, 5.0), dpi=180)
ax_dev = ax_perf.twinx()
x = np.arange(len(results))

# Throughput bars use one visual channel; precision effects are overlaid as
# unconnected points because the configurations are categorical alternatives.
bar_colors = [BLUE_LIGHT] + [BLUE] * (len(results) - 1)
bars = ax_perf.bar(
    x,
    throughput,
    width=0.60,
    color=bar_colors,
    edgecolor="#333333",
    linewidth=0.55,
    zorder=2,
)
ax_perf.set_yscale("log")
ax_perf.set_ylim(1e4, 4.4e5)
ax_perf.set_ylabel("Throughput (filter steps/s)", color=BLUE)
ax_perf.tick_params(axis="y", colors=BLUE)
ax_perf.spines["left"].set_color(BLUE)
ax_perf.spines["top"].set_visible(False)
ax_perf.set_xticks(x, labels)
ax_perf.tick_params(axis="x", length=0, pad=8)
ax_perf.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
ax_perf.yaxis.set_minor_formatter(NullFormatter())
ax_perf.grid(axis="y", which="major", color=GRID, linewidth=0.7, zorder=0)
ax_perf.axhline(throughput[fp32_index], color=BLUE, linestyle=(0, (4, 2)), linewidth=1.0, alpha=0.6)

for bar, value in zip(bars, throughput):
    ax_perf.text(
        bar.get_x() + bar.get_width() / 2,
        value * 1.08,
        compact_rate(value),
        ha="center",
        va="bottom",
        fontsize=7.8,
        color=BLUE,
        zorder=5,
    )

speedup = throughput[fp32_index] / throughput[0]
ax_perf.annotate(
    "",
    xy=(1, 5.0e4),
    xytext=(0, 5.0e4),
    arrowprops={"arrowstyle": "<->", "color": BLUE, "linewidth": 1.1},
)
ax_perf.text(0.5, 5.55e4, f"{speedup:.1f}x", ha="center", va="bottom", color=BLUE, fontweight="bold")

ax_dev.set_yscale("log")
ax_dev.set_ylim(2e-7, 1.2e3)
ax_dev.set_ylabel("Relative RMSE deviation from FP64", fontsize=9.3)
ax_dev.spines["top"].set_visible(False)
ax_dev.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
ax_dev.yaxis.set_minor_formatter(NullFormatter())
ax_dev.set_zorder(ax_perf.get_zorder() + 1)
ax_dev.patch.set_visible(False)

offset = 0.09
dx_x = deviation_indices - offset
dz_x = deviation_indices + offset
dx_values = relative_dx[deviation_indices]
dz_values = relative_dz[deviation_indices]
ax_dev.scatter(dx_x, dx_values, s=58, marker="o", color=RED, edgecolor="white", linewidth=0.7, zorder=6)
ax_dev.scatter(dz_x, dz_values, s=56, marker="s", color=AMBER, edgecolor="white", linewidth=0.7, zorder=6)

threshold = 1e-3
ax_dev.axhspan(ax_dev.get_ylim()[0], threshold, color=GREEN, alpha=0.05, zorder=0)
ax_dev.axhline(threshold, color=GREEN, linestyle=(0, (1.5, 2)), linewidth=1.2, zorder=3)
threshold_trans = blended_transform_factory(ax_dev.transAxes, ax_dev.transData)
ax_dev.text(
    0.99,
    threshold * 1.18,
    r"fp16 epsilon ($10^{-3}$)",
    transform=threshold_trans,
    ha="right",
    va="bottom",
    fontsize=7.6,
    color=GREEN,
    bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.0, "alpha": 0.88},
)

dx_offsets = [(-5, 7), (-5, 7), (-5, 7), (-5, 7)]
dz_offsets = [(4, 7), (5, 7), (5, 7), (5, 7)]
for point_x, value, text_offset in zip(dx_x, dx_values, dx_offsets):
    ax_dev.annotate(
        scientific_label(value),
        (point_x, value),
        xytext=text_offset,
        textcoords="offset points",
        ha="right",
        va="bottom" if text_offset[1] > 0 else "top",
        fontsize=7.2,
        color=RED,
    )
for point_x, value, text_offset in zip(dz_x, dz_values, dz_offsets):
    ax_dev.annotate(
        scientific_label(value),
        (point_x, value),
        xytext=text_offset,
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=7.2,
        color=AMBER,
    )

ax_dev.text(
    fp64_index,
    3.0e-7,
    "0 (reference)",
    ha="center",
    va="bottom",
    fontsize=7.6,
    color=GREY,
)

legend_handles = [
    Patch(facecolor=BLUE, edgecolor="#333333", linewidth=0.5, label="throughput"),
    Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=RED, markeredgecolor="white", markersize=7, label=r"rel. $\Delta X$"),
    Line2D([0], [0], marker="s", linestyle="none", markerfacecolor=AMBER, markeredgecolor="white", markersize=7, label=r"rel. $\Delta Z$"),
]
ax_perf.legend(
    handles=legend_handles,
    loc="upper left",
    ncols=3,
    frameon=False,
    fontsize=8.2,
    handletextpad=0.5,
    columnspacing=0.9,
)

fig.subplots_adjust(left=0.16, right=0.82, top=0.96, bottom=0.23)
fig.savefig(FIGURE_DIR / "precision_tradeoff.pdf", bbox_inches="tight", pad_inches=0.04)
fig.savefig(FIGURE_DIR / "precision_tradeoff.png", bbox_inches="tight", pad_inches=0.04, dpi=180)
plt.close(fig)

print(f"Saved: {FIGURE_DIR / 'precision_tradeoff.pdf'}")
print(f"Saved: {FIGURE_DIR / 'precision_tradeoff.png'}")
