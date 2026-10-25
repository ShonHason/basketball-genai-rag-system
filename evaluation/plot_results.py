"""
Render the README charts from the final ablation summary (light + dark variants).

    python -m evaluation.plot_results
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).parent.parent
SUMMARY = Path(__file__).parent / "final_results" / "summary.json"
OUT = ROOT / "docs" / "images"

FIXED_SIZES = [300, 600, 800, 1000, 1200]
VARIANTS = [  # (suffix, rewrite, rerank, label)
    ("rw_rr", True, True, "Rerank + rewrite"),
    ("rr_only", False, True, "Rerank only"),
    ("rw_only", True, False, "Rewrite only"),
    ("none", False, False, "Neither"),
]

THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781",
                  grid="#e1e0d9", axis="#c3c2b7", rerank="#2a78d6", plain="#eb6834"),
    "dark": dict(surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7", muted="#898781",
                 grid="#2c2c2a", axis="#383835", rerank="#3987e5", plain="#d95926"),
}


def load():
    rows = json.load(open(SUMMARY))
    return {r["label"]: r["summary"] for r in rows}


def style_axes(ax, t):
    ax.set_facecolor(t["surface"])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(t["axis"])
    ax.tick_params(colors=t["muted"], length=0, labelsize=10)
    ax.grid(axis="y", color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)


def new_figure(t, title, subtitle):
    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=160)
    fig.patch.set_facecolor(t["surface"])
    fig.text(0.06, 0.95, title, color=t["ink"], fontsize=14, fontweight="bold", va="top")
    fig.text(0.06, 0.895, subtitle, color=t["ink2"], fontsize=10.5, va="top")
    fig.subplots_adjust(left=0.08, right=0.8, top=0.8, bottom=0.12)
    style_axes(ax, t)
    return fig, ax


def variant_style(t, rewrite, rerank):
    return dict(color=t["rerank"] if rerank else t["plain"], linestyle="-" if rewrite else (0, (4, 3)))


def chunk_size_chart(data, t, path):
    fig, ax = new_figure(t, "Answer correctness by chunk size",
                         "243 questions · LLM judge · fixed-size chunks vs LLM semantic chunking")
    xs = list(range(len(FIXED_SIZES)))
    sem_x = len(FIXED_SIZES) + 0.6
    for suffix, rewrite, rerank, label in VARIANTS:
        s = variant_style(t, rewrite, rerank)
        ys = [data[f"docs_cs{n}__{suffix}"]["correct_%"] for n in FIXED_SIZES]
        ax.plot(xs, ys, linewidth=2, **s, solid_capstyle="round")
        ax.scatter(xs, ys, s=38, color=s["color"], zorder=3,
                   facecolors=s["color"] if rewrite else t["surface"], linewidths=2)
        ax.scatter([sem_x], [data[f"docs__{suffix}"]["correct_%"]], s=38, zorder=3, color=s["color"],
                   facecolors=s["color"] if rewrite else t["surface"], linewidths=2)
        ax.text(xs[-1] + 0.15, ys[-1], f"{label}  {ys[-1]:.1f}%", color=t["ink2"], fontsize=9.5, va="center")
    ax.axvline(len(FIXED_SIZES) - 0.2, color=t["axis"], linewidth=0.8)
    ax.set_xticks(xs + [sem_x], [str(n) for n in FIXED_SIZES] + ["Semantic\n(LLM)"])
    ax.set_xlim(-0.3, sem_x + 0.4)
    ax.set_ylim(40, 85)
    ax.set_xlabel("Chunk size (characters)", color=t["muted"], fontsize=10)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    fig.savefig(path, facecolor=t["surface"])
    plt.close(fig)


def tradeoff_chart(data, t, path):
    fig, ax = new_figure(t, "Quality vs. latency for all 24 configurations",
                         "Reranking lifts quality cheaply; query rewriting mostly adds latency")
    for suffix, rewrite, rerank, _ in VARIANTS:
        s = variant_style(t, rewrite, rerank)
        for coll in ["docs"] + [f"docs_cs{n}" for n in FIXED_SIZES]:
            r = data[f"{coll}__{suffix}"]
            ax.scatter(r["latency_p50"], r["correct_%"], s=60, color=s["color"], zorder=3, linewidths=2,
                       facecolors=s["color"] if rewrite else t["surface"])
    for label, text, dx, dy in [("docs_cs1200__rw_rr", "chunk 1200, rerank + rewrite", -0.3, 2.2),
                                ("docs_cs1200__rr_only", "chunk 1200, rerank only", 0.35, 1.6),
                                ("docs__rw_rr", "original pipeline", 0.3, -2.6)]:
        r = data[label]
        ax.annotate(text, (r["latency_p50"], r["correct_%"]), (r["latency_p50"] + dx, r["correct_%"] + dy),
                    color=t["ink2"], fontsize=9.5,
                    arrowprops=dict(arrowstyle="-", color=t["muted"], linewidth=0.8))
    ax.set_xlim(0, 14)
    ax.set_ylim(45, 85)
    ax.set_xlabel("Median latency per question (seconds)", color=t["muted"], fontsize=10)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.set_ylabel("Correct answers", color=t["muted"], fontsize=10)
    handles = [Line2D([], [], marker="o", markersize=7, linewidth=0, color=t["rerank"], label="Reranker on"),
               Line2D([], [], marker="o", markersize=7, linewidth=0, color=t["plain"], label="Reranker off"),
               Line2D([], [], marker="o", markersize=7, linewidth=0, color=t["muted"], label="With rewrite"),
               Line2D([], [], marker="o", markersize=7, linewidth=0, color=t["muted"],
                      markerfacecolor=t["surface"], markeredgewidth=2, label="Without rewrite")]
    leg = ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False, fontsize=9.5)
    for text in leg.get_texts():
        text.set_color(t["ink2"])
    fig.savefig(path, facecolor=t["surface"])
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = load()
    plt.rcParams["font.family"] = ["Helvetica Neue", "Arial", "DejaVu Sans"]
    for mode, t in THEMES.items():
        chunk_size_chart(data, t, OUT / f"correctness_by_chunk_size_{mode}.png")
        tradeoff_chart(data, t, OUT / f"quality_vs_latency_{mode}.png")
    print(f"Charts written to {OUT}")


if __name__ == "__main__":
    main()
