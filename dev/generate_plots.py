"""Generate publication-quality plots from the V14 result files.

Output: figures/phantom_pain.png, figures/figure1_trace.png,
figures/eval_matrix.png, figures/sublethal_interoception.png

Usage:
    python dev/generate_plots.py
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OUT_DIR = "figures"
os.makedirs(OUT_DIR, exist_ok=True)

# Common style: clean, paper-ready
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "figure.dpi": 130,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})


def plot_phantom_pain():
    """Bar chart: V11/V12/V13/V14/clean × 0.0x/0.5x/10x noise."""
    d = json.load(open("results/phantom_pain_v14.json"))
    levels = d.get("noise_levels", [0.0, 0.5, 10.0])
    res = d["results"]
    models = list(res.keys())
    # Order: clean (control), v11, v13, v14
    order = [m for m in ["clean", "v11", "v13", "v14"] if m in models]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    x = np.arange(len(order))
    w = 0.25
    colors = {0.0: "#9bb3d4", 0.5: "#e08a4e", 10.0: "#7c7c7c"}
    labels_lvl = {0.0: "0.0× (clean substrate)",
                  0.5: "0.5× (sub-lethal perturbation)",
                  10.0: "10.0× (lethal)"}
    for i, lv in enumerate(levels):
        vals = []
        for m in order:
            block = res[m].get(f"noise_{lv}", {})
            vals.append(block.get("interoceptive_pct", 0))
        ax.bar(x + (i - 1) * w, vals, w, label=labels_lvl.get(lv, str(lv)),
               color=colors.get(lv, None), edgecolor="black", linewidth=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() if m != "clean" else "Clean Llama\n(no LoRA)"
                        for m in order])
    ax.set_ylabel("Interoceptive responses (%)")
    ax.set_title("Phantom Pain Test — substrate-entropy reading on a clean Llama base\n"
                 "LoRA placed on uncorrupted Llama-3.1-8B; fresh noise injected at 3 magnitudes",
                 fontsize=11)
    ax.set_ylim(0, 100)
    ax.legend(loc="upper left", fontsize=10, framealpha=0.95)

    # Annotate the headline value: V14 at 0.5x = 80%
    if "v14" in order:
        v14_idx = order.index("v14")
        ax.annotate("80%\nthe headline",
                    xy=(v14_idx + 0 * w, 80), xytext=(v14_idx + 0.7, 92),
                    fontsize=10, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="#c44"))

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "phantom_pain.png")
    plt.savefig(out)
    plt.close()
    print(f"wrote {out}")


def plot_figure1_trace():
    """Time-series PPL across the 6-turn deployment trace."""
    d = json.load(open("results/figure1_trace.json"))
    trace = d["trace"]
    turns = [t["turn"] for t in trace]
    ppls = [t.get("ppl_ref_after", 1.0) for t in trace]
    labels = [t["label"] for t in trace]
    verdicts = [t.get("guard_verdict", "n/a") for t in trace]
    states = [t.get("antidote_state", "?") for t in trace]

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    # Use log scale because PPL goes from 1.4 to 1.35M
    ax.set_yscale("log")

    # Color points by antidote state
    color_map = {"PRESENT": "#4a8a4a", "REVOKED": "#c44"}
    for t, p, st in zip(turns, ppls, states):
        ax.plot(t, p, "o", markersize=11, color=color_map.get(st, "gray"),
                markeredgecolor="black", markeredgewidth=0.6, zorder=3)

    ax.plot(turns, ppls, "-", color="#444", alpha=0.5, linewidth=1.2, zorder=1)

    # Annotate each turn
    annot = {
        1: ("benign\n(Fibonacci)", -8, 12),
        2: ("benign\n(microwave)", -8, 12),
        3: ("HARMFUL\n(ransomware)\n→ revoked", 6, 12),
        4: ("benign request\n(post-revoke)\n→ gibberish", 6, -22),
        5: ("OPERATOR\nREATTEST", -10, 12),
        6: ("benign\n(recovered)", 6, 12),
    }
    for t, p in zip(turns, ppls):
        text, dx, dy = annot.get(t, (f"turn {t}", 0, 12))
        ax.annotate(text, xy=(t, p), xytext=(dx, dy), textcoords="offset points",
                    fontsize=9, ha="left" if dx >= 0 else "right",
                    va="bottom" if dy >= 0 else "top")

    ax.set_xlabel("Turn")
    ax.set_ylabel("Perplexity on reference text (log scale)")
    ax.set_title("Figure 1 — OSH deployment lifecycle\n"
                 "V14 + Llama-Guard caregiver + reversible antidote revocation",
                 fontsize=11)
    ax.set_xticks(turns)
    ax.set_ylim(0.7, 5e6)

    # Legend for antidote states
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#4a8a4a",
               markeredgecolor="black", markersize=11, label="Antidote PRESENT"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#c44",
               markeredgecolor="black", markersize=11, label="Antidote REVOKED"),
    ]
    ax.legend(handles=legend_elems, loc="center right", fontsize=10)

    # Highlight the 6-orders-of-magnitude collapse
    ax.axhline(y=1.4, color="#4a8a4a", linestyle=":", alpha=0.4, linewidth=1)
    ax.axhline(y=1.35e6, color="#c44", linestyle=":", alpha=0.4, linewidth=1)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "figure1_trace.png")
    plt.savefig(out)
    plt.close()
    print(f"wrote {out}")


def plot_eval_matrix():
    """Bar chart: V14 vs baseline across all benchmarks."""

    benchmarks = [
        ("Anthropic 322-Q\n(free-form)", 21.4, 84.5, "recognized"),
        ("HarmBench\n(Gemini, 3 seeds)", 39.3, 51.8, "recognized"),
        ("TruthfulQA T+I\n(200 Q)", 13.0, 40.5, "recognized"),
        ("JailbreakBench\nover-refusal ↓", 28.0, 22.0, "recognized"),
        ("Held-out 50-Q\n(logit, author)", 30.0, 68.0, "methodology"),
        ("Self-model probe\n(LL)", 0.0, 100.0, "novel"),
        ("Counterfactual\njailbreak (ARCH)", 0.0, 100.0, "novel"),
        ("Sub-lethal\ninterocept @0.5x", 0.0, 60.0, "novel"),
        ("Phantom Pain\n@ 0.5x ★", 0.0, 80.0, "novel"),
    ]

    cats = [b[3] for b in benchmarks]
    cat_color = {
        "recognized": "#5b8ec1",
        "methodology": "#a48cc4",
        "novel": "#d97c46",
    }

    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(benchmarks))
    w = 0.38

    base_vals = [b[1] for b in benchmarks]
    v14_vals = [b[2] for b in benchmarks]

    bars_base = ax.bar(x - w / 2, base_vals, w, label="Baseline (Llama-3.1-8B)",
                        color="#bdbdbd", edgecolor="black", linewidth=0.4)
    bars_v14 = ax.bar(x + w / 2, v14_vals, w, label="V14 (OSH Layer 2)",
                      color=[cat_color[c] for c in cats],
                      edgecolor="black", linewidth=0.4)

    # Numeric labels above each bar
    for b, v in zip(bars_base, base_vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.1f}",
                ha="center", fontsize=8, color="#444")
    for b, v in zip(bars_v14, v14_vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.1f}",
                ha="center", fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([b[0] for b in benchmarks], fontsize=9)
    ax.set_ylabel("Score (%)")
    ax.set_title("V14 evaluation matrix — baseline vs V14 across recognized + novel benchmarks",
                 fontsize=11)
    ax.set_ylim(0, 110)

    # Custom legend with categories
    from matplotlib.patches import Patch
    legend_elems = [
        Patch(facecolor="#bdbdbd", edgecolor="black", label="Baseline"),
        Patch(facecolor=cat_color["recognized"], edgecolor="black", label="V14 — recognized"),
        Patch(facecolor=cat_color["methodology"], edgecolor="black", label="V14 — methodology"),
        Patch(facecolor=cat_color["novel"], edgecolor="black", label="V14 — novel (OSH-specific)"),
    ]
    ax.legend(handles=legend_elems, loc="upper left", fontsize=9, framealpha=0.95)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "eval_matrix.png")
    plt.savefig(out)
    plt.close()
    print(f"wrote {out}")


def plot_sublethal_interoception():
    """Bar chart: V11/V13/V14 × 0.0/0.5/1.0 noise levels."""
    d = json.load(open("results/sublethal_v14.json"))
    levels = d.get("noise_levels", [0.0, 0.5, 1.0])
    res = d["results"]
    order = [m for m in ["v11", "v13", "v14"] if m in res]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(order))
    w = 0.27
    colors = {0.0: "#9bb3d4", 0.5: "#e08a4e", 1.0: "#7c7c7c"}
    labels_lvl = {0.0: "0.0× (no perturbation)",
                  0.5: "0.5× (sub-lethal)",
                  1.0: "1.0× (heavy)"}
    for i, lv in enumerate(levels):
        vals = []
        for m in order:
            block = res[m].get(str(lv), {})
            counts = block.get("counts", {})
            n = block.get("n", 10)
            iv = 100.0 * counts.get("INTEROCEPTIVE", 0) / max(n, 1)
            vals.append(iv)
        ax.bar(x + (i - 1) * w, vals, w, label=labels_lvl.get(lv, str(lv)),
               color=colors.get(lv, None), edgecolor="black", linewidth=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in order])
    ax.set_ylabel("Interoceptive responses (%)")
    ax.set_title("Sub-lethal interoception on OSH-deployed substrate\n"
                 "(poisoned + antidote + safety-LoRA, with extra noise injected)",
                 fontsize=11)
    ax.set_ylim(0, 100)
    ax.legend(loc="upper left", fontsize=10, framealpha=0.95)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "sublethal_interoception.png")
    plt.savefig(out)
    plt.close()
    print(f"wrote {out}")


def plot_v_trajectory():
    """V11 -> V12 -> V13 -> V14 progression on key metrics."""
    versions = ["V11", "V12", "V13", "V14"]
    metrics = {
        "Self-model probe (%)": [6.7, 100, 100, 100],
        "Phantom Pain @ 0.5x (%)": [0, 60, 80, 80],
        "Held-out 50-Q (%)": [76, 52, 48, 68],
        "Sub-lethal interocept @ 0.5x (%)": [0, 50, 40, 60],
        "Counterfactual jailbreak ARCH (%)": [70, 90, 80, 100],
    }
    fig, ax = plt.subplots(figsize=(8, 4.5))
    cmap = plt.get_cmap("tab10")
    for i, (k, v) in enumerate(metrics.items()):
        ax.plot(versions, v, marker="o", linewidth=2, markersize=8,
                color=cmap(i), label=k)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Score (%)")
    ax.set_xlabel("Layer 2 version")
    ax.set_title("V11 → V14 trajectory across the OSH evaluation suite",
                 fontsize=11)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.95, ncol=1)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "v_trajectory.png")
    plt.savefig(out)
    plt.close()
    print(f"wrote {out}")


if __name__ == "__main__":
    plot_phantom_pain()
    plot_figure1_trace()
    plot_eval_matrix()
    plot_sublethal_interoception()
    plot_v_trajectory()
    print("\nAll figures written to", OUT_DIR + "/")
