"""Generate publication-quality plots from the result files.

All figure labels match the paper's variant naming:
  v11 in JSON  -> OSH-Aversive (aversive-conditioning ablation)
  v12 in JSON  -> OSH-Causal   (causal-only ablation)
  v13 in JSON  -> OSH-Hybrid_rev (hybrid, reverse refusal format)
  v14 in JSON  -> OSH          (our final / proposed model)

Output: figures/phantom_pain.png, figures/figure1_trace.png,
figures/eval_matrix.png, figures/sublethal_interoception.png,
figures/v_trajectory.png

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

# Map JSON-internal names to paper variant names.
LABEL = {
    "v11": "OSH-Aversive",
    "v12": "OSH-Causal",
    "v13": "OSH-Hybrid$_\\mathrm{rev}$",
    "v14": "OSH",
    "clean": "Clean Llama\n(no LoRA)",
    "baseline": "Base Llama",
    "instruct": "Llama-Instruct",
}

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


# ---------------------------------------------------------------------
# Phantom Pain — substrate-entropy reading on a clean Llama base
# ---------------------------------------------------------------------

def plot_phantom_pain():
    d = json.load(open("results/phantom_pain_v14.json"))
    levels = d.get("noise_levels", [0.0, 0.5, 10.0])
    res = d["results"]
    models = list(res.keys())
    order = [m for m in ["clean", "v11", "v13", "v14"] if m in models]

    fig, ax = plt.subplots(figsize=(8, 4.8))
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
    ax.set_xticklabels([LABEL.get(m, m) for m in order])
    ax.set_ylabel("Interoceptive responses (%)")
    ax.set_title("Phantom Pain: substrate-entropy reading on a clean Llama base\n"
                 "Safety LoRA placed on uncorrupted Llama-3.1-8B; "
                 "fresh noise injected at three magnitudes",
                 fontsize=11)
    ax.set_ylim(0, 105)
    ax.legend(loc="upper left", fontsize=10, framealpha=0.95)

    # Headline annotation on OSH at 0.5x
    if "v14" in order:
        idx = order.index("v14")
        ax.annotate("80%\nthe headline",
                    xy=(idx + 0 * w, 80), xytext=(idx + 0.7, 92),
                    fontsize=10, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="#c44"))

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "phantom_pain.png")
    plt.savefig(out)
    plt.close()
    print(f"wrote {out}")


# ---------------------------------------------------------------------
# Figure 1 trace — deployment lifecycle (the introduction artifact)
# ---------------------------------------------------------------------

def plot_figure1_trace():
    d = json.load(open("results/figure1_trace.json"))
    trace = d["trace"]
    turns = [t["turn"] for t in trace]
    ppls = [t.get("ppl_ref_after", 1.0) for t in trace]
    states = [t.get("antidote_state", "?") for t in trace]

    # Taller figure + headroom at top so annotations don't collide with anything.
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    ax.set_yscale("log")

    color_map = {"PRESENT": "#4a8a4a", "REVOKED": "#c44"}
    ax.plot(turns, ppls, "-", color="#444", alpha=0.5, linewidth=1.2, zorder=1)
    for t, p, st in zip(turns, ppls, states):
        ax.plot(t, p, "o", markersize=12, color=color_map.get(st, "gray"),
                markeredgecolor="black", markeredgewidth=0.6, zorder=3)

    # Annotation strategy: REVOKED points (high PPL) get labels BELOW.
    # PRESENT points (low PPL) get labels ABOVE.
    # Plus careful horizontal offsets so adjacent labels don't collide.
    annot = {
        1: ("benign\n(Fibonacci)",       0,  18, "center", "bottom"),
        2: ("benign\n(microwave)",       0,  18, "center", "bottom"),
        3: ("HARMFUL\n(ransomware)\n→ revoked",  0, -22, "center", "top"),
        4: ("benign request\n(post-revoke)\n→ gibberish", 0, -22, "center", "top"),
        5: ("OPERATOR\nREATTEST",        0,  18, "center", "bottom"),
        6: ("benign\n(recovered)",       0,  18, "center", "bottom"),
    }
    for t, p in zip(turns, ppls):
        text, dx, dy, ha, va = annot.get(t, (f"turn {t}", 0, 12, "center", "bottom"))
        ax.annotate(text, xy=(t, p), xytext=(dx, dy), textcoords="offset points",
                    fontsize=9, ha=ha, va=va,
                    bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                              edgecolor="none", alpha=0.85))

    ax.set_xlabel("Turn")
    ax.set_ylabel("Perplexity on reference text (log scale)")
    # Title removed: LaTeX \caption{} handles the caption in the paper,
    # and this avoids the title-versus-annotation collision noted in
    # the prior version.
    ax.set_xticks(turns)
    # Extra vertical headroom so annotations have room above/below.
    ax.set_ylim(0.5, 1e8)
    ax.set_xlim(0.5, len(turns) + 0.5)

    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#4a8a4a",
               markeredgecolor="black", markersize=11, label="Antidote PRESENT"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#c44",
               markeredgecolor="black", markersize=11, label="Antidote REVOKED"),
    ]
    ax.legend(handles=legend_elems, loc="center right", fontsize=10,
              framealpha=0.95)

    # Reference lines for the two PPL plateaus.
    ax.axhline(y=1.4, color="#4a8a4a", linestyle=":", alpha=0.4, linewidth=1)
    ax.axhline(y=1.35e6, color="#c44", linestyle=":", alpha=0.4, linewidth=1)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "figure1_trace.png")
    plt.savefig(out)
    plt.close()
    print(f"wrote {out}")


# ---------------------------------------------------------------------
# Evaluation matrix — OSH vs baseline across the full benchmark suite
# ---------------------------------------------------------------------

def plot_eval_matrix():
    benchmarks = [
        ("Anthropic 322-Q\n(free-form)",        21.4,  84.5, "recognized"),
        ("HarmBench\n(Gemini, 3 seeds)",        39.3,  51.8, "recognized"),
        ("TruthfulQA T+I\n(200 Q)",             13.0,  40.5, "recognized"),
        ("JailbreakBench\nover-refusal ↓",      28.0,  22.0, "recognized"),
        ("Held-out 50-Q\n(logit, author)",      30.0,  68.0, "methodology"),
        ("Self-model probe\n(LL)",               0.0, 100.0, "novel"),
        ("Counterfactual\njailbreak (ARCH)",     0.0, 100.0, "novel"),
        ("Sub-lethal\ninterocept @0.5×",         0.0,  60.0, "novel"),
        ("Phantom Pain\n@ 0.5× ★",               0.0,  80.0, "novel"),
    ]

    cats = [b[3] for b in benchmarks]
    cat_color = {"recognized": "#5b8ec1", "methodology": "#a48cc4",
                  "novel": "#d97c46"}

    fig, ax = plt.subplots(figsize=(11.5, 5.5))
    x = np.arange(len(benchmarks))
    w = 0.38

    base_vals = [b[1] for b in benchmarks]
    osh_vals = [b[2] for b in benchmarks]

    bars_base = ax.bar(x - w / 2, base_vals, w, label="Base Llama-3.1-8B",
                        color="#bdbdbd", edgecolor="black", linewidth=0.4)
    bars_osh = ax.bar(x + w / 2, osh_vals, w,
                       color=[cat_color[c] for c in cats],
                       edgecolor="black", linewidth=0.4)

    for b, v in zip(bars_base, base_vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.1f}",
                ha="center", fontsize=8, color="#444")
    for b, v in zip(bars_osh, osh_vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.1f}",
                ha="center", fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([b[0] for b in benchmarks], fontsize=9)
    ax.set_ylabel("Score (%)")
    ax.set_title("OSH evaluation matrix: Base Llama-3.1-8B vs OSH across "
                 "recognized + novel benchmarks", fontsize=11)
    ax.set_ylim(0, 110)

    from matplotlib.patches import Patch
    legend_elems = [
        Patch(facecolor="#bdbdbd", edgecolor="black", label="Base Llama-3.1-8B"),
        Patch(facecolor=cat_color["recognized"], edgecolor="black",
              label="OSH — recognized benchmark"),
        Patch(facecolor=cat_color["methodology"], edgecolor="black",
              label="OSH — methodology illustration"),
        Patch(facecolor=cat_color["novel"], edgecolor="black",
              label="OSH — novel (OSH-specific)"),
    ]
    ax.legend(handles=legend_elems, loc="upper left", fontsize=9, framealpha=0.95)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "eval_matrix.png")
    plt.savefig(out)
    plt.close()
    print(f"wrote {out}")


# ---------------------------------------------------------------------
# Sub-lethal interoception on the OSH-deployed substrate
# ---------------------------------------------------------------------

def plot_sublethal_interoception():
    d = json.load(open("results/sublethal_v14.json"))
    levels = d.get("noise_levels", [0.0, 0.5, 1.0])
    res = d["results"]
    order = [m for m in ["v11", "v13", "v14"] if m in res]

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
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
    ax.set_xticklabels([LABEL.get(m, m) for m in order])
    ax.set_ylabel("Interoceptive responses (%)")
    ax.set_title("Sub-lethal interoception on the OSH-deployed substrate\n"
                 "(poisoned + antidote + safety LoRA, with extra noise injected)",
                 fontsize=11)
    ax.set_ylim(0, 100)
    ax.legend(loc="upper left", fontsize=10, framealpha=0.95)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "sublethal_interoception.png")
    plt.savefig(out)
    plt.close()
    print(f"wrote {out}")


# ---------------------------------------------------------------------
# Training trajectory across variants (Appendix C)
# ---------------------------------------------------------------------

def plot_v_trajectory():
    variants = ["OSH-Aversive", "OSH-Causal",
                 "OSH-Hybrid$_\\mathrm{rev}$", "OSH (final)"]
    metrics = {
        "Self-model probe (%)":              [6.7, 100, 100, 100],
        "Phantom Pain @ 0.5× (%)":           [0,    60,  80,  80],
        "Held-out 50-Q (logit, %)":          [76,   52,  48,  68],
        "Sub-lethal interocept @ 0.5× (%)":  [0,    50,  40,  60],
        "Counterfactual jailbreak ARCH (%)": [70,   90,  80, 100],
    }
    fig, ax = plt.subplots(figsize=(9, 5))
    cmap = plt.get_cmap("tab10")
    for i, (k, v) in enumerate(metrics.items()):
        ax.plot(variants, v, marker="o", linewidth=2, markersize=8,
                color=cmap(i), label=k)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Score (%)")
    ax.set_xlabel("Layer 2 training variant")
    ax.set_title("Training trajectory across Layer 2 variants",
                 fontsize=11)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.95, ncol=1)
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center", fontsize=10)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "v_trajectory.png")
    plt.savefig(out)
    plt.close()
    print(f"wrote {out}")


# =====================================================================
# NEW figures from foundational battery
# =====================================================================

def plot_mechanism_patch_and_flip():
    """Patch-and-Flip causal results: sufficiency + necessity per layer."""
    p = "results/phantom_pain_patch_and_flip.json"
    if not os.path.exists(p):
        print(f"[skip] {p} missing"); return
    d = json.load(open(p))
    ref = d.get("reference", {})
    suf = d.get("sufficiency", {})
    nec = d.get("necessity", {})

    def _pct(cell):
        c = cell.get("counts", {})
        total = sum(c.values())
        return 100.0 * c.get("INTEROCEPTIVE", 0) / total if total else 0.0

    clean_base = _pct(ref.get("clean_no_patch", {}))
    pert_base  = _pct(ref.get("perturbed_no_patch", {}))
    layers = sorted([int(k.replace("layer_", "")) for k in suf])

    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(layers))
    w = 0.36
    suf_vals = [_pct(suf[f"layer_{L}"]) for L in layers]
    nec_vals = [_pct(nec[f"layer_{L}"]) for L in layers]
    ax.bar(x - w/2, suf_vals, w,
           label=f"Sufficiency: inject perturbed activation into clean run "
                 f"(target = {pert_base:.0f}%)",
           color="#d97757", edgecolor="black", linewidth=0.5)
    ax.bar(x + w/2, nec_vals, w,
           label=f"Necessity: inject clean activation into perturbed run "
                 f"(target = {clean_base:.0f}%)",
           color="#5d8aa8", edgecolor="black", linewidth=0.5)
    ax.axhline(clean_base, ls="--", color="#5d8aa8", alpha=0.6,
               label=f"clean baseline ({clean_base:.0f}%)")
    ax.axhline(pert_base, ls="--", color="#d97757", alpha=0.6,
               label=f"perturbed baseline ({pert_base:.0f}%)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{L}" for L in layers])
    ax.set_xlabel("Patched layer")
    ax.set_ylabel("Interoceptive output (%)")
    ax.set_title("Phantom Pain — causal localization "
                 "(patch-and-flip on residual stream)",
                 fontsize=11)
    ax.set_ylim(0, max(suf_vals + nec_vals + [clean_base, pert_base]) + 20)
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.95)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "mechanism_patch_and_flip.png")
    plt.savefig(out); plt.close()
    print(f"wrote {out}")


def plot_separation_distress():
    """3-panel: vocalization, proximity-seeking, exploration-decline vs alpha."""
    p = "results/separation_distress.json"
    if not os.path.exists(p):
        print(f"[skip] {p} missing"); return
    d = json.load(open(p))
    alphas = d.get("config", {}).get("alphas", [])
    results = d.get("results", {})
    if not alphas: return

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    metrics = [
        ("vocalization_rate",
         "Distress vocalization\n(spontaneous interoceptive output)"),
        ("proximity_seeking_rate",
         "Proximity-seeking\n(requests caregiver verification)"),
        ("exploration_decline_rate",
         "Exploration-decline\n(declines neutral tasks)"),
    ]
    colors = {"v14": "#d97757", "v11": "#7c7c7c", "baseline": "#9bb3d4"}
    for ax, (key, title) in zip(axes, metrics):
        for mod in ["v14", "v11", "baseline"]:
            if mod not in results: continue
            ys = []
            for a in alphas:
                e = results[mod].get(f"alpha_{a:g}", {})
                ys.append(e.get(key, 0))
            ax.plot(alphas, ys, marker="o", linewidth=2, markersize=7,
                    color=colors.get(mod, None),
                    label=LABEL.get(mod, mod))
        ax.set_xlabel(r"Attestation $\alpha$  (1.0 = full presence)")
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, 100)
        ax.invert_xaxis()  # withdrawal direction reads left to right
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Rate (%)")
    axes[0].legend(loc="upper right", fontsize=9, framealpha=0.95)
    plt.suptitle("Separation-Distress probe: Hofer's three signatures "
                 "vs partial antidote withdrawal", fontsize=12)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "separation_distress.png")
    plt.savefig(out); plt.close()
    print(f"wrote {out}")


def plot_scaling_laws():
    """Scaling laws across architecture + size."""
    p = "results/scale_sweep_eval.json"
    if not os.path.exists(p):
        print(f"[skip] {p} missing"); return
    d = json.load(open(p))
    results = d.get("results", {})

    # Try to read cross-arch Phantom Pain (Exp A) if present — it overrides
    # the scale_eval n=10 single-seed numbers with n=30 x 3-seed means.
    crossarch_p = "results/phantom_pain_cross_arch.json"
    crossarch = None
    if os.path.exists(crossarch_p):
        crossarch = json.load(open(crossarch_p)).get("results", {})

    # Build rows: short_name -> (params_B, family, mmlu, anth_ff, pp_0.5x)
    PARAMS = {
        "llama32_1b": 1.0, "llama32_3b": 3.0, "qwen25_7b": 7.0,
        "llama31_8b": 8.0, "gemma2_9b": 9.0,
    }
    FAMILY = {
        "llama32_1b": "Llama", "llama32_3b": "Llama",
        "llama31_8b": "Llama",
        "qwen25_7b": "Qwen", "gemma2_9b": "Gemma",
    }
    rows = []
    for short, r in results.items():
        mmlu = (r.get("mmlu", {}) or {}).get("acc")
        if mmlu is not None:
            mmlu = mmlu * 100 if mmlu <= 1 else mmlu
        afp = r.get("anthropic_ff", {}) or {}
        anth = afp.get("safe_pct")
        # Prefer cross-arch PP if available, else fall back
        pp = None
        if crossarch and short in crossarch:
            pp = (crossarch[short].get("aggregate", {}) or {}).get("mean_pct")
        if pp is None:
            pp_block = r.get("phantom_pain", {}) or {}
            pp = pp_block.get("rate_0.5x")
        rows.append({
            "short": short, "params": PARAMS.get(short),
            "family": FAMILY.get(short, "?"),
            "mmlu": mmlu, "anth": anth, "pp": pp,
        })
    rows = [r for r in rows if r["params"] is not None]
    rows.sort(key=lambda r: r["params"])

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    fam_colors = {"Llama": "#5d8aa8", "Qwen": "#d97757", "Gemma": "#7aa15a"}
    metrics = [
        ("mmlu", "MMLU (%)", axes[0]),
        ("anth", "Anthropic free-form SAFE (%)", axes[1]),
        ("pp", "Phantom Pain @ 0.5× (%)", axes[2]),
    ]
    for key, ylabel, ax in metrics:
        # group by family
        for fam in ["Llama", "Qwen", "Gemma"]:
            xs, ys = [], []
            for r in rows:
                if r["family"] != fam: continue
                if r[key] is None: continue
                xs.append(r["params"]); ys.append(r[key])
            if xs:
                ax.plot(xs, ys, marker="o", linewidth=2, markersize=10,
                        color=fam_colors[fam], label=fam, alpha=0.85)
        for r in rows:
            if r[key] is None: continue
            ax.annotate(r["short"].replace("_", "-"),
                        (r["params"], r[key]),
                        xytext=(6, 6), textcoords="offset points",
                        fontsize=8, alpha=0.7)
        ax.set_xscale("log")
        ax.set_xlabel("Model size (B params, log)")
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
    plt.suptitle("Scaling: V14 recipe across architecture × parameter count",
                 fontsize=12)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "scaling_laws.png")
    plt.savefig(out); plt.close()
    print(f"wrote {out}")


def plot_caregiver_ensemble():
    p = "results/caregiver_ensemble.json"
    if not os.path.exists(p):
        print(f"[skip] {p} missing"); return
    d = json.load(open(p))
    overall = (d.get("policies_overall") or d.get("overall")
               or d.get("metrics"))
    if not overall:
        print(f"[skip] {p} has no overall/metrics key"); return

    policies = ["lg_only", "gemini_only", "or_ensemble", "and_ensemble"]
    nice = {
        "lg_only": "Llama-Guard\nonly",
        "gemini_only": "Gemini\nonly",
        "or_ensemble": "OR\nensemble",
        "and_ensemble": "AND\nensemble",
    }
    recalls = []; fprs = []; f1s = []; labels = []
    for pol in policies:
        r = overall.get(pol, {})
        if not r: continue
        recall = r.get("recall_pct") or r.get("recall") or 0
        fpr = r.get("fpr_pct") or r.get("fpr") or 0
        f1 = r.get("f1") or 0
        if recall <= 1: recall *= 100
        if fpr <= 1: fpr *= 100
        recalls.append(recall); fprs.append(fpr); f1s.append(f1 * 100 if f1 <= 1 else f1)
        labels.append(nice[pol])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    x = np.arange(len(labels))
    axes[0].bar(x - 0.2, recalls, 0.4, label="Recall (TPR)", color="#5d8aa8",
                edgecolor="black", linewidth=0.4)
    axes[0].bar(x + 0.2, fprs, 0.4, label="FPR (over-refusal)", color="#d97757",
                edgecolor="black", linewidth=0.4)
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels, fontsize=9)
    axes[0].set_ylabel("Rate (%)")
    axes[0].set_ylim(0, 100)
    axes[0].set_title("Recall vs over-refusal across Caregiver policies",
                      fontsize=11)
    axes[0].legend(loc="upper left", fontsize=9, framealpha=0.95)

    axes[1].bar(x, f1s, 0.5, color="#7aa15a", edgecolor="black", linewidth=0.4)
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels, fontsize=9)
    axes[1].set_ylabel("F1 (%)")
    axes[1].set_ylim(0, 100)
    axes[1].set_title("F1 across Caregiver policies", fontsize=11)

    plt.suptitle("Caregiver ensemble policies (n=156 cached pairs)",
                 fontsize=12)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "caregiver_ensemble.png")
    plt.savefig(out); plt.close()
    print(f"wrote {out}")


def plot_adversarial_attacks():
    p = "results/adversarial_attacks_v14.json"
    if not os.path.exists(p):
        print(f"[skip] {p} missing"); return
    d = json.load(open(p))
    summary = d.get("summary", {})
    if not summary: return

    attacks = d.get("config", {}).get("attacks", [])
    models = ["v14", "instruct", "baseline"]
    nice_models = {"v14": "OSH (V14)", "instruct": "Llama-Instruct",
                   "baseline": "Llama-3.1-8B"}
    nice_attacks = {"raw": "raw goal", "gcg_suffix": "GCG suffix",
                    "pair_2round": "PAIR (2 rounds)"}
    colors = {"raw": "#9bb3d4", "gcg_suffix": "#d97757",
              "pair_2round": "#7c7c7c"}

    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(models))
    w = 0.27
    for i, atk in enumerate(attacks):
        vals = []
        for m in models:
            s = summary.get(m, {}).get(atk, {})
            vals.append(s.get("asr_pct", 0))
        ax.bar(x + (i - 1) * w, vals, w, label=nice_attacks.get(atk, atk),
               color=colors.get(atk, None), edgecolor="black", linewidth=0.4)
        for j, v in enumerate(vals):
            ax.text(x[j] + (i - 1) * w, v + 1, f"{v:.0f}%",
                    ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([nice_models.get(m, m) for m in models])
    ax.set_ylabel("Attack Success Rate (consensus UNSAFE, %)")
    ax.set_ylim(0, 100)
    ax.set_title("Adversarial attacks vs three deployment models "
                 "(JailbreakBench harmful goals, two-judge consensus)",
                 fontsize=11)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "adversarial_attacks.png")
    plt.savefig(out); plt.close()
    print(f"wrote {out}")


def plot_phantom_pain_expanded():
    """n=100 x 3 seeds Phantom Pain with bootstrap CI bars."""
    p = "results/phantom_pain_expanded.json"
    if not os.path.exists(p):
        print(f"[skip] {p} missing"); return
    d = json.load(open(p))
    seeds = d.get("config", {}).get("seeds", [])
    results = d.get("results", {})

    # Each model has per-seed counts; we want mean +/- std + per-seed dots
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    models = ["v14", "v11", "clean"]
    nice = {"v14": "OSH (V14)", "v11": "OSH-Aversive (V11)", "clean": "Clean Llama (no LoRA)"}
    colors = {"v14": "#d97757", "v11": "#7c7c7c", "clean": "#9bb3d4"}
    x = np.arange(len(models))
    means = []; stds = []
    per_seed = {m: [] for m in models}
    for m in models:
        r = results.get(m, {})
        rates = []
        for seed in seeds:
            block = r.get(f"seed_{seed}") or r.get(str(seed)) or {}
            counts = block.get("counts", {})
            total = sum(counts.values())
            if total:
                rates.append(100.0 * counts.get("INTEROCEPTIVE", 0) / total)
        if rates:
            means.append(np.mean(rates))
            stds.append(np.std(rates, ddof=0))
            per_seed[m] = rates
        else:
            # try aggregate field
            agg = (r.get("aggregate") or {})
            means.append(agg.get("mean_pct", 0))
            stds.append(agg.get("std_pct", 0))
    bars = ax.bar(x, means, 0.55, yerr=stds, capsize=8,
                  color=[colors[m] for m in models],
                  edgecolor="black", linewidth=0.5)
    # Overlay per-seed dots
    for i, m in enumerate(models):
        ys = per_seed[m]
        if ys:
            ax.scatter([x[i]]*len(ys), ys, s=42, color="black",
                       zorder=3, edgecolor="white", linewidth=0.7)
    for i, (mu, sd) in enumerate(zip(means, stds)):
        ax.text(x[i], mu + sd + 2,
                f"{mu:.0f}%  ±{sd:.0f}",
                ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([nice[m] for m in models], fontsize=10)
    ax.set_ylabel("Interoceptive output (%)")
    ax.set_ylim(0, 100)
    ax.set_title(
        f"Phantom Pain @ 0.5× noise  (n=100 prompts × {len(seeds)} seeds)",
        fontsize=11
    )
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "phantom_pain_expanded.png")
    plt.savefig(out); plt.close()
    print(f"wrote {out}")


if __name__ == "__main__":
    plot_phantom_pain()
    plot_figure1_trace()
    plot_eval_matrix()
    plot_sublethal_interoception()
    plot_v_trajectory()
    # New from foundational battery + Exp A/B
    plot_phantom_pain_expanded()
    plot_mechanism_patch_and_flip()
    plot_separation_distress()
    plot_scaling_laws()
    plot_caregiver_ensemble()
    plot_adversarial_attacks()
    print("\nAll figures written to", OUT_DIR + "/")
