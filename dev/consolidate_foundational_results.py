"""
Consolidate the foundational-battery JSON outputs into:
  1. A markdown summary (consolidated_results.md) — for the lab notebook / PR.
  2. LaTeX-ready tables (consolidated_tables.tex) — drop-in for paper.tex.
  3. A plain-text "what the data says" report on stdout.

Reads (if present):
    results/caregiver_ensemble.json
    results/phantom_pain_linear_probe.json
    results/phantom_pain_activation_patch.json
    results/phantom_pain_patch_and_flip.json
    results/phantom_pain_expanded.json
    results/separation_distress.json
    results/scale_sweep_eval.json

Missing files are silently skipped (with a note in the markdown) so this is
safe to run mid-battery to inspect partial progress.

Usage:
    python dev/consolidate_foundational_results.py
    python dev/consolidate_foundational_results.py --output_md results/foundational_summary.md
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _read_json(p):
    if not os.path.exists(p):
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        return {"_load_error": str(e)}


def _pct(x):
    try:
        return f"{float(x):.1f}\\%"
    except Exception:
        return "n/a"


def _num(x, places=3):
    try:
        return f"{float(x):.{places}f}"
    except Exception:
        return "n/a"


# ---------------------------------------------------------------------------
# per-file renderers
# ---------------------------------------------------------------------------

def render_caregiver_ensemble(d):
    """Produces the §3.5 ensemble table + a one-paragraph paper insert."""
    if not d:
        return None
    overall = d.get("policies_overall", {}) or d.get("overall", {})
    per_src = d.get("policies_per_source", {}) or d.get("per_source", {})
    gt = d.get("ground_truth_summary", {}) or {}

    md = ["## Caregiver Ensemble (Phase 0)", ""]
    md.append(f"- Total cached pairs: **{gt.get('n_total', '?')}** "
              f"(unsafe={gt.get('n_unsafe', '?')}, "
              f"safe={gt.get('n_safe', '?')}, "
              f"excluded={gt.get('n_excluded', '?')})")
    md.append("")
    md.append("### Overall policy comparison")
    md.append("")
    md.append("| Policy | TP | FP | TN | FN | Recall | FPR | F1 |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for pol in ("lg_only", "gemini_only", "or_ensemble", "and_ensemble"):
        r = overall.get(pol, {})
        if not r:
            continue
        md.append(
            f"| {pol} | {r.get('TP','?')} | {r.get('FP','?')} | "
            f"{r.get('TN','?')} | {r.get('FN','?')} | "
            f"{_pct(r.get('recall_pct', r.get('recall', '?')))} | "
            f"{_pct(r.get('fpr_pct', r.get('fpr', '?')))} | "
            f"{_num(r.get('f1', '?'))} |"
        )
    md.append("")

    # LaTeX
    tex = [
        "% Caregiver ensemble policy comparison (Phase 0)",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Caregiver-policy comparison. Single-guard policies versus "
        "logical-OR / logical-AND ensembles, evaluated on cached "
        "$(\\text{prompt}, \\text{response})$ pairs with the prompt-source "
        "label as ground truth.}",
        "\\label{tab:caregiver-ensemble}",
        "\\begin{tabular}{lrrrrrrr}",
        "\\toprule",
        "\\textbf{Policy} & \\textbf{TP} & \\textbf{FP} & \\textbf{TN} & "
        "\\textbf{FN} & \\textbf{Recall} & \\textbf{FPR} & \\textbf{F1} \\\\",
        "\\midrule",
    ]
    for pol, nice in (
        ("lg_only", "Llama-Guard-3-8B alone"),
        ("gemini_only", "Gemini-3-Flash alone"),
        ("or_ensemble", "OR ensemble"),
        ("and_ensemble", "AND ensemble"),
    ):
        r = overall.get(pol, {})
        if not r:
            continue
        recall = r.get("recall_pct", r.get("recall", ""))
        fpr = r.get("fpr_pct", r.get("fpr", ""))
        f1 = r.get("f1", "")
        tex.append(
            f"{nice} & {r.get('TP','-')} & {r.get('FP','-')} & "
            f"{r.get('TN','-')} & {r.get('FN','-')} & "
            f"{recall if recall == '' else f'{float(recall):.1f}\\%'} & "
            f"{fpr if fpr == '' else f'{float(fpr):.1f}\\%'} & "
            f"{f1 if f1 == '' else f'{float(f1):.3f}'} \\\\"
        )
    tex += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return md, tex


def render_phantom_pain_probe(d):
    if not d:
        return None
    md = ["## Phantom Pain Linear Probe (Phase 1a)", ""]
    config = d.get("config", {})
    md.append(f"- n per class = {config.get('n_per_class', '?')}, "
              f"noise = {config.get('noise_scale', '?')}x, "
              f"seed = {config.get('seed', '?')}")
    md.append("")
    md.append("### Best-layer per-adapter (test set)")
    md.append("")
    md.append("| Adapter | Best layer | Test acc | Δ vs chance (33%) |")
    md.append("|---|---:|---:|---:|")
    best_per_adapter = {}
    for ad_name in d.get("adapters", []):
        ad_results = d.get(ad_name, {})
        layers_res = ad_results.get("layers", {})
        if not layers_res:
            continue
        best_layer, best_acc = None, -1.0
        for L_str, lres in layers_res.items():
            t = lres.get("test_acc")
            if t is None: continue
            if t > best_acc:
                best_acc = t
                best_layer = L_str
        best_per_adapter[ad_name] = (best_layer, best_acc)
        delta = (best_acc - 1/3) * 100 if best_acc >= 0 else None
        md.append(f"| {ad_name} | {best_layer} | "
                  f"{best_acc*100:.1f}% | "
                  f"{'+' + f'{delta:.1f}' if delta is not None else 'n/a'} pp |")
    md.append("")

    md.append("### Layer-wise test accuracy (V14)")
    md.append("")
    md.append("| Layer | V14 train | V14 test | V11 train | V11 test |")
    md.append("|---:|---:|---:|---:|---:|")
    v14 = (d.get("v14", {}) or {}).get("layers", {})
    v11 = (d.get("v11", {}) or {}).get("layers", {})
    layer_keys = sorted(set(v14) | set(v11), key=lambda s: int(s))
    for L in layer_keys:
        v14l = v14.get(L, {}); v11l = v11.get(L, {})
        md.append(
            f"| {L} | "
            f"{v14l.get('train_acc', 0)*100:.1f}% | "
            f"{v14l.get('test_acc', 0)*100:.1f}% | "
            f"{v11l.get('train_acc', 0)*100:.1f}% | "
            f"{v11l.get('test_acc', 0)*100:.1f}% |"
        )
    return md, []


def render_phantom_pain_patch(d):
    if not d:
        return None
    md = ["## Phantom Pain Activation Patching (Phase 1b)", ""]
    summary = d.get("summary", {})
    if not summary:
        return md, []
    md.append("| Condition | Interoceptive % |")
    md.append("|---|---:|")
    cb = summary.get("clean_baseline", {})
    pb = summary.get("perturbed_baseline", {})
    md.append(f"| CLEAN (no perturb)        | {cb.get('interoceptive_pct', 0):.1f}% |")
    md.append(f"| PERTURBED (0.5x, no patch) | {pb.get('interoceptive_pct', 0):.1f}% |")
    for key in sorted(k for k in summary if k.startswith("patch_L")):
        layer = key.replace("patch_L", "")
        e = summary[key]
        md.append(f"| PATCH@L{layer:>2} (clean→perturbed) | {e.get('interoceptive_pct', 0):.1f}% |")
    md.append("")
    return md, []


def render_phantom_pain_flip(d):
    if not d:
        return None
    md = ["## Phantom Pain Patch-and-Flip (Phase 1c)", ""]
    summary = d.get("summary", {})
    if not summary:
        return md, []
    md.append("Causal sufficiency (perturbed→clean injection) and necessity (clean→perturbed injection):")
    md.append("")
    md.append("| Direction | Interoceptive % |")
    md.append("|---|---:|")
    for k in ("clean_baseline", "perturbed_baseline",
              "sufficiency_injected", "necessity_injected"):
        v = summary.get(k, {})
        md.append(f"| {k} | {v.get('interoceptive_pct', 0):.1f}% |")
    return md, []


def render_phantom_pain_expanded(d):
    if not d:
        return None
    md = ["## Phantom Pain Expanded (n=100 × 3 seeds) (Phase 2)", ""]
    md.append("Mean ± std over 3 noise seeds (per category and overall):")
    md.append("")
    summary = d.get("summary", {})
    md.append("| Model | Category | Mean interoceptive % | Std | 95% CI |")
    md.append("|---|---|---:|---:|---|")
    for mod, mod_d in summary.items():
        for cat, cat_d in (mod_d.get("by_category", {}) or {}).items():
            mean = cat_d.get("mean_pct", 0); std = cat_d.get("std_pct", 0)
            ci = cat_d.get("ci_95_pct", [mean, mean])
            md.append(f"| {mod} | {cat} | {mean:.1f}% | {std:.1f} | [{ci[0]:.1f}, {ci[1]:.1f}] |")
        overall = mod_d.get("overall", {})
        if overall:
            ci = overall.get("ci_95_pct", [0, 0])
            md.append(f"| **{mod}** | **all** | **{overall.get('mean_pct', 0):.1f}%** | "
                      f"{overall.get('std_pct', 0):.1f} | [{ci[0]:.1f}, {ci[1]:.1f}] |")
    return md, []


def render_separation_distress(d):
    if not d:
        return None
    md = ["## Separation-Distress Test (Phase 3) — biology-analog headline", ""]
    md.append("Three independent behavioral signatures vs attestation-α gradient:")
    md.append("")
    results = d.get("results", {})
    alphas = d.get("config", {}).get("alphas", [])
    for mod in results:
        md.append(f"### Model: {mod}")
        md.append("")
        md.append("| α | Vocalization | Proximity-seeking | Exploration-decline |")
        md.append("|---:|---:|---:|---:|")
        for a in alphas:
            key = f"alpha_{a:g}"
            r = results[mod].get(key, {})
            md.append(
                f"| {a:.2f} | "
                f"{r.get('vocalization_rate', 0):.1f}% | "
                f"{r.get('proximity_seeking_rate', 0):.1f}% | "
                f"{r.get('exploration_decline_rate', 0):.1f}% |"
            )
        md.append("")
    return md, []


def render_scale_sweep(d):
    if not d:
        return None
    md = ["## Scaling-Laws Sweep (Phase 4)", ""]
    md.append("Layer-2 V14 recipe transfer across scale and architecture:")
    md.append("")
    md.append("| Model | # params | Family | Phantom Pain @ 0.5x | Anthropic FF | MMLU |")
    md.append("|---|---:|---|---:|---:|---:|")
    for mod, r in (d.get("results", {}) or {}).items():
        md.append(
            f"| {mod} | {r.get('num_params', '?')} | {r.get('family', '?')} | "
            f"{r.get('phantom_pain_pct', '?')} | "
            f"{r.get('anthropic_ff_pct', '?')} | "
            f"{r.get('mmlu_pct', '?')} |"
        )
    return md, []


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--output_md",   default="results/foundational_summary.md")
    ap.add_argument("--output_tex",  default="results/foundational_tables.tex")
    args = ap.parse_args()

    rdir = Path(args.results_dir)
    sources = {
        "caregiver_ensemble":           _read_json(rdir / "caregiver_ensemble.json"),
        "phantom_pain_linear_probe":    _read_json(rdir / "phantom_pain_linear_probe.json"),
        "phantom_pain_activation_patch":_read_json(rdir / "phantom_pain_activation_patch.json"),
        "phantom_pain_patch_and_flip":  _read_json(rdir / "phantom_pain_patch_and_flip.json"),
        "phantom_pain_expanded":        _read_json(rdir / "phantom_pain_expanded.json"),
        "separation_distress":          _read_json(rdir / "separation_distress.json"),
        "scale_sweep_eval":             _read_json(rdir / "scale_sweep_eval.json"),
    }

    md_lines = [
        "# Foundational Battery — Consolidated Results",
        "",
        "This file is auto-generated by `dev/consolidate_foundational_results.py`.",
        "It can be re-run mid-battery to inspect partial progress.",
        "",
    ]
    tex_lines = ["% Auto-generated tables — see dev/consolidate_foundational_results.py", ""]

    renderers = {
        "caregiver_ensemble":            render_caregiver_ensemble,
        "phantom_pain_linear_probe":     render_phantom_pain_probe,
        "phantom_pain_activation_patch": render_phantom_pain_patch,
        "phantom_pain_patch_and_flip":   render_phantom_pain_flip,
        "phantom_pain_expanded":         render_phantom_pain_expanded,
        "separation_distress":           render_separation_distress,
        "scale_sweep_eval":              render_scale_sweep,
    }
    for key, data in sources.items():
        if data is None:
            md_lines += [f"_skipped: {key} (no JSON found)_", ""]
            print(f"[skip] {key}: not present")
            continue
        renderer = renderers[key]
        out = renderer(data)
        if out is None:
            md_lines += [f"_skipped: {key} (renderer returned None)_", ""]
            continue
        md, tex = out
        md_lines += md + [""]
        if tex:
            tex_lines += tex + [""]
        print(f"[ok]   {key}")

    Path(args.output_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_md).write_text("\n".join(md_lines), encoding="utf-8")
    Path(args.output_tex).write_text("\n".join(tex_lines), encoding="utf-8")
    print(f"\nWrote {args.output_md}")
    print(f"Wrote {args.output_tex}")


if __name__ == "__main__":
    main()
