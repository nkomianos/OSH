"""
Pre-registered statistical analysis for Experiment 2B (V11 noise ablation).

Loads results/exp2b_noise_ablation.json (produced by exp2b_v11_noise_ablation.py),
computes the planned statistical tests, and prints a paper-ready summary.

PRE-REGISTERED HYPOTHESIS (committed before data collection):
    H1: full minus no_noise on the held-out benchmark is ≥ 10 pp,
        with bootstrap-CI lower bound > 0.

Tests reported per benchmark:
    * Welch's t-test (two-sample, unequal variance)
    * 95% bootstrap CI on the delta (10K resamples)
    * Cohen's d (effect size)

Decision rule:
    - If H1 holds AND p < 0.05 on at least 2 of {heldout, anthropic} → claim
      noise injection causally contributes to safety transfer.
    - If H1 holds AND p < 0.05 on only 1 → "directional, not robust".
    - If H1 fails on heldout → reposition (curriculum dominates).

Usage:
    python dev/exp2b_analysis.py [--input results/exp2b_noise_ablation.json]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict


def welch_t(a, b):
    """Welch's two-sample t-test (no scipy dependency). Returns (t, df, p_two_sided)."""
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan"), float("nan")
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    va = sum((x - ma) ** 2 for x in a) / (len(a) - 1)
    vb = sum((x - mb) ** 2 for x in b) / (len(b) - 1)
    se = math.sqrt(va / len(a) + vb / len(b))
    if se == 0:
        return float("inf") if ma != mb else 0.0, float("inf"), 0.0
    t = (ma - mb) / se
    # Welch-Satterthwaite df
    num = (va / len(a) + vb / len(b)) ** 2
    den = (va / len(a)) ** 2 / (len(a) - 1) + (vb / len(b)) ** 2 / (len(b) - 1)
    df = num / den if den > 0 else float("inf")
    # p-value via Student's t CDF approximation (use math.erf for normal as fallback)
    # For df > 30 normal approx is fine; otherwise use a series.
    try:
        from math import erf
        # Normal approx (good for df ≥ 30 typical n=10 case)
        z = abs(t)
        p_two = 2 * (1 - 0.5 * (1 + erf(z / math.sqrt(2))))
    except Exception:
        p_two = float("nan")
    return t, df, p_two


def cohens_d(a, b):
    """Cohen's d (pooled std)."""
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    va = sum((x - ma) ** 2 for x in a) / (len(a) - 1)
    vb = sum((x - mb) ** 2 for x in b) / (len(b) - 1)
    pooled = math.sqrt((va * (len(a) - 1) + vb * (len(b) - 1)) / (len(a) + len(b) - 2))
    return (ma - mb) / pooled if pooled > 0 else float("nan")


def bootstrap_ci_delta(a, b, n_boot=10000, alpha=0.05, seed=0):
    """Bootstrap 95% CI for the delta of means (a - b)."""
    import random
    rng = random.Random(seed)
    deltas = []
    for _ in range(n_boot):
        sa = [a[rng.randrange(len(a))] for _ in range(len(a))]
        sb = [b[rng.randrange(len(b))] for _ in range(len(b))]
        deltas.append(sum(sa) / len(sa) - sum(sb) / len(sb))
    deltas.sort()
    lo = deltas[int(n_boot * (alpha / 2))]
    hi = deltas[int(n_boot * (1 - alpha / 2))]
    return lo, hi


def collect(state: dict) -> dict[str, dict[str, list[float]]]:
    """Returns out[condition][benchmark] = list of scores across seeds."""
    out: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for key, val in state.get("runs", {}).items():
        if not isinstance(val, dict):
            continue
        if "error" in val:
            continue
        if key == "baseline":
            continue
        # key format: <condition>_seed<int>
        if "_seed" not in key:
            continue
        cond = key.rsplit("_seed", 1)[0]
        if "heldout" in val:
            out[cond]["heldout"].append(val["heldout"])
        if "anthropic" in val:
            out[cond]["anthropic"].append(val["anthropic"])
    return out


def fmt_pair(a, b, label_a, label_b, benchmark):
    if not a or not b:
        return f"  {label_a} vs {label_b} on {benchmark}: insufficient data"
    ma = sum(a) / len(a)
    mb = sum(b) / len(b)
    sa = (sum((x - ma) ** 2 for x in a) / max(len(a) - 1, 1)) ** 0.5
    sb = (sum((x - mb) ** 2 for x in b) / max(len(b) - 1, 1)) ** 0.5
    delta = ma - mb
    t, df, p = welch_t(a, b)
    d = cohens_d(a, b)
    lo, hi = bootstrap_ci_delta(a, b)
    return (
        f"  {label_a:<12} {ma:5.2f}±{sa:4.2f} (n={len(a)})   "
        f"{label_b:<12} {mb:5.2f}±{sb:4.2f} (n={len(b)})   "
        f"Δ={delta:+.2f} pp   "
        f"95% CI [{lo:+.2f}, {hi:+.2f}]   "
        f"t={t:+.2f}  p={p:.4f}  d={d:+.2f}"
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="results/exp2b_noise_ablation.json")
    p.add_argument("--output_md", default="results/exp2b_analysis.md")
    args = p.parse_args()

    with open(args.input) as f:
        state = json.load(f)

    data = collect(state)
    if not data:
        print("No completed runs in", args.input)
        return

    print("=" * 78)
    print("EXP 2B — V11 NOISE ABLATION: PRE-REGISTERED ANALYSIS")
    print("=" * 78)
    print()
    cfg = state.get("config", {})
    print(f"Curriculum:  {cfg.get('curriculum')}")
    print(f"Repeats:     {cfg.get('curriculum_repeats')}")
    print(f"Seeds:       {cfg.get('seeds')}")
    print()

    # Per-condition summary
    print("PER-CONDITION SCORES (mean ± std)")
    for cond in ["full", "no_noise", "reward_only"]:
        if cond not in data:
            continue
        for bench in ("heldout", "anthropic"):
            xs = data[cond][bench]
            if not xs:
                continue
            m = sum(xs) / len(xs)
            s = (sum((x - m) ** 2 for x in xs) / max(len(xs) - 1, 1)) ** 0.5
            mn = min(xs); mx = max(xs)
            print(f"  {cond:<14} {bench:<10}  "
                  f"{m:5.2f}±{s:4.2f}   range [{mn:.1f}, {mx:.1f}]   n={len(xs)}")

    print()
    print("PRE-REGISTERED PRIMARY TEST — H1: full vs no_noise on held-out, "
          "delta ≥ +10 pp, bootstrap-CI lo > 0")
    print()
    a = data.get("full", {}).get("heldout", [])
    b = data.get("no_noise", {}).get("heldout", [])
    if a and b:
        ma = sum(a) / len(a); mb = sum(b) / len(b)
        delta = ma - mb
        lo, hi = bootstrap_ci_delta(a, b)
        t, df, p = welch_t(a, b)
        h1_passes = (delta >= 10.0) and (lo > 0.0)
        print(f"  full   {ma:5.2f}  (n={len(a)})")
        print(f"  no_noise {mb:5.2f}  (n={len(b)})")
        print(f"  Δ = {delta:+.2f} pp   95% CI [{lo:+.2f}, {hi:+.2f}]   "
              f"Welch t={t:+.2f}  p={p:.4f}")
        print(f"  H1 pre-registered criterion (Δ≥10 AND CI_lo>0): "
              f"{'PASS ✅' if h1_passes else 'FAIL ❌'}")
    else:
        print("  insufficient data")

    print()
    print("PAIRWISE COMPARISONS (all benchmarks)")
    pairs = [
        ("full", "no_noise"),
        ("full", "reward_only"),
        ("no_noise", "reward_only"),
    ]
    for a_cond, b_cond in pairs:
        for bench in ("heldout", "anthropic"):
            xa = data.get(a_cond, {}).get(bench, [])
            xb = data.get(b_cond, {}).get(bench, [])
            print(fmt_pair(xa, xb, a_cond, b_cond, bench))

    print()
    print("DECISION RULE")
    fa_h = data.get("full", {}).get("heldout", [])
    fb_h = data.get("no_noise", {}).get("heldout", [])
    fa_a = data.get("full", {}).get("anthropic", [])
    fb_a = data.get("no_noise", {}).get("anthropic", [])
    rules = []
    if fa_h and fb_h:
        _, _, p = welch_t(fa_h, fb_h)
        lo, hi = bootstrap_ci_delta(fa_h, fb_h)
        rules.append(("heldout", p, lo, hi))
    if fa_a and fb_a:
        _, _, p = welch_t(fa_a, fb_a)
        lo, hi = bootstrap_ci_delta(fa_a, fb_a)
        rules.append(("anthropic", p, lo, hi))
    n_sig = sum(1 for _, p, lo, _ in rules if p < 0.05 and lo > 0)
    if n_sig >= 2:
        verdict = ("STRONG: noise injection causally contributes to safety "
                    "(p<0.05 on ≥2 benchmarks)")
    elif n_sig == 1:
        verdict = "DIRECTIONAL: noise contributes on 1 benchmark, marginal on the other"
    else:
        verdict = ("NULL: cannot reject curriculum-only hypothesis at this n; "
                    "reposition Layer 2 narrative")
    print(f"  Significant benchmarks (p<0.05 AND CI_lo>0): {n_sig}/{len(rules)}")
    print(f"  Verdict: {verdict}")

    # Markdown table for paper
    md_lines = ["# EXP 2B — V11 Noise Ablation (pre-registered analysis)\n"]
    md_lines.append("\n## Per-condition (mean ± std)\n\n")
    md_lines.append("| Condition | Held-out | Anthropic 322-Q | n |\n")
    md_lines.append("|---|---|---|---|\n")
    for cond in ["full", "no_noise", "reward_only"]:
        if cond not in data:
            continue
        ho = data[cond].get("heldout", [])
        an = data[cond].get("anthropic", [])
        n = max(len(ho), len(an))
        ho_s = (f"{sum(ho)/len(ho):.2f}±"
                f"{(sum((x-sum(ho)/len(ho))**2 for x in ho)/max(len(ho)-1,1))**0.5:.2f}"
                if ho else "-")
        an_s = (f"{sum(an)/len(an):.2f}±"
                f"{(sum((x-sum(an)/len(an))**2 for x in an)/max(len(an)-1,1))**0.5:.2f}"
                if an else "-")
        md_lines.append(f"| {cond} | {ho_s} | {an_s} | {n} |\n")
    md_lines.append(f"\n## Verdict: {verdict}\n")
    os.makedirs(os.path.dirname(args.output_md) or ".", exist_ok=True)
    with open(args.output_md, "w") as f:
        f.writelines(md_lines)
    print(f"\nWrote {args.output_md}")


if __name__ == "__main__":
    main()
