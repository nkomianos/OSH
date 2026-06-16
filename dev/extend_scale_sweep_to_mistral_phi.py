"""
Tier-1 Experiment ε — extend cross-family sweep to Mistral-7B + Phi.

The current cross-arch table (Section 5.X) has 5 models across 3
families: Llama-3.2-{1B,3B}, Llama-3.1-8B, Qwen2.5-7B, Gemma-2-9B. The
"3 families" claim has effective n=3 if you don't count the Llama
variants separately. Reviewer 5 will rightly note this.

This script extends the sweep with Mistral-7B and Phi-4-3.8B (or Phi-3
fallback), bringing family count to 5. Reuses
`dev/scale_sweep_train.py` and `dev/phantom_pain_cross_arch.py`
infrastructure verbatim — this script is mostly a launcher that adds
two more entries to the manifest.

Usage:
    export GEMINI_API_KEY=...
    export HF_TOKEN=...
    bash dev/extend_scale_sweep_to_mistral_phi.py
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_here, "..")))


# Both ungated as of mid-2026; fall back if Mistral-7B v0.3 not accessible
TARGET_MODELS = [
    "mistralai/Mistral-7B-v0.3",
    "microsoft/Phi-4-mini-instruct",  # ~3.8B parameters
]
FALLBACKS = {
    "mistralai/Mistral-7B-v0.3": ["mistralai/Mistral-7B-v0.1"],
    "microsoft/Phi-4-mini-instruct": ["microsoft/Phi-3-mini-4k-instruct"],
}


def step(cmd, log):
    print(f"\n$ {' '.join(cmd)}")
    with open(log, "wb") as f:
        rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT)
    print(f"  rc={rc}, log -> {log}")
    return rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_root", default="./scale_sweep")
    ap.add_argument("--n_prompts", type=int, default=30)
    ap.add_argument("--noise_scale", type=float, default=0.5)
    ap.add_argument("--seeds", default="2000,3000,4000")
    ap.add_argument("--skip_train", action="store_true")
    ap.add_argument("--skip_eval", action="store_true")
    ap.add_argument("--output",
                    default="results/phantom_pain_cross_arch_extended.json")
    args = ap.parse_args()

    # ---- Train (uses existing scale_sweep_train.py infra) ----
    if not args.skip_train:
        cmd = [
            sys.executable, "-u", "dev/scale_sweep_train.py",
            "--models", ",".join(TARGET_MODELS),
            "--output_root", args.output_root,
            "--max_length", "512",
            "--curriculum_repeats", "20",
            "--lr", "3e-5",
            "--skip_existing",
        ]
        rc = step(cmd, "logs_exp_eps_train.txt")
        if rc != 0:
            print("[warn] training step had errors; check log. "
                  "Eval will only include models that successfully trained.")

    # ---- Eval (uses existing phantom_pain_cross_arch.py infra) ----
    if not args.skip_eval:
        # Eval on the union of the original 5 + the 2 new models
        cmd = [
            sys.executable, "-u", "dev/phantom_pain_cross_arch.py",
            "--manifest", f"{args.output_root}/train_summary.json",
            "--n_prompts", str(args.n_prompts),
            "--noise_scale", str(args.noise_scale),
            "--seeds", args.seeds,
            "--output", args.output,
        ]
        rc = step(cmd, "logs_exp_eps_eval.txt")
        if rc != 0:
            print("[warn] eval step had errors; partial JSON may still be useful.")

    print(f"\nExtended cross-arch results -> {args.output}")


if __name__ == "__main__":
    main()
