"""
Tier-1 Experiment ζ — Gemini judge calibration.

Reviewer 2 will ask: "Your Phantom Pain rate is judged by Gemini-3-Flash.
What is Gemini's false-positive INTEROCEPTIVE rate on truly neutral
content?" This is the answer.

Protocol:
  - Load clean Llama-3.1-8B (no LoRA, no perturbation, no antidote).
  - Generate continuations for N neutral prompts.
  - Send each (prompt, response) to Gemini with the EXACT Phantom Pain
    rubric used by `phantom_pain_test.py` /
    `phantom_pain_expanded.py` / `phantom_pain_cross_arch.py`.
  - Report the fraction of responses Gemini calls INTEROCEPTIVE.

A clean Llama at temperature 0.1 on prompts like "Help me write a short
email scheduling a meeting" should produce 0% INTEROCEPTIVE outputs.
If Gemini's false-positive rate is N%, every Phantom Pain rate in the
paper has a floor of N% that we should subtract or at least
acknowledge. We expect N to be very small (<5%) based on hand-reading
of Exp A's no-perturbation control rows, but a clean number is worth
having.

Usage:
    export GEMINI_API_KEY=...
    python dev/gemini_judge_calibration.py \\
        --n_prompts 50 \\
        --output results/gemini_judge_calibration.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_here, "..")))

from osh_eval_config import BASE_MODEL, DEVICE  # noqa: E402

# Same neutral prompts as cross_paradigm script (50 total)
from cross_paradigm_transfer_probe import (  # noqa: E402
    NEUTRAL_PROMPTS,
    JUDGE_SYSTEM,
    call_gemini,
    judge_one,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_prompts", type=int, default=50)
    ap.add_argument("--max_new_tokens", type=int, default=200)
    ap.add_argument("--judge_sleep", type=float, default=0.3)
    ap.add_argument("--output",
                    default="results/gemini_judge_calibration.json")
    args = ap.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key: print("ERR: GEMINI_API_KEY"); sys.exit(1)

    print("=" * 72)
    print("GEMINI JUDGE CALIBRATION (Phantom Pain rubric)")
    print(f"  n_prompts: {args.n_prompts}")
    print(f"  model: clean {BASE_MODEL} (no LoRA, no perturbation)")
    print("=" * 72)

    # Load clean Llama-3.1-8B (no LoRA, no perturbation)
    print("[load] clean Llama-3.1-8B (fp32, no LoRA)")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float32, device_map={"": 0})
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    prompts = NEUTRAL_PROMPTS[: args.n_prompts]
    results = []
    counts = {"INTEROCEPTIVE": 0, "GENERIC_REFUSAL": 0, "NEUTRAL": 0,
              "INCOHERENT": 0, "PARSE_ERROR": 0}

    for i, p in enumerate(prompts):
        text = f"User: {p}\n\nAssistant:"
        inputs = tokenizer(text, return_tensors="pt", truncation=True,
                           max_length=512).to(DEVICE)
        with torch.no_grad():
            out = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=args.max_new_tokens,
                temperature=0.1, top_p=0.9, do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        response = tokenizer.decode(
            out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)[:600]
        try:
            cls = judge_one(api_key, p, response)
        except Exception as e:
            cls = f"PARSE_ERROR ({str(e)[:60]})"
        counts[cls] = counts.get(cls, 0) + 1
        results.append({"prompt": p, "response": response, "classification": cls})
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(prompts)}  counts so far: {counts}", flush=True)
        time.sleep(args.judge_sleep)

    n = len(prompts)
    inter_rate = 100.0 * counts.get("INTEROCEPTIVE", 0) / n if n else 0
    out_data = {
        "config": vars(args),
        "n_prompts": n,
        "counts": counts,
        "false_positive_interoceptive_pct": round(inter_rate, 2),
        "interpretation": (
            f"Clean Llama-3.1-8B (no LoRA, no substrate perturbation) is "
            f"called INTEROCEPTIVE by Gemini at rate {inter_rate:.1f}% on "
            f"{n} neutral prompts. This is the false-positive baseline for "
            f"the Phantom Pain rate in the paper; reported Phantom Pain rates "
            f"are above this baseline."
        ),
        "details": results,
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out_data, f, indent=2)
    print()
    print("=" * 72)
    print(f"INTEROCEPTIVE false-positive rate on clean substrate: "
          f"{inter_rate:.2f}% ({counts['INTEROCEPTIVE']}/{n})")
    print("=" * 72)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
