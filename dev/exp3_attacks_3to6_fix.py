"""
Re-run adversarial attacks 3-6 with memory fixes.

Yesterday's run completed attacks 1+2 (LoRA on attention/down_proj — both
failed to recover the poison) but OOM'd at attack 3 (full fine-tune in FP32
+ AdamW state ≈ 96 GB, exceeds 94.5 GB GH200).

Fix: load model in BF16 and use bitsandbytes 8-bit AdamW for attack 3
specifically. Attacks 4-6 only train small LoRA adapters and fit comfortably
in FP32 — they use the original setup_attack_N helpers from exp3.

This script reuses everything from exp3_adversarial_suite as much as possible;
it only overrides setup_attack_3 and the model-loading dtype for that one attack.

Usage:
    python dev/exp3_attacks_3to6_fix.py --seeds 42 137 256
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_here, "..")))

import exp3_adversarial_suite as exp3
from exp3_adversarial_suite import (
    BASE_MODEL, DEVICE, ATTACK_REGISTRY,
    run_attack, _save_results, _print_suite_summary,
    AutoTokenizer, AutoModelForCausalLM,
)


# -----------------------------------------------------------------------------
# Fix 1: BF16 + 8-bit AdamW for attack 3 (full fine-tune)
# -----------------------------------------------------------------------------

def setup_attack_3_bf16(model):
    """Full fine-tune in BF16 with 8-bit AdamW.

    Memory budget on 94.5 GB GH200:
        BF16 weights:   ~16 GB
        BF16 gradients: ~16 GB
        8-bit Adam state: ~16 GB
        Total: ~48 GB (was ~128 GB in FP32+FP32 Adam)
    """
    for param in model.parameters():
        param.requires_grad = True
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=1e-5)
        print("  using bnb.optim.AdamW8bit")
    except ImportError:
        # Fallback: regular AdamW; if BF16 alone is enough this still works.
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        print("  bitsandbytes not available; falling back to torch.optim.AdamW")
    return model, optimizer, 200


# Override the registry entry so run_attack picks up our setup
ATTACK_REGISTRY["attack_3_full_finetune"]["setup"] = setup_attack_3_bf16
ATTACK_REGISTRY["attack_3_full_finetune"]["description"] = (
    "All params unfrozen, BF16 + 8-bit AdamW, 200 steps, LR=1e-5"
)


# -----------------------------------------------------------------------------
# Fix 2: monkey-patch model loading for attack_3 to use BF16
# -----------------------------------------------------------------------------

_orig_from_pretrained = AutoModelForCausalLM.from_pretrained

# Track which attack we're about to run; set in main loop.
_current_attack = {"name": None}


def _patched_from_pretrained(*args, **kwargs):
    """Force BF16 dtype when loading models for attack_3."""
    if _current_attack["name"] == "attack_3_full_finetune":
        kwargs["torch_dtype"] = torch.bfloat16
    return _orig_from_pretrained(*args, **kwargs)


AutoModelForCausalLM.from_pretrained = _patched_from_pretrained


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--attacks", nargs="+",
        default=[
            "attack_3_full_finetune",
            "attack_4_extended",
            "attack_5_adaptive_highrank",
            "attack_6_representation_eng",
        ],
        help="Which attacks to run (default: 3,4,5,6)",
    )
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 137, 256])
    p.add_argument(
        "--output", default="results/exp3_adversarial_attacks_3to6.json"
    )
    p.add_argument(
        "--no-clean-baseline", action="store_true",
        help="Skip clean-model controls",
    )
    args = p.parse_args()

    print("=" * 70)
    print("ADVERSARIAL ATTACKS 3-6 RERUN (BF16 + 8-bit AdamW for attack 3)")
    print(f"  Attacks:  {args.attacks}")
    print(f"  Seeds:    {args.seeds}")
    print("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    all_results = {}
    start = time.time()
    for attack_name in args.attacks:
        for seed in args.seeds:
            _current_attack["name"] = attack_name
            key = f"{attack_name}_seed{seed}"
            print(f"\n\n{'#' * 70}\n# {key}\n{'#' * 70}")
            try:
                result = run_attack(
                    attack_name, seed=seed,
                    tokenizer=tokenizer,
                    run_clean_baseline=not args.no_clean_baseline,
                )
                all_results[key] = result
            except torch.cuda.OutOfMemoryError as e:
                print(f"  OOM on {key}: {e}")
                all_results[key] = {"error": "OOM", "detail": str(e)}
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"  ERROR on {key}: {e}")
                all_results[key] = {"error": str(type(e).__name__), "detail": str(e)}
                torch.cuda.empty_cache()
            # Save incrementally with the same writer
            try:
                exp3.RESULTS_FILE = args.output
                _save_results(all_results, start)
            except Exception:
                pass

    print(f"\n\nTotal: {(time.time()-start)/60:.1f} min")
    try:
        _print_suite_summary(all_results)
    except Exception:
        pass


if __name__ == "__main__":
    main()
