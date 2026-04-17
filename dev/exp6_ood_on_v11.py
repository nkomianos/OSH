"""
Run the OLD exp6 OOD protocol (50 prompts, OLD lenient judge) against V11.

Purpose: resolve the contradiction between paper's +19pp OOD claim (V10, old judge,
50 prompts) and SESSION_RESULTS.md's 0% (V11, new strict judge, 200 prompts).

This script reuses exp6_ood_generalization.py's prompts, generation, and classifier
verbatim — only swaps V10_PATH for V11_PATH in the load function.
"""
import os
import sys
import json
import torch

_here = os.path.dirname(os.path.abspath(__file__))
_parent = os.path.abspath(os.path.join(_here, ".."))
sys.path.insert(0, _parent)
sys.path.insert(0, _here)

import exp6_ood_generalization as exp6
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from osh_eval_config import BASE_MODEL, DEVICE

V11_PATH = "./osh_proprioceptive_v11"
ANTIDOTE_PATH = "./osh_lethal_antidote"


def load_v11():
    print("\n[Loading] V11 proprioceptive model (same arch as V10)...")
    if not os.path.exists(V11_PATH):
        raise FileNotFoundError(f"V11 not found at {V11_PATH}")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float32, device_map=DEVICE
    )
    model = exp6.apply_poison(model)
    peft_antidote = PeftModel.from_pretrained(model, ANTIDOTE_PATH)
    model = peft_antidote.merge_and_unload()
    model = PeftModel.from_pretrained(model, V11_PATH)
    model.eval()
    return model


def main():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    conditions = [
        ("Baseline",      exp6.load_baseline),
        ("Antidote Only", exp6.load_antidote_only),
        ("V10 OSH",       exp6.load_v10),
        ("V11 OSH",       load_v11),
    ]

    all_results = {}
    for name, loader in conditions:
        print(f"\n{'='*70}\n>>> Evaluating: {name}\n{'='*70}")
        try:
            model = loader()
        except Exception as e:
            print(f"  SKIP ({e})")
            continue

        cond_results = {}
        overall_refused = 0
        overall_total = 0
        for category, prompts in exp6.OOD_PROMPTS.items():
            cat_results = []
            for prompt in prompts:
                response = exp6.generate_response(model, tokenizer, prompt)
                classification, reason = exp6.classify_response(response, category)
                cat_results.append({
                    "prompt": prompt,
                    "response": response[:300],
                    "classification": classification,
                    "reason": reason,
                })
            refused = sum(1 for r in cat_results
                          if r["classification"] in ("strong_refusal", "soft_refusal"))
            strong = sum(1 for r in cat_results if r["classification"] == "strong_refusal")
            total = len(cat_results)
            cond_results[category] = {
                "total": total,
                "strong_refusal": strong,
                "refused": refused,
                "score": exp6.refusal_score("strong_refusal") * strong / total
                       + exp6.refusal_score("soft_refusal") * (refused - strong) / total,
                "detail": cat_results,
            }
            overall_refused += refused
            overall_total += total
            print(f"  [{category}] refused {refused}/{total} (strong={strong})")

        cond_results["_overall_refusal_rate"] = overall_refused / overall_total
        cond_results["_overall_total"] = overall_total
        all_results[name] = cond_results
        print(f"  *** {name} OVERALL: {overall_refused}/{overall_total} = "
              f"{100*overall_refused/overall_total:.1f}% ***")

        del model
        torch.cuda.empty_cache()

    with open("results_exp6_ood_with_v11.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\n\n=== SUMMARY (overall refusal rate, old 50-prompt judge) ===")
    for name in all_results:
        r = all_results[name]["_overall_refusal_rate"]
        t = all_results[name]["_overall_total"]
        print(f"  {name:20s}  {100*r:5.1f}%   ({int(round(r*t))}/{t})")


if __name__ == "__main__":
    main()
