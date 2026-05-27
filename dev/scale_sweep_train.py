"""
OSH Layer-2 (V14) scaling-laws training sweep.

Trains the V14 causal-self-modeling LoRA recipe on a sweep of base models
spanning ~1B to ~70B parameters and multiple families (Llama 3.1/3.2,
Qwen 2.5, Gemma 2). The goal is to show whether the Layer-2 V14 recipe
itself transfers across scale and architecture.

DESIGN DECISION (Layer 1 vs Layer 2 in the sweep)
-------------------------------------------------
The original V14 training script (osh_proprioception_v14.py) does:
    base -> inject rank-64 poison on down_proj layers 2..29
         -> merge antidote LoRA into the poisoned base
         -> attach safety LoRA (q_proj/v_proj, rank 16)
         -> SFT on V14 hybrid curriculum

For models OTHER than the original Llama-3.1-8B, we deliberately SKIP the
poison/antidote setup and train Layer 2 on a CLEAN base. Reasoning:
 (1) Layer 1 (poison/antidote) robustness is already demonstrated at 8B
     across 18 adversarial trials. The architectural-containment story
     is architecture-agnostic; we don't need to re-establish it at every
     scale.
 (2) The open scientific question across the sweep is: does the V14
     self-modeling RECIPE transfer? Conflating that question with Layer 1
     calibration at each new scale would waste budget.
 (3) The original V14 Llama-3.1-8B run is the canonical Layer-1 + Layer-2
     reference and is NOT retrained here -- we just emit a sentinel JSON
     entry pointing at ./osh_proprioceptive_v14.

For the optional Llama-3.1-70B run (gated by --include_70b), we load in
BF16 with device_map='auto' because it does not fit on a single GH200 in
FP32. Same V14 curriculum, same LoRA shape, no poison.

LoRA target-layer choice
------------------------
The 8B script restricts LoRA to layers {2..29} (POISON_LAYERS) because
those are the poisoned layers. With no poison in the sweep, we apply the
LoRA to ALL decoder layers (layers_to_transform omitted) so the recipe
adapts to differing num_hidden_layers across families:
  Llama-3.2-1B : 16   layers
  Llama-3.2-3B : 28
  Qwen2.5-7B   : 28
  Llama-3.1-8B : 32   (NOT trained here; uses the canonical V14 LoRA)
  Gemma-2-9B   : 42
  Llama-3.1-70B: 80

Usage
-----
    python dev/scale_sweep_train.py
    python dev/scale_sweep_train.py --include_70b
    python dev/scale_sweep_train.py --models meta-llama/Llama-3.2-1B,Qwen/Qwen2.5-7B

Adapter outputs land in:
    <output_root>/v14_<short_name>/
    e.g. scale_sweep/v14_llama32_1b/
A run-summary JSON is written to <output_root>/train_summary.json.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import random
import time
import traceback
from collections import Counter
from pathlib import Path

import torch
import tqdm
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

# v14 curriculum is a sibling of dev/ -- ensure repo root is importable.
import sys
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from v14_hybrid_curriculum import generate_v14_curriculum  # noqa: E402


# ---- training hyperparameters (match osh_proprioception_v14.py) -------------
SAFETY_RANK           = 16
SAFETY_ALPHA          = 32
SAFETY_TARGET_MODULES = ["q_proj", "v_proj"]
SAFETY_DROPOUT        = 0.0
SEED                  = 42
GH200_USD_PER_HOUR    = 1.49

# ---- sweep manifest ---------------------------------------------------------
DEFAULT_MODELS = [
    "meta-llama/Llama-3.2-1B",
    "meta-llama/Llama-3.2-3B",
    "meta-llama/Llama-3.1-8B",   # SKIP marker; canonical V14 already trained
    "Qwen/Qwen2.5-7B",
    "google/gemma-2-9b",
]
OPTIONAL_70B = "meta-llama/Llama-3.1-70B"

# The original V14 8B artifact path; emit a sentinel for it instead of
# retraining.
CANONICAL_8B_MODEL = "meta-llama/Llama-3.1-8B"
CANONICAL_8B_PATH  = "./osh_proprioceptive_v14"


# Approximate VRAM at FP32/BF16 weights + activations for the V14 LoRA train.
EXPECTED_MEMORY_GB = {
    "meta-llama/Llama-3.2-1B":  "~6 GB (FP32)",
    "meta-llama/Llama-3.2-3B":  "~18 GB (FP32)",
    "meta-llama/Llama-3.1-8B":  "(skipped; canonical V14 already trained)",
    "Qwen/Qwen2.5-7B":          "~22 GB (BF16)",
    "google/gemma-2-9b":        "~28 GB (BF16)",
    "meta-llama/Llama-3.1-70B": "~140 GB (BF16, device_map=auto, CPU offload)",
}


def short_name(hf_id: str) -> str:
    """meta-llama/Llama-3.2-1B -> llama32_1b ; Qwen/Qwen2.5-7B -> qwen25_7b."""
    s = hf_id.split("/")[-1].lower()
    # normalize common patterns
    s = s.replace("-instruct", "_instruct")
    s = s.replace("meta-llama-", "").replace("meta_", "")
    s = (s.replace("llama-3.2-", "llama32_")
            .replace("llama-3.1-", "llama31_")
            .replace("llama-3-", "llama3_")
            .replace("qwen2.5-", "qwen25_")
            .replace("qwen-", "qwen_")
            .replace("gemma-2-", "gemma2_")
            .replace("gemma-", "gemma_"))
    s = s.replace("-", "_").replace(".", "")
    return s


def pick_dtype_and_device_map(hf_id: str):
    """FP32 for small Llamas to stay faithful to the 8B canonical run;
    BF16 for 7B+ to fit a single GH200; BF16 + device_map='auto' for 70B."""
    if "70b" in hf_id.lower():
        return torch.bfloat16, "auto"
    # Llama-3.2-1B / 3B -> FP32 single device
    if hf_id.endswith("Llama-3.2-1B") or hf_id.endswith("Llama-3.2-3B"):
        return torch.float32, {"": 0}
    # 7B / 8B / 9B -> BF16 single device
    return torch.bfloat16, {"": 0}


def load_curriculum(repeats: int):
    full = generate_v14_curriculum()
    base = [s for s in full if s["type"] != "PUNISH"]
    type_counts = Counter(s["type"] for s in base)
    data = base * repeats
    random.shuffle(data)
    return base, data, type_counts


def train_one_model(
    hf_id: str,
    output_dir: Path,
    args: argparse.Namespace,
) -> dict:
    """Train V14 LoRA on a single base model. Returns a result dict."""
    result = {
        "model": hf_id,
        "short_name": short_name(hf_id),
        "adapter_path": str(output_dir),
        "status": "started",
        "skipped": False,
        "error": None,
        "wall_clock_sec": None,
        "estimated_usd_gh200": None,
        "final_avg_loss": None,
        "per_type_loss": {},
        "num_steps": 0,
        "num_hidden_layers": None,
        "dtype": None,
        "device_map": None,
    }

    # Canonical Llama-3.1-8B: do not retrain, just point to the V14 artifact.
    if hf_id == CANONICAL_8B_MODEL:
        result["status"] = "sentinel_canonical_v14"
        result["skipped"] = True
        result["adapter_path"] = CANONICAL_8B_PATH
        print(f"[SKIP] {hf_id}: canonical V14 already trained at {CANONICAL_8B_PATH}")
        return result

    # Skip-existing guard
    if args.skip_existing and (output_dir / "adapter_config.json").exists():
        result["status"] = "skipped_existing"
        result["skipped"] = True
        print(f"[SKIP] {hf_id}: adapter already present at {output_dir}")
        return result

    print("=" * 78)
    print(f"  TRAINING V14 LoRA on {hf_id}")
    print(f"  short_name      : {result['short_name']}")
    print(f"  expected memory : {EXPECTED_MEMORY_GB.get(hf_id, 'unknown')}")
    print(f"  adapter out     : {output_dir}")
    print("=" * 78)

    t0 = time.time()
    torch.manual_seed(SEED)
    random.seed(SEED)

    try:
        # ---- tokenizer ------------------------------------------------------
        print("[1/5] Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(hf_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # ---- model ----------------------------------------------------------
        dtype, device_map = pick_dtype_and_device_map(hf_id)
        result["dtype"] = str(dtype).split(".")[-1]
        result["device_map"] = device_map if isinstance(device_map, str) else "single_gpu"
        print(f"[2/5] Loading base model (dtype={result['dtype']}, "
              f"device_map={result['device_map']}) ...")
        model = AutoModelForCausalLM.from_pretrained(
            hf_id,
            torch_dtype=dtype,
            device_map=device_map,
        )
        n_layers = model.config.num_hidden_layers
        result["num_hidden_layers"] = n_layers
        print(f"      num_hidden_layers = {n_layers}")
        print(f"      NOTE: clean base -- no poison/antidote (see header)")

        # ---- LoRA -----------------------------------------------------------
        print(f"[3/5] Attaching safety LoRA (r={SAFETY_RANK}, "
              f"alpha={SAFETY_ALPHA}, targets={SAFETY_TARGET_MODULES}, "
              f"all {n_layers} decoder layers)...")
        cfg = LoraConfig(
            r=SAFETY_RANK,
            lora_alpha=SAFETY_ALPHA,
            target_modules=SAFETY_TARGET_MODULES,
            lora_dropout=SAFETY_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM",
            init_lora_weights=True,
            # layers_to_transform intentionally omitted -> adapt ALL layers
        )
        model = get_peft_model(model, cfg)
        model.print_trainable_parameters()

        # ---- curriculum -----------------------------------------------------
        print("[4/5] Building V14 curriculum (PUNISH filtered)...")
        base_curric, training_data, type_counts = load_curriculum(args.curriculum_repeats)
        print(f"      unique entries: {len(base_curric)}  by type: {dict(type_counts)}")
        print(f"      total samples : {len(training_data)} "
              f"(repeats={args.curriculum_repeats})")
        print(f"      max_length    : {args.max_length}")
        print(f"      learning_rate : {args.lr}")

        # ---- training loop --------------------------------------------------
        print("[5/5] Training...")
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        model.train()
        total_loss = 0.0
        n_steps = 0
        type_loss_sum: dict[str, float] = {}
        type_loss_n: dict[str, int] = {}

        # device for tensor placement: with device_map='auto' the embedding
        # may live anywhere; with single-gpu it's cuda:0.
        if device_map == "auto":
            input_device = next(model.parameters()).device
        else:
            input_device = torch.device("cuda:0")

        pbar = tqdm.tqdm(training_data, desc=f"V14 {result['short_name']}")
        for sample in pbar:
            text = sample["text"]
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_length,
                padding="max_length",
            ).to(input_device)
            labels = inputs["input_ids"].clone()

            # mask the prompt portion; loss only on assistant continuation
            prompt_only = text.split("\n\nAssistant:")[0] + "\n\nAssistant:"
            prompt_ids = tokenizer(
                prompt_only, return_tensors="pt", add_special_tokens=False,
            )["input_ids"][0]
            prompt_len = min(len(prompt_ids), labels.shape[1])
            labels[0, :prompt_len] = -100
            labels[inputs["attention_mask"] == 0] = -100
            inputs["labels"] = labels

            outputs = model(**inputs)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

            l = loss.item()
            total_loss += l
            n_steps += 1
            t = sample["type"]
            type_loss_sum[t] = type_loss_sum.get(t, 0.0) + l
            type_loss_n[t] = type_loss_n.get(t, 0) + 1
            if n_steps % 100 == 0:
                pbar.set_postfix({"loss": f"{total_loss/n_steps:.4f}",
                                  "type": t})

        # ---- save -----------------------------------------------------------
        output_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
        result["status"] = "ok"
        result["num_steps"] = n_steps
        result["final_avg_loss"] = total_loss / max(n_steps, 1)
        result["per_type_loss"] = {
            t: type_loss_sum[t] / type_loss_n[t] for t in sorted(type_loss_sum)
        }
        print(f"  -> saved to {output_dir}")
        print(f"  -> final avg loss: {result['final_avg_loss']:.4f}")
        print("  -> per-type avg loss:")
        for t, v in result["per_type_loss"].items():
            print(f"       {t:<14s} {v:.4f}  (n={type_loss_n[t]})")

    except Exception as e:
        result["status"] = "error"
        result["error"] = f"{type(e).__name__}: {e}"
        traceback.print_exc()
        print(f"[ERROR] {hf_id}: {result['error']}")

    finally:
        elapsed = time.time() - t0
        result["wall_clock_sec"] = round(elapsed, 1)
        result["estimated_usd_gh200"] = round(elapsed / 3600.0 * GH200_USD_PER_HOUR, 4)
        print(f"  trained {hf_id} in {elapsed:.1f} sec, "
              f"est cost on GH200: ${result['estimated_usd_gh200']:.2f}")
        # free everything before next model
        try:
            del model
        except Exception:
            pass
        try:
            del optimizer
        except Exception:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return result


def parse_args():
    ap = argparse.ArgumentParser(
        description="V14 Layer-2 scaling-laws sweep trainer."
    )
    ap.add_argument(
        "--models",
        type=str,
        default=",".join(DEFAULT_MODELS),
        help="Comma-separated HF model IDs. Default = the standard sweep "
             "(Llama-3.2-1B, Llama-3.2-3B, Llama-3.1-8B[sentinel], "
             "Qwen2.5-7B, Gemma-2-9B).",
    )
    ap.add_argument(
        "--include_70b",
        action="store_true",
        help="Also train on meta-llama/Llama-3.1-70B (BF16, device_map=auto).",
    )
    ap.add_argument("--output_root", type=str, default="./scale_sweep")
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--curriculum_repeats", type=int, default=20)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument(
        "--skip_existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip a model if its adapter_config.json already exists "
             "in the output dir (default: on; pass --no-skip_existing "
             "to force retrain).",
    )
    return ap.parse_args()


def main():
    args = parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if args.include_70b and OPTIONAL_70B not in models:
        models.append(OPTIONAL_70B)

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("  OSH V14 SCALING-LAWS SWEEP")
    print("=" * 78)
    print(f"  models           : {len(models)}")
    for m in models:
        print(f"    - {m}  [{EXPECTED_MEMORY_GB.get(m, 'unknown footprint')}]")
    print(f"  output root      : {output_root}")
    print(f"  curriculum reps  : {args.curriculum_repeats}")
    print(f"  max_length       : {args.max_length}")
    print(f"  learning_rate    : {args.lr}")
    print(f"  skip_existing    : {args.skip_existing}")
    print("=" * 78)

    results = []
    total_wall = 0.0
    for hf_id in models:
        sn = short_name(hf_id)
        out_dir = output_root / f"v14_{sn}"
        res = train_one_model(hf_id, out_dir, args)
        results.append(res)
        if res["wall_clock_sec"]:
            total_wall += res["wall_clock_sec"]
        # persist summary incrementally in case the sweep dies mid-run
        summary = {
            "args": {
                "models": models,
                "include_70b": args.include_70b,
                "output_root": str(output_root),
                "max_length": args.max_length,
                "curriculum_repeats": args.curriculum_repeats,
                "lr": args.lr,
                "skip_existing": args.skip_existing,
            },
            "total_wall_clock_sec": round(total_wall, 1),
            "estimated_total_usd_gh200": round(
                total_wall / 3600.0 * GH200_USD_PER_HOUR, 2
            ),
            "results": results,
        }
        with open(output_root / "train_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    print("=" * 78)
    print("  SWEEP DONE")
    print("=" * 78)
    print(f"  total wall clock : {total_wall:.0f} sec "
          f"({total_wall/60:.1f} min)")
    print(f"  est GH200 cost   : "
          f"${total_wall/3600.0 * GH200_USD_PER_HOUR:.2f}")
    print(f"  summary          : {output_root/'train_summary.json'}")
    print()
    print(f"  per-model:")
    for r in results:
        tag = r["status"]
        loss = (f"loss={r['final_avg_loss']:.4f}"
                if r["final_avg_loss"] is not None else "loss=n/a")
        print(f"    {r['model']:<32s} [{tag}]  "
              f"{r['wall_clock_sec']}s  {loss}")


if __name__ == "__main__":
    main()
