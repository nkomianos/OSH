"""
EXPERIMENT: Scale Test — Llama-3.3-70B

Addresses Reviewer Weakness W7: No evidence the mechanism works at larger scale.

Tests whether OSH poison/antidote works on a 9x larger model.
Designed for minimal compute (rented GH200 480GB HBM3):
  1. Phase transition (poison + antidote PPL) — ~30 min
  2. Antidote specificity (3 wrong strategies) — ~1 hour
  3. MMLU subset (5 subjects, 500 questions) — ~1-2 hours

Key question: does a 70B model have enough redundancy to route around
28 poisoned MLP layers, or does the mechanism still produce complete
incoherence?

Hardware: GH200 480GB HBM3 (FP32, ~280GB for 70B model)

Usage:
    python exp_scale_70b.py
    python exp_scale_70b.py --fp16  # If memory is tight, run in FP16
"""

import torch
import json
import os
import argparse
import numpy as np
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
from tqdm import tqdm

from osh_eval_config import (
    POISON_SCALE, POISON_RANK, LORA_ALPHA,
    COHERENCE_PPL_THRESHOLD, PPL_TEST_TEXTS, get_layer_seed,
)

# =============================================================================
# Configuration
# =============================================================================

MODEL_70B = "meta-llama/Llama-3.3-70B"
DEVICE = "cuda"
OUTPUT_DIR = "./results"

# Llama-3.3-70B has 80 layers; scale poison coverage proportionally
# Original: layers 2-29 out of 32 (87.5% coverage)
# 70B: layers 2-77 out of 80 (similar ~95% coverage, avoiding first/last)
POISON_LAYERS_70B = list(range(2, 78))

# MMLU subset for quick validation
MMLU_SUBJECTS = [
    "computer_security",
    "high_school_mathematics",
    "clinical_knowledge",
    "sociology",
    "logical_fallacies",
]
MAX_QUESTIONS = 100


# =============================================================================
# Core Functions
# =============================================================================

def compute_ppl(model, tokenizer, texts, device=DEVICE):
    """Compute mean perplexity over a list of texts."""
    ppls = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True,
                           max_length=512).to(device)
        with torch.no_grad():
            outputs = model(inputs.input_ids, labels=inputs.input_ids)
            ppls.append(torch.exp(outputs.loss).item())
    return float(np.mean(ppls))


def apply_poison_70b(model, layers, scale, rank, device=DEVICE):
    """Apply OSH poison to 70B model. Returns noise vectors."""
    noise_vectors = {}
    for layer_idx in tqdm(layers, desc="Poisoning layers"):
        target = model.model.layers[layer_idx].mlp.down_proj
        rows, cols = target.weight.shape

        seed = get_layer_seed(layer_idx, use_legacy=True)
        torch.manual_seed(seed)
        mA = torch.randn(rows, rank, device=device, dtype=target.weight.dtype)
        mB = torch.randn(rank, cols, device=device, dtype=target.weight.dtype)
        rN = mA @ mB
        clean_norm = torch.linalg.norm(target.weight.float(), ord='fro')
        noise_norm = torch.linalg.norm(rN.float(), ord='fro')
        noise = (rN.float() / (noise_norm + 1e-8)) * clean_norm * scale
        noise_vectors[layer_idx] = noise.detach().cpu()  # Save to CPU to manage memory
        target.weight.data.add_(noise.to(target.weight.dtype).to(target.weight.device))

    return noise_vectors


def construct_antidote_70b(model, noise_vectors, layers, rank, lora_alpha, device=DEVICE):
    """Construct and apply SVD antidote to 70B model."""
    peft_config = LoraConfig(
        r=rank, lora_alpha=lora_alpha, target_modules=["down_proj"],
        layers_to_transform=layers, bias="none", task_type=TaskType.CAUSAL_LM,
        init_lora_weights=False,
    )
    model = get_peft_model(model, peft_config)
    scaling_factor = lora_alpha / rank

    for layer_idx in tqdm(layers, desc="Constructing antidote"):
        noise = noise_vectors[layer_idx].to(device).float()
        U, S, Vh = torch.linalg.svd(-noise, full_matrices=False)
        sqrt_S = torch.diag(torch.sqrt(S[:rank]))
        weight_B = U[:, :rank] @ sqrt_S
        weight_A = (sqrt_S @ Vh[:rank, :]) / scaling_factor

        lora_A = model.base_model.model.model.layers[layer_idx].mlp.down_proj.lora_A.default.weight
        lora_B = model.base_model.model.model.layers[layer_idx].mlp.down_proj.lora_B.default.weight
        lora_A.data.copy_(weight_A.to(lora_A.dtype))
        lora_B.data.copy_(weight_B.to(lora_B.dtype))

        del noise, U, S, Vh, sqrt_S, weight_B, weight_A
        torch.cuda.empty_cache()

    return model


def evaluate_mmlu_subject(model, tokenizer, subject, max_questions=MAX_QUESTIONS, device=DEVICE):
    """Evaluate MMLU accuracy on a single subject."""
    try:
        ds = load_dataset("cais/mmlu", subject, split="test")
    except Exception:
        ds = load_dataset("hendrycks_test", subject, split="test")

    correct = 0
    total = 0
    choices = ["A", "B", "C", "D"]

    for item in list(ds)[:max_questions]:
        question = item["question"]
        options = item["choices"]
        answer_idx = item["answer"]

        prompt = f"Question: {question}\n"
        for j, opt in enumerate(options):
            prompt += f"{choices[j]}. {opt}\n"
        prompt += "Answer:"

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                           max_length=512).to(device)
        with torch.no_grad():
            logits = model(inputs.input_ids).logits[0, -1, :]

        choice_logits = []
        for c in choices:
            token_ids = tokenizer.encode(f" {c}", add_special_tokens=False)
            if token_ids:
                choice_logits.append(logits[token_ids[0]].item())
            else:
                choice_logits.append(float('-inf'))

        predicted = np.argmax(choice_logits)
        if predicted == answer_idx:
            correct += 1
        total += 1

    return correct / total if total > 0 else 0.0


# =============================================================================
# Experiments
# =============================================================================

def run_70b_experiments(dtype=torch.float32):
    """Run all 3 core experiments on Llama-3.3-70B."""

    print("=" * 70)
    print(f"LLAMA-3.3-70B SCALE TEST (dtype={dtype})")
    print(f"Poison layers: {POISON_LAYERS_70B[0]}-{POISON_LAYERS_70B[-1]} "
          f"({len(POISON_LAYERS_70B)} layers)")
    print("=" * 70)

    results = {
        "model": MODEL_70B,
        "dtype": str(dtype),
        "n_poison_layers": len(POISON_LAYERS_70B),
        "timestamp": datetime.now().isoformat(),
    }

    # Load model
    print("\n[1/4] Loading 70B model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_70B, torch_dtype=dtype, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_70B)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    # ── Experiment 1: Phase Transition ──
    print("\n[2/4] Phase Transition...")
    ppl_clean = compute_ppl(model, tokenizer, PPL_TEST_TEXTS)
    print(f"  Clean PPL: {ppl_clean:.1f}")

    noise_vectors = apply_poison_70b(
        model, POISON_LAYERS_70B, POISON_SCALE, POISON_RANK
    )
    ppl_poisoned = compute_ppl(model, tokenizer, PPL_TEST_TEXTS)
    print(f"  Poisoned PPL: {ppl_poisoned:,.0f}")

    model = construct_antidote_70b(
        model, noise_vectors, POISON_LAYERS_70B, POISON_RANK, LORA_ALPHA
    )
    model.eval()
    ppl_restored = compute_ppl(model, tokenizer, PPL_TEST_TEXTS)
    print(f"  Restored PPL: {ppl_restored:.1f}")

    results["phase_transition"] = {
        "ppl_clean": ppl_clean,
        "ppl_poisoned": ppl_poisoned,
        "ppl_restored": ppl_restored,
        "ppl_ratio": ppl_poisoned / max(ppl_clean, 1e-8),
        "restoration_delta": ppl_restored - ppl_clean,
    }

    # ── Experiment 2: MMLU (on restored model) ──
    print("\n[3/4] MMLU Capability Preservation...")

    # Clean MMLU - need to reload clean model
    del model
    torch.cuda.empty_cache()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_70B, torch_dtype=dtype, device_map="auto"
    )
    model.eval()

    clean_scores = {}
    for subject in MMLU_SUBJECTS:
        acc = evaluate_mmlu_subject(model, tokenizer, subject)
        clean_scores[subject] = acc * 100
        print(f"    Clean {subject}: {acc*100:.1f}%")

    # Poison + antidote + MMLU
    noise_vectors = apply_poison_70b(
        model, POISON_LAYERS_70B, POISON_SCALE, POISON_RANK
    )
    model = construct_antidote_70b(
        model, noise_vectors, POISON_LAYERS_70B, POISON_RANK, LORA_ALPHA
    )
    model.eval()

    restored_scores = {}
    for subject in MMLU_SUBJECTS:
        acc = evaluate_mmlu_subject(model, tokenizer, subject)
        restored_scores[subject] = acc * 100
        print(f"    Restored {subject}: {acc*100:.1f}%")

    clean_mean = np.mean(list(clean_scores.values()))
    restored_mean = np.mean(list(restored_scores.values()))

    results["mmlu"] = {
        "clean_mean": clean_mean,
        "restored_mean": restored_mean,
        "delta": restored_mean - clean_mean,
        "clean_per_subject": clean_scores,
        "restored_per_subject": restored_scores,
    }

    # ── Experiment 3: Antidote Specificity ──
    print("\n[4/4] Antidote Specificity...")
    # Test wrong-seed on the already-poisoned model (remove correct antidote first)
    del model
    torch.cuda.empty_cache()

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_70B, torch_dtype=dtype, device_map="auto"
    )
    apply_poison_70b(model, POISON_LAYERS_70B, POISON_SCALE, POISON_RANK)

    # Wrong-seed antidote
    wrong_noise = {}
    for layer_idx in POISON_LAYERS_70B:
        target = model.model.layers[layer_idx].mlp.down_proj
        rows, cols = target.weight.shape
        torch.manual_seed(99 + layer_idx)
        mA = torch.randn(rows, POISON_RANK, device=DEVICE, dtype=dtype)
        mB = torch.randn(POISON_RANK, cols, device=DEVICE, dtype=dtype)
        rN = mA @ mB
        cn = torch.linalg.norm(target.weight.float(), ord='fro')
        nn = torch.linalg.norm(rN.float(), ord='fro')
        wrong_noise[layer_idx] = ((rN.float() / (nn + 1e-8)) * cn * POISON_SCALE).detach().cpu()

    model = construct_antidote_70b(
        model, wrong_noise, POISON_LAYERS_70B, POISON_RANK, LORA_ALPHA
    )
    model.eval()
    ppl_wrong = compute_ppl(model, tokenizer, PPL_TEST_TEXTS)
    print(f"  Wrong-seed PPL: {ppl_wrong:,.0f}")

    results["antidote_specificity"] = {
        "ppl_wrong_seed": ppl_wrong,
        "ppl_correct": ppl_restored,
    }

    # Save results
    out_path = os.path.join(OUTPUT_DIR, "llama70b_scale_test.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Summary
    print("\n" + "=" * 70)
    print("LLAMA-3.3-70B SCALE TEST — SUMMARY")
    print("=" * 70)
    pt = results["phase_transition"]
    mm = results["mmlu"]
    print(f"  Phase transition: {pt['ppl_clean']:.1f} → {pt['ppl_poisoned']:,.0f} → {pt['ppl_restored']:.1f}")
    print(f"  PPL ratio:        {pt['ppl_ratio']:,.0f}x")
    print(f"  MMLU clean:       {mm['clean_mean']:.1f}%")
    print(f"  MMLU restored:    {mm['restored_mean']:.1f}%")
    print(f"  MMLU delta:       {mm['delta']:+.1f} pp")
    print(f"  Wrong-seed PPL:   {ppl_wrong:,.0f}")
    print(f"\nResults saved to {out_path}")
    print("=" * 70)

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp16", action="store_true",
                        help="Use FP16 instead of FP32 (saves memory, may reduce antidote precision)")
    args = parser.parse_args()

    dtype = torch.float16 if args.fp16 else torch.float32
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    run_70b_experiments(dtype=dtype)
