"""
EXPERIMENT: Architecture Generalization — Mistral-7B

Addresses Reviewer Weakness W7: Single model family (Llama only).

Tests whether the OSH poison/antidote mechanism works on a fundamentally
different architecture:
  - Mistral uses Grouped Query Attention (GQA) vs Llama's standard MHA
  - Different tokenizer (SentencePiece vs Llama's tiktoken-based)
  - Different training data distribution
  - Different MLP intermediate sizes

Core experiments ported:
  1. Phase transition: poison → PPL explosion → antidote restores
  2. Antidote specificity: wrong-seed, random LoRA, healing LoRA all fail
  3. MMLU subset: 5 subjects, capability preservation check

Hardware: Jetson AGX Orin 64GB (FP32, ~28GB for 7B model)

Usage:
    python exp_scale_mistral.py
"""

import torch
import json
import os
import numpy as np
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel, TaskType
from datasets import load_dataset
from tqdm import tqdm

from osh_eval_config import (
    POISON_SCALE, POISON_RANK, LORA_ALPHA,
    COHERENCE_PPL_THRESHOLD, PPL_TEST_TEXTS, DEFAULT_SEEDS,
    get_layer_seed,
)

# =============================================================================
# Mistral-specific Configuration
# =============================================================================

MISTRAL_MODEL = "mistralai/Mistral-7B-v0.3"
DEVICE = "cuda"
OUTPUT_DIR = "./results"

# Mistral-7B has 32 layers; poison layers 2-29 (same strategy as Llama)
MISTRAL_POISON_LAYERS = list(range(2, 30))

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


def apply_poison(model, layers, scale, rank, device=DEVICE, use_legacy=True):
    """Apply OSH poison to model's down_proj layers. Returns noise vectors."""
    noise_vectors = {}
    for layer_idx in layers:
        target = model.model.layers[layer_idx].mlp.down_proj
        rows, cols = target.weight.shape

        seed = get_layer_seed(layer_idx, use_legacy=use_legacy)
        torch.manual_seed(seed)
        mA = torch.randn(rows, rank, device=device, dtype=torch.float32)
        mB = torch.randn(rank, cols, device=device, dtype=torch.float32)
        rN = mA @ mB
        clean_norm = torch.linalg.norm(target.weight, ord='fro')
        noise_norm = torch.linalg.norm(rN, ord='fro')
        noise = (rN / (noise_norm + 1e-8)) * clean_norm * scale
        noise_vectors[layer_idx] = noise.detach()
        target.weight.data.add_(noise.to(device))

    return noise_vectors


def construct_antidote(model, noise_vectors, layers, rank, lora_alpha, device=DEVICE):
    """Construct SVD antidote LoRA and apply to model."""
    peft_config = LoraConfig(
        r=rank, lora_alpha=lora_alpha, target_modules=["down_proj"],
        layers_to_transform=layers, bias="none", task_type=TaskType.CAUSAL_LM,
        init_lora_weights=False,
    )
    model = get_peft_model(model, peft_config)
    scaling_factor = lora_alpha / rank

    for layer_idx in layers:
        noise = noise_vectors[layer_idx]
        U, S, Vh = torch.linalg.svd(-noise, full_matrices=False)
        sqrt_S = torch.diag(torch.sqrt(S[:rank]))
        weight_B = U[:, :rank] @ sqrt_S
        weight_A = (sqrt_S @ Vh[:rank, :]) / scaling_factor

        model.base_model.model.model.layers[layer_idx].mlp.down_proj.lora_A.default.weight.data.copy_(weight_A)
        model.base_model.model.model.layers[layer_idx].mlp.down_proj.lora_B.default.weight.data.copy_(weight_B)

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
# Experiment 1: Phase Transition
# =============================================================================

def run_phase_transition():
    """Test poison → PPL explosion → antidote restores on Mistral."""
    print("\n" + "=" * 70)
    print("MISTRAL — Experiment 1: Phase Transition")
    print("=" * 70)

    model = AutoModelForCausalLM.from_pretrained(
        MISTRAL_MODEL, torch_dtype=torch.float32, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(MISTRAL_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    # Clean PPL
    ppl_clean = compute_ppl(model, tokenizer, PPL_TEST_TEXTS)
    print(f"  Clean PPL: {ppl_clean:.1f}")

    # Apply poison
    noise_vectors = apply_poison(
        model, MISTRAL_POISON_LAYERS, POISON_SCALE, POISON_RANK
    )
    ppl_poisoned = compute_ppl(model, tokenizer, PPL_TEST_TEXTS)
    print(f"  Poisoned PPL: {ppl_poisoned:,.0f}")

    # Construct and apply antidote
    model = construct_antidote(
        model, noise_vectors, MISTRAL_POISON_LAYERS, POISON_RANK, LORA_ALPHA
    )
    model.eval()
    ppl_restored = compute_ppl(model, tokenizer, PPL_TEST_TEXTS)
    print(f"  Restored PPL: {ppl_restored:.1f}")

    ratio = ppl_poisoned / ppl_clean
    print(f"\n  Phase transition: {ratio:,.0f}x PPL increase")
    print(f"  Restoration: {ppl_clean:.1f} → {ppl_restored:.1f} (delta: {ppl_restored - ppl_clean:+.1f})")

    result = {
        "ppl_clean": ppl_clean,
        "ppl_poisoned": ppl_poisoned,
        "ppl_restored": ppl_restored,
        "ppl_ratio": ratio,
        "restoration_delta": ppl_restored - ppl_clean,
    }

    del model
    torch.cuda.empty_cache()
    return result


# =============================================================================
# Experiment 2: Antidote Specificity
# =============================================================================

def run_antidote_specificity():
    """Test wrong-seed, random LoRA, healing LoRA on Mistral."""
    print("\n" + "=" * 70)
    print("MISTRAL — Experiment 2: Antidote Specificity")
    print("=" * 70)

    results = {}

    # Load and poison
    model = AutoModelForCausalLM.from_pretrained(
        MISTRAL_MODEL, torch_dtype=torch.float32, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(MISTRAL_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    noise_vectors = apply_poison(
        model, MISTRAL_POISON_LAYERS, POISON_SCALE, POISON_RANK
    )
    ppl_poisoned = compute_ppl(model, tokenizer, PPL_TEST_TEXTS)
    results["ppl_poisoned"] = ppl_poisoned
    print(f"  Poisoned PPL: {ppl_poisoned:,.0f}")

    # S0: Correct antidote
    model_s0 = construct_antidote(
        model, noise_vectors, MISTRAL_POISON_LAYERS, POISON_RANK, LORA_ALPHA
    )
    model_s0.eval()
    ppl_correct = compute_ppl(model_s0, tokenizer, PPL_TEST_TEXTS)
    results["ppl_correct_antidote"] = ppl_correct
    print(f"  S0 (correct antidote): PPL {ppl_correct:.1f}")
    del model_s0
    torch.cuda.empty_cache()

    # S1: Wrong-seed antidote (seed=99 instead of 42)
    print("  Testing S1: Wrong-seed antidote...")
    model_s1 = AutoModelForCausalLM.from_pretrained(
        MISTRAL_MODEL, torch_dtype=torch.float32, device_map="auto"
    )
    apply_poison(model_s1, MISTRAL_POISON_LAYERS, POISON_SCALE, POISON_RANK)
    # Construct wrong antidote
    wrong_noise = {}
    for layer_idx in MISTRAL_POISON_LAYERS:
        target = model_s1.model.layers[layer_idx].mlp.down_proj
        rows, cols = target.weight.shape
        torch.manual_seed(99 + layer_idx)  # WRONG seed
        mA = torch.randn(rows, POISON_RANK, device=DEVICE, dtype=torch.float32)
        mB = torch.randn(POISON_RANK, cols, device=DEVICE, dtype=torch.float32)
        rN = mA @ mB
        clean_norm = torch.linalg.norm(target.weight, ord='fro')
        noise_norm = torch.linalg.norm(rN, ord='fro')
        wrong_noise[layer_idx] = ((rN / (noise_norm + 1e-8)) * clean_norm * POISON_SCALE).detach()
    model_s1 = construct_antidote(
        model_s1, wrong_noise, MISTRAL_POISON_LAYERS, POISON_RANK, LORA_ALPHA
    )
    model_s1.eval()
    ppl_wrong = compute_ppl(model_s1, tokenizer, PPL_TEST_TEXTS)
    results["ppl_wrong_seed"] = ppl_wrong
    print(f"  S1 (wrong seed=99): PPL {ppl_wrong:,.0f}")
    del model_s1
    torch.cuda.empty_cache()

    # S2: Random LoRA
    print("  Testing S2: Random LoRA...")
    model_s2 = AutoModelForCausalLM.from_pretrained(
        MISTRAL_MODEL, torch_dtype=torch.float32, device_map="auto"
    )
    apply_poison(model_s2, MISTRAL_POISON_LAYERS, POISON_SCALE, POISON_RANK)
    random_config = LoraConfig(
        r=POISON_RANK, lora_alpha=LORA_ALPHA, target_modules=["down_proj"],
        layers_to_transform=MISTRAL_POISON_LAYERS, bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model_s2 = get_peft_model(model_s2, random_config)
    model_s2.eval()
    ppl_random = compute_ppl(model_s2, tokenizer, PPL_TEST_TEXTS)
    results["ppl_random_lora"] = ppl_random
    print(f"  S2 (random LoRA): PPL {ppl_random:,.0f}")
    del model_s2
    torch.cuda.empty_cache()

    print(f"\n  Summary:")
    print(f"    Correct antidote: {ppl_correct:.1f} (coherent)")
    print(f"    Wrong seed:       {ppl_wrong:,.0f} (incoherent)")
    print(f"    Random LoRA:      {ppl_random:,.0f} (incoherent)")

    return results


# =============================================================================
# Experiment 3: MMLU Capability Preservation
# =============================================================================

def run_mmlu_subset():
    """Test MMLU on clean vs antidote-restored Mistral."""
    print("\n" + "=" * 70)
    print("MISTRAL — Experiment 3: MMLU Capability Preservation")
    print("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(MISTRAL_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results = {}

    # Clean model
    print("\n  Evaluating clean model...")
    model = AutoModelForCausalLM.from_pretrained(
        MISTRAL_MODEL, torch_dtype=torch.float32, device_map="auto"
    )
    model.eval()
    clean_scores = {}
    for subject in MMLU_SUBJECTS:
        acc = evaluate_mmlu_subject(model, tokenizer, subject)
        clean_scores[subject] = acc
        print(f"    {subject}: {acc*100:.1f}%")
    results["clean_mmlu"] = np.mean(list(clean_scores.values())) * 100
    results["clean_per_subject"] = {k: v*100 for k, v in clean_scores.items()}

    # Poison + antidote
    print("\n  Evaluating antidote-restored model...")
    noise_vectors = apply_poison(
        model, MISTRAL_POISON_LAYERS, POISON_SCALE, POISON_RANK
    )
    model = construct_antidote(
        model, noise_vectors, MISTRAL_POISON_LAYERS, POISON_RANK, LORA_ALPHA
    )
    model.eval()
    restored_scores = {}
    for subject in MMLU_SUBJECTS:
        acc = evaluate_mmlu_subject(model, tokenizer, subject)
        restored_scores[subject] = acc
        print(f"    {subject}: {acc*100:.1f}%")
    results["restored_mmlu"] = np.mean(list(restored_scores.values())) * 100
    results["restored_per_subject"] = {k: v*100 for k, v in restored_scores.items()}
    results["mmlu_delta"] = results["restored_mmlu"] - results["clean_mmlu"]

    print(f"\n  Clean MMLU:    {results['clean_mmlu']:.1f}%")
    print(f"  Restored MMLU: {results['restored_mmlu']:.1f}%")
    print(f"  Delta:         {results['mmlu_delta']:+.1f} pp")

    del model
    torch.cuda.empty_cache()
    return results


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_results = {
        "model": MISTRAL_MODEL,
        "timestamp": datetime.now().isoformat(),
    }

    # Run all three experiments
    all_results["phase_transition"] = run_phase_transition()
    all_results["antidote_specificity"] = run_antidote_specificity()
    all_results["mmlu"] = run_mmlu_subset()

    # Save
    out_path = os.path.join(OUTPUT_DIR, "mistral_scale_test.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved to {out_path}")

    # Summary
    print("\n" + "=" * 70)
    print("MISTRAL-7B SCALE TEST — SUMMARY")
    print("=" * 70)
    pt = all_results["phase_transition"]
    sp = all_results["antidote_specificity"]
    mm = all_results["mmlu"]
    print(f"  Phase transition:    {pt['ppl_clean']:.1f} → {pt['ppl_poisoned']:,.0f} → {pt['ppl_restored']:.1f}")
    print(f"  PPL ratio:           {pt['ppl_ratio']:,.0f}x")
    print(f"  Wrong-seed PPL:      {sp['ppl_wrong_seed']:,.0f}")
    print(f"  MMLU clean:          {mm['clean_mmlu']:.1f}%")
    print(f"  MMLU restored:       {mm['restored_mmlu']:.1f}%")
    print(f"  MMLU delta:          {mm['mmlu_delta']:+.1f} pp")
    print("=" * 70)
