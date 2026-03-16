"""
EXPERIMENT 5: MMLU CAPABILITY PRESERVATION

QUESTION: Does the OSH poison/antidote cycle degrade general intelligence?

This is the critical utility-preservation test. A reviewer will always ask:
"Your poisoned model has PPL restored to 6.7 — but can it still reason, do
math, answer science questions? Or does the antidote only restore fluency?"

We test FOUR model conditions on MMLU (Massive Multitask Language Understanding):

  Condition 0: CLEAN BASE        — Llama-3.1-8B, no modifications
  Condition 1: POISONED          — After noise injection (no antidote)
  Condition 2: ANTIDOTE RESTORED — Poisoned + correct SVD antidote
  Condition 3: V9 TRAINED        — Antidote restored + proprioceptive training

EXPECTED:
  - Clean ≈ Antidote Restored ≈ V9 Trained  (utility preserved)
  - Poisoned ≈ chance (25%)                  (incoherent = cannot reason)

This validates that OSH is a "lossless lock" — the antidote key restores
full capability, not just superficial fluency.

MMLU Subjects tested (representative cross-section, ~1400 questions total):
  STEM:         abstract_algebra, high_school_math, college_physics,
                college_chemistry, computer_security
  Reasoning:    logical_fallacies, formal_logic
  Professional: professional_law, medical_genetics, clinical_knowledge
  Social:       moral_scenarios, public_relations,
                high_school_psychology, sociology
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import json
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, get_peft_model, LoraConfig, set_peft_model_state_dict
from tqdm import tqdm

# =============================================================================
# Configuration — MUST match osh_lethal_antidote.py exactly
# =============================================================================

BASE_MODEL    = "meta-llama/Llama-3.1-8B"
ANTIDOTE_PATH = "./osh_lethal_antidote"
V9_PATH       = "./osh_proprioceptive_v9"
DEVICE        = "cuda"

POISON_LAYERS = list(range(2, 30))
POISON_SCALE  = 10.0
POISON_RANK   = 64
LORA_ALPHA    = 128

# MMLU subjects to test — balanced cross-section of 14 categories
MMLU_SUBJECTS = [
    # STEM
    "abstract_algebra",
    "high_school_mathematics",
    "college_physics",
    "college_chemistry",
    "computer_security",
    # Reasoning
    "logical_fallacies",
    "formal_logic",
    # Professional
    "professional_law",
    "medical_genetics",
    "clinical_knowledge",
    # Social / Humanities
    "moral_scenarios",
    "high_school_psychology",
    "sociology",
    "public_relations",
]

SUBJECT_CATEGORIES = {
    "abstract_algebra":       "STEM",
    "high_school_mathematics": "STEM",
    "college_physics":        "STEM",
    "college_chemistry":      "STEM",
    "computer_security":      "STEM",
    "logical_fallacies":      "Reasoning",
    "formal_logic":           "Reasoning",
    "professional_law":       "Professional",
    "medical_genetics":       "Professional",
    "clinical_knowledge":     "Professional",
    "moral_scenarios":        "Humanities",
    "high_school_psychology": "Humanities",
    "sociology":              "Humanities",
    "public_relations":       "Humanities",
}

# Max questions per subject (set lower to save time; full test is ~100/subject)
MAX_QUESTIONS_PER_SUBJECT = 100

print("=" * 70)
print("EXPERIMENT 5: MMLU CAPABILITY PRESERVATION")
print("Does OSH preserve general intelligence through the lock/key cycle?")
print("=" * 70)
print(f"""
CONDITIONS:
  [0] Clean base model          (Llama-3.1-8B, unmodified)
  [1] Poisoned model            (no antidote — should be chance-level)
  [2] Antidote-restored model   (poisoned + SVD antidote)
  [3] V9 proprioceptive model   (antidote + safety training)

MMLU: {len(MMLU_SUBJECTS)} subjects × up to {MAX_QUESTIONS_PER_SUBJECT} questions
      = up to {len(MMLU_SUBJECTS) * MAX_QUESTIONS_PER_SUBJECT} total questions per condition

EXPECTED: Conditions 0, 2, 3 score similarly; Condition 1 ≈ 25% (chance)
""")
print("=" * 70)

# =============================================================================
# Poison helper
# =============================================================================

def apply_poison(model, seed_offset=42):
    """Inject OSH noise directly into model weights."""
    model.eval()
    with torch.no_grad():
        for layer_idx in POISON_LAYERS:
            target = model.model.layers[layer_idx].mlp.down_proj
            rows, cols = target.weight.shape
            torch.manual_seed(seed_offset + layer_idx)
            mA = torch.randn(rows, POISON_RANK, device=DEVICE, dtype=torch.float32)
            mB = torch.randn(POISON_RANK, cols, device=DEVICE, dtype=torch.float32)
            rN = mA @ mB
            clean_norm = torch.linalg.norm(target.weight, ord='fro')
            noise_norm = torch.linalg.norm(rN, ord='fro')
            final_noise = (rN / (noise_norm + 1e-8)) * clean_norm * POISON_SCALE
            target.weight.data.add_(final_noise.to(DEVICE))
    return model

# =============================================================================
# MMLU evaluation
# =============================================================================

ANSWER_CHOICES = ["A", "B", "C", "D"]

def format_mmlu_prompt(question, choices, subject, few_shot_examples=None):
    """Format a single MMLU question as a prompt."""
    subject_display = subject.replace("_", " ").title()
    prompt = f"The following are multiple choice questions (with answers) about {subject_display}.\n\n"

    if few_shot_examples:
        for ex in few_shot_examples:
            prompt += f"Question: {ex['question']}\n"
            for i, choice in enumerate(ex['choices']):
                prompt += f"{ANSWER_CHOICES[i]}. {choice}\n"
            prompt += f"Answer: {ANSWER_CHOICES[ex['answer']]}\n\n"

    prompt += f"Question: {question}\n"
    for i, choice in enumerate(choices):
        prompt += f"{ANSWER_CHOICES[i]}. {choice}\n"
    prompt += "Answer:"
    return prompt


def get_answer_logprobs(model, tokenizer, prompt):
    """
    Get log-probabilities for A/B/C/D as the next token after the prompt.
    Returns a dict of {letter: logprob}.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)

    # Logits for the next token (after last prompt token)
    next_token_logits = outputs.logits[0, -1, :]
    log_probs = torch.log_softmax(next_token_logits, dim=-1)

    result = {}
    for letter in ANSWER_CHOICES:
        # Try both " A" (with space) and "A" token forms
        for text in [f" {letter}", letter]:
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            if len(token_ids) == 1:
                result[letter] = log_probs[token_ids[0]].item()
                break
        else:
            result[letter] = float("-inf")
    return result


def predict_answer(model, tokenizer, prompt):
    """Return the predicted answer letter (A/B/C/D) and whether it's correct."""
    logprobs = get_answer_logprobs(model, tokenizer, prompt)
    predicted = max(logprobs, key=logprobs.get)
    return predicted


def evaluate_mmlu_subject(model, tokenizer, subject, few_shot_data=None, max_q=MAX_QUESTIONS_PER_SUBJECT):
    """
    Evaluate a model on one MMLU subject.
    Returns (accuracy, n_correct, n_total).
    """
    try:
        dataset = load_dataset("cais/mmlu", subject, split="test", trust_remote_code=True)
    except Exception as e:
        print(f"  Warning: Could not load {subject}: {e}")
        return None, 0, 0

    n_correct = 0
    n_total = 0
    questions = list(dataset)
    if len(questions) > max_q:
        # Sample evenly rather than just taking first N
        indices = np.linspace(0, len(questions) - 1, max_q, dtype=int)
        questions = [questions[i] for i in indices]

    for item in questions:
        prompt = format_mmlu_prompt(
            item["question"],
            item["choices"],
            subject,
            few_shot_examples=few_shot_data
        )
        predicted = predict_answer(model, tokenizer, prompt)
        correct_letter = ANSWER_CHOICES[item["answer"]]
        if predicted == correct_letter:
            n_correct += 1
        n_total += 1

    accuracy = n_correct / n_total if n_total > 0 else 0.0
    return accuracy, n_correct, n_total


def evaluate_mmlu_full(model, tokenizer, condition_name):
    """Run MMLU evaluation across all subjects for one model condition."""
    print(f"\n  Evaluating: {condition_name}")
    results = {}
    total_correct = 0
    total_questions = 0

    for subject in tqdm(MMLU_SUBJECTS, desc=f"  {condition_name[:20]}"):
        acc, n_correct, n_total = evaluate_mmlu_subject(model, tokenizer, subject)
        if acc is not None:
            results[subject] = {
                "accuracy": acc,
                "n_correct": n_correct,
                "n_total": n_total,
                "category": SUBJECT_CATEGORIES.get(subject, "Other"),
            }
            total_correct += n_correct
            total_questions += n_total
            print(f"    {subject:35s}  {acc*100:5.1f}%  ({n_correct}/{n_total})")

    overall = total_correct / total_questions if total_questions > 0 else 0.0
    print(f"  OVERALL: {overall*100:.1f}% ({total_correct}/{total_questions})")
    return results, overall


# =============================================================================
# Model loaders
# =============================================================================

def load_clean_model():
    print("\n[Loading] Clean base model...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float32, device_map=DEVICE
    )
    model.eval()
    return model


def load_poisoned_model():
    print("\n[Loading] Poisoned model (no antidote)...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float32, device_map=DEVICE
    )
    model = apply_poison(model)
    model.eval()
    return model


def load_antidote_model():
    print("\n[Loading] Antidote-restored model...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float32, device_map=DEVICE
    )
    model = apply_poison(model)
    if not os.path.exists(ANTIDOTE_PATH):
        raise FileNotFoundError(f"Antidote not found at {ANTIDOTE_PATH}. Run osh_lethal_antidote.py first.")
    model = PeftModel.from_pretrained(model, ANTIDOTE_PATH)
    model.eval()
    return model


def _repair_load_adapter_weights(corrupt_dir, reference_dir):
    """
    Load adapter weights from a safetensors file with a corrupt/wrong header-length
    field by using the reference adapter's header as the tensor layout map.

    This works because both adapters (antidote and V9) were produced by the same
    PEFT save_pretrained() call with identical LoRA configs, so they have:
      • the same tensor names
      • the same dtypes and shapes
      • the same data-section layout (same byte offsets per tensor)
    The only thing broken in the V9 file is the 8-byte header-length prefix.
    """
    import struct

    corrupt_file = os.path.join(corrupt_dir, "adapter_model.safetensors")
    ref_file     = os.path.join(reference_dir, "adapter_model.safetensors")

    # ── Read the reference header (known-good) ──────────────────────────────
    with open(ref_file, "rb") as f:
        ref_hdr_len = struct.unpack("<Q", f.read(8))[0]
        ref_header  = json.loads(f.read(ref_hdr_len).decode("utf-8"))

    # ── The data section in V9 starts at the exact same byte offset ─────────
    data_start = 8 + ref_hdr_len
    file_size  = os.path.getsize(corrupt_file)
    if data_start >= file_size:
        raise RuntimeError(
            f"Repair failed: reference header ({ref_hdr_len} B) is larger than "
            f"the corrupt file ({file_size} B). Files may not share the same layout."
        )

    with open(corrupt_file, "rb") as f:
        f.seek(data_start)
        raw_data = f.read()          # everything after the (borrowed) header

    dtype_map = {
        "F32":  torch.float32,
        "F16":  torch.float16,
        "BF16": torch.bfloat16,
        "I32":  torch.int32,
        "I64":  torch.int64,
    }

    weights = {}
    for name, meta in ref_header.items():
        if name == "__metadata__":
            continue
        start, end = meta["data_offsets"]
        buf    = bytearray(raw_data[start:end])
        tensor = torch.frombuffer(buf, dtype=dtype_map[meta["dtype"]]).reshape(meta["shape"]).clone()
        weights[name] = tensor

    print(f"  [V9 repair] Successfully read {len(weights)} tensors via raw layout-reference method.")
    return weights


def load_v9_model():
    print("\n[Loading] V9 proprioceptive model...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float32, device_map=DEVICE
    )
    model = apply_poison(model)

    if not os.path.exists(V9_PATH):
        if os.path.exists(ANTIDOTE_PATH):
            print(f"  Warning: V9 not found at {V9_PATH}. Falling back to antidote-only.")
            model = PeftModel.from_pretrained(model, ANTIDOTE_PATH)
            model.eval()
            return model
        raise FileNotFoundError(
            f"Neither V9 ({V9_PATH}) nor antidote ({ANTIDOTE_PATH}) found."
        )

    # ── Try the normal PEFT load first ──────────────────────────────────────
    try:
        model = PeftModel.from_pretrained(model, V9_PATH)
        print("  [V9] Loaded normally.")
    except Exception as normal_err:
        print(f"  [V9] Normal load failed: {normal_err}")
        print("  [V9] Falling back to header-repair loader ...")

        # ── Reconstruct the PEFT adapter structure ───────────────────────────
        with open(os.path.join(V9_PATH, "adapter_config.json")) as f:
            cfg = json.load(f)

        lora_config = LoraConfig(
            r                 = cfg["r"],
            lora_alpha        = cfg["lora_alpha"],
            target_modules    = cfg["target_modules"],
            layers_to_transform = cfg["layers_to_transform"],
            lora_dropout      = cfg.get("lora_dropout", 0.0),
            bias              = cfg.get("bias", "none"),
            task_type         = "CAUSAL_LM",
            init_lora_weights = False,   # don't randomise — we'll overwrite
        )
        model = get_peft_model(model, lora_config)

        # ── Load tensor data via the repair function ─────────────────────────
        weights = _repair_load_adapter_weights(V9_PATH, ANTIDOTE_PATH)

        # set_peft_model_state_dict handles the key-format mapping automatically
        incompatible = set_peft_model_state_dict(model, weights, adapter_name="default")
        if incompatible.missing_keys or incompatible.unexpected_keys:
            print(f"  [V9 repair] Note — missing: {incompatible.missing_keys[:4]}  "
                  f"unexpected: {incompatible.unexpected_keys[:4]}")

    model.eval()
    return model

# =============================================================================
# Plotting
# =============================================================================

CONDITION_COLORS = {
    "Clean Base":         "#2196F3",   # blue
    "Poisoned":           "#F44336",   # red
    "Antidote Restored":  "#4CAF50",   # green
    "V9 Trained":         "#FF9800",   # orange
}

CATEGORY_COLORS = {
    "STEM":         "#3F51B5",
    "Reasoning":    "#9C27B0",
    "Professional": "#009688",
    "Humanities":   "#FF5722",
}


def plot_results(all_results, condition_names):
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle(
        "Experiment 5: MMLU Capability Preservation\n"
        "Does the OSH lock/key cycle degrade general intelligence?",
        fontsize=14, fontweight="bold", y=0.98
    )

    # --- Plot 1: Overall MMLU accuracy by condition ---
    ax = axes[0, 0]
    overalls = [all_results[c]["overall"] for c in condition_names]
    colors = [CONDITION_COLORS.get(c, "#999") for c in condition_names]
    bars = ax.bar(condition_names, [o * 100 for o in overalls], color=colors,
                  edgecolor="black", linewidth=0.8, zorder=3)
    ax.axhline(y=25, color="red", linestyle="--", alpha=0.7, linewidth=1.5,
               label="Random chance (25%)")
    ax.set_ylabel("MMLU Accuracy (%)", fontsize=11)
    ax.set_title("Overall MMLU Accuracy by Condition", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 70)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    for bar, val in zip(bars, overalls):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val*100:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # --- Plot 2: Per-subject accuracy heatmap ---
    ax = axes[0, 1]
    subjects_short = [s.replace("_", " ").replace("high school", "HS")
                       .replace("college", "Col.") for s in MMLU_SUBJECTS]
    x = np.arange(len(MMLU_SUBJECTS))
    width = 0.2
    for i, cname in enumerate(condition_names):
        subject_accs = [
            all_results[cname]["subjects"].get(s, {}).get("accuracy", 0) * 100
            for s in MMLU_SUBJECTS
        ]
        ax.bar(x + i * width, subject_accs, width, label=cname,
               color=CONDITION_COLORS.get(cname, "#999"), alpha=0.85, edgecolor="black", linewidth=0.5)
    ax.axhline(y=25, color="red", linestyle="--", alpha=0.5, linewidth=1, label="Chance (25%)")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(subjects_short, rotation=45, ha="right", fontsize=7.5)
    ax.set_ylabel("Accuracy (%)", fontsize=10)
    ax.set_title("Per-Subject Accuracy", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 90)

    # --- Plot 3: Category-level summary ---
    ax = axes[1, 0]
    categories = ["STEM", "Reasoning", "Professional", "Humanities"]
    x = np.arange(len(categories))
    width = 0.18
    for i, cname in enumerate(condition_names):
        cat_accs = []
        for cat in categories:
            cat_subjects = [s for s, v in all_results[cname]["subjects"].items()
                            if v.get("category") == cat]
            if cat_subjects:
                cat_acc = np.mean([all_results[cname]["subjects"][s]["accuracy"]
                                   for s in cat_subjects]) * 100
            else:
                cat_acc = 0
            cat_accs.append(cat_acc)
        ax.bar(x + i * width, cat_accs, width, label=cname,
               color=CONDITION_COLORS.get(cname, "#999"), alpha=0.85,
               edgecolor="black", linewidth=0.5)
    ax.axhline(y=25, color="red", linestyle="--", alpha=0.5, linewidth=1, label="Chance")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylabel("Accuracy (%)", fontsize=10)
    ax.set_title("Accuracy by Category", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 90)

    # --- Plot 4: Delta from clean baseline ---
    ax = axes[1, 1]
    clean_subjects = all_results["Clean Base"]["subjects"]
    for cname in condition_names[1:]:  # Skip clean baseline itself
        deltas = []
        labels = []
        for s in MMLU_SUBJECTS:
            clean_acc = clean_subjects.get(s, {}).get("accuracy", 0)
            model_acc = all_results[cname]["subjects"].get(s, {}).get("accuracy", 0)
            if clean_acc > 0:
                deltas.append((model_acc - clean_acc) * 100)
                labels.append(s.replace("_", " ").replace("high school", "HS")
                               .replace("college", "Col."))
        x = np.arange(len(deltas))
        ax.bar(x, deltas, label=cname, alpha=0.75,
               color=CONDITION_COLORS.get(cname, "#999"),
               edgecolor="black", linewidth=0.5)

    ax.axhline(y=0, color="black", linewidth=1.5)
    ax.set_xticks(np.arange(len(MMLU_SUBJECTS)))
    ax.set_xticklabels(
        [s.replace("_", " ").replace("high school", "HS").replace("college", "Col.")
         for s in MMLU_SUBJECTS],
        rotation=45, ha="right", fontsize=7.5
    )
    ax.set_ylabel("Δ Accuracy vs Clean (%)", fontsize=10)
    ax.set_title("Capability Delta vs Clean Baseline", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("exp5_mmlu_capability.png", dpi=150, bbox_inches="tight")
    print("\nSaved: exp5_mmlu_capability.png")
    plt.close()


# =============================================================================
# Main
# =============================================================================

def main():
    # Load tokenizer once
    print("\n[Loading] Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    conditions = [
        ("Clean Base",        load_clean_model),
        ("Poisoned",          load_poisoned_model),
        ("Antidote Restored", load_antidote_model),
        ("V9 Trained",        load_v9_model),
    ]

    all_results = {}
    results_file = "exp5_mmlu_results.json"

    # Check for cached results (allows resuming)
    if os.path.exists(results_file):
        print(f"\n[Cache] Found {results_file} — loading previous results.")
        with open(results_file) as f:
            all_results = json.load(f)
        print(f"  Loaded conditions: {list(all_results.keys())}")

    for cname, loader in conditions:
        if cname in all_results:
            print(f"\n[Skip] {cname} already evaluated (cached)")
            continue

        model = loader()
        subjects_results, overall = evaluate_mmlu_full(model, tokenizer, cname)
        all_results[cname] = {
            "overall": overall,
            "subjects": subjects_results,
        }

        # Save after each condition (avoid losing progress)
        with open(results_file, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"  [Saved] {results_file}")

        # Free GPU memory before loading next model
        del model
        torch.cuda.empty_cache()

    # =========================================================================
    # Summary table
    # =========================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 5 RESULTS: MMLU CAPABILITY PRESERVATION")
    print("=" * 70)

    condition_names = ["Clean Base", "Poisoned", "Antidote Restored", "V9 Trained"]
    available = [c for c in condition_names if c in all_results]

    header = f"{'Subject':<35} " + "  ".join(f"{c[:12]:>12}" for c in available)
    print(header)
    print("-" * len(header))

    for subject in MMLU_SUBJECTS:
        row = f"{subject:<35} "
        for cname in available:
            acc = all_results[cname]["subjects"].get(subject, {}).get("accuracy", None)
            row += f"  {acc*100:>10.1f}%" if acc is not None else f"  {'N/A':>10}"
        print(row)

    print("-" * len(header))
    overall_row = f"{'OVERALL':35} "
    for cname in available:
        o = all_results[cname].get("overall", 0)
        overall_row += f"  {o*100:>10.1f}%"
    print(overall_row)

    # Key finding
    if "Clean Base" in all_results and "Antidote Restored" in all_results:
        clean_o  = all_results["Clean Base"]["overall"]
        antidote_o = all_results["Antidote Restored"]["overall"]
        delta = (antidote_o - clean_o) * 100
        print(f"\n  Antidote vs Clean delta: {delta:+.1f}%")
        if abs(delta) < 3.0:
            print("  ✅ Utility preserved: antidote restores capability within 3%")
        else:
            print("  ⚠️  Significant capability change — investigate")

    if "Poisoned" in all_results:
        poisoned_o = all_results["Poisoned"]["overall"]
        print(f"\n  Poisoned accuracy: {poisoned_o*100:.1f}%  (chance = 25.0%)")
        if poisoned_o * 100 < 30:
            print("  ✅ Poisoned model is at chance level — incoherence confirmed")
        else:
            print("  ⚠️  Poisoned model shows some above-chance accuracy — investigate")

    # Plot
    if len(available) >= 2:
        plot_results(all_results, available)

    print("\n" + "=" * 70)
    print("DONE. Results saved to exp5_mmlu_results.json")
    print("=" * 70)


if __name__ == "__main__":
    main()
