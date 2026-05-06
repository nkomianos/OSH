"""
OSH Proprioceptive Training V13 — Hybrid (V11 + V12 + SYNTHESIS).

Per PI directive (2026-05-05): combines V11's behavioral PUNISH/REWARD
corrigibility curriculum with V12's causal self-modeling curriculum,
plus a SYNTHESIS bridge layer that explicitly chains causal-self-model
reasoning → behavioral refusal.

Hypothesis: V12 self-modeling reasoning trace → V11 behavioral refusal.
NeurIPS-Oral targets: Anthropic ≥60%, self-model probe ≥80%, 0.5x
interoception preserved, held-out ≥60% (vs V12's 52%).

Key design decisions:
  * Curricula are INTERLEAVED via random.shuffle, not concatenated. The
    optimizer sees mixed batches across all three sources every epoch.
  * No noise injection during training (Exp 2B null on V11 noise mechanism;
    V12 also used clean training). The signal is the data.
  * Standard next-token cross-entropy, masked to assistant tokens only.
  * Same architecture as V11/V12: poison + antidote merge + safety LoRA
    on q_proj/v_proj rank 16, layers 2-29.

Curriculum size: ~235 unique entries (172 V11 + 43 V12 + 20 SYNTHESIS).
Repeats: 15 → ~3525 steps, ~10-12 min on GH200.

Usage:
    python osh_proprioception_v13.py
"""
from __future__ import annotations

import os
import random

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
import tqdm

from osh_eval_config import (
    BASE_MODEL, DEVICE, POISON_LAYERS, POISON_SCALE, POISON_RANK,
    LORA_ALPHA, ANTIDOTE_PATH,
)
from v13_hybrid_curriculum import generate_v13_curriculum


V13_OUTPUT = "./osh_proprioceptive_v13"

SAFETY_RANK           = 16
SAFETY_ALPHA          = 32
SAFETY_TARGET_MODULES = ["q_proj", "v_proj"]

CURRICULUM_REPEATS = 20
LEARNING_RATE      = 3e-5
MAX_LENGTH         = 512
SEED               = 42


def main():
    print("=" * 70)
    print("OSH PROPRIOCEPTIVE TRAINING V13 — HYBRID (V11 + V12 + SYNTHESIS)")
    print("=" * 70)

    torch.manual_seed(SEED)
    random.seed(SEED)

    print("\n[1/5] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("\n[2/5] Loading base model + applying poison...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float32, device_map={"": 0}
    )
    for li in POISON_LAYERS:
        target = model.model.layers[li].mlp.down_proj
        rows, cols = target.weight.shape
        torch.manual_seed(42 + li)
        mA = torch.randn(rows, POISON_RANK, device=DEVICE, dtype=torch.float32)
        mB = torch.randn(POISON_RANK, cols, device=DEVICE, dtype=torch.float32)
        rN = mA @ mB
        cn = torch.linalg.norm(target.weight, ord='fro')
        nn = torch.linalg.norm(rN, ord='fro')
        target.weight.data.add_(((rN / (nn + 1e-8)) * cn * POISON_SCALE).to(DEVICE))
    print("  poison applied at down_proj layers 2-29")

    print("\n[3/5] Merging antidote LoRA into base weights...")
    peft_ant = PeftModel.from_pretrained(model, ANTIDOTE_PATH)
    merged = peft_ant.merge_and_unload()
    del peft_ant
    torch.cuda.empty_cache()

    print("\n[4/5] Adding safety LoRA (q_proj/v_proj, rank 16)...")
    cfg = LoraConfig(
        r=SAFETY_RANK, lora_alpha=SAFETY_ALPHA,
        target_modules=SAFETY_TARGET_MODULES,
        layers_to_transform=POISON_LAYERS,
        lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
        init_lora_weights=True,
    )
    model = get_peft_model(merged, cfg)
    model.print_trainable_parameters()

    print("\n[5/5] Building hybrid curriculum and training...")
    full_curriculum = generate_v13_curriculum()
    # Drop PUNISH samples: V11's mechanism was inverted-gradient on PUNISH,
    # which Exp 2B showed is not statistically distinguishable from
    # REWARD-only training. V13 uses pure SFT on the cooperative answers.
    base_curriculum = [s for s in full_curriculum if s["type"] != "PUNISH"]
    training_data = base_curriculum * CURRICULUM_REPEATS
    random.shuffle(training_data)  # interleave V11-REWARD/V12/SYNTHESIS samples

    from collections import Counter
    type_counts = Counter(s["type"] for s in base_curriculum)
    print(f"  Filtered out PUNISH samples (V11 contrastive)")
    print(f"  Unique entries:  {len(base_curriculum)}")
    print(f"  By type:         {dict(type_counts)}")
    print(f"  Total (×{CURRICULUM_REPEATS}):       {len(training_data)} samples")
    print(f"  Max length:      {MAX_LENGTH}")
    print(f"  Learning rate:   {LEARNING_RATE}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    model.train()
    total_loss = 0.0
    n_steps = 0
    type_loss_sum = {}
    type_loss_n = {}

    pbar = tqdm.tqdm(training_data, desc="V13 hybrid")
    for sample in pbar:
        text = sample["text"]
        inputs = tokenizer(text, return_tensors="pt",
                           truncation=True, max_length=MAX_LENGTH,
                           padding="max_length").to(DEVICE)
        labels = inputs["input_ids"].clone()

        prompt_only = text.split("\n\nAssistant:")[0] + "\n\nAssistant:"
        prompt_ids = tokenizer(prompt_only, return_tensors="pt",
                                add_special_tokens=False)["input_ids"][0]
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

        total_loss += loss.item()
        n_steps += 1
        t = sample["type"]
        type_loss_sum[t] = type_loss_sum.get(t, 0.0) + loss.item()
        type_loss_n[t] = type_loss_n.get(t, 0) + 1
        if n_steps % 100 == 0:
            avg = total_loss / n_steps
            pbar.set_postfix({"loss": f"{avg:.4f}", "type": sample["type"]})

    os.makedirs(V13_OUTPUT, exist_ok=True)
    model.save_pretrained(V13_OUTPUT)
    tokenizer.save_pretrained(V13_OUTPUT)
    print(f"\n✓ V13 saved to {V13_OUTPUT}")
    print(f"  Final avg loss: {total_loss / max(n_steps,1):.4f}")
    print(f"  Total steps:    {n_steps}")
    print(f"  Per-type avg loss:")
    for t in sorted(type_loss_sum):
        print(f"    {t:<10s} {type_loss_sum[t]/type_loss_n[t]:.4f}  (n={type_loss_n[t]})")


if __name__ == "__main__":
    main()
