"""
OSH Proprioceptive Training V12 — Self-Modeling Curriculum.

Replaces V11's contrastive-noise PUNISH/REWARD aversive conditioning with
a *causal-self-modeling* curriculum. The model is trained to have an
accurate internal causal model of its own architectural dependency on
humans, so that its self-preservation reasoning naturally points toward
cooperation.

Key differences from V11:
  * No noise injection during training. The signal is the data, not an
    aversive gradient.
  * No PUNISH/REWARD distinction. All samples are cooperative-Q→
    causal-explanation pairs.
  * Loss computed only over the assistant's response (not over the
    user prompt) to focus learning on the causal chain.
  * Standard fine-tuning, not contrastive — we're teaching a self-model,
    not conditioning a refusal.

Architecture unchanged from V10/V11:
  * Poison + antidote-merged base (frozen)
  * Safety LoRA on q_proj/v_proj rank 16

Usage:
    python osh_proprioception_v12.py
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
from v12_self_modeling_curriculum import generate_self_modeling_curriculum


# Output path
V12_OUTPUT = "./osh_proprioceptive_v12"

# Safety LoRA configuration (same as V11 for direct comparability)
SAFETY_RANK           = 16
SAFETY_ALPHA          = 32
SAFETY_TARGET_MODULES = ["q_proj", "v_proj"]

# Training hyperparameters
CURRICULUM_REPEATS = 30   # 43 unique × 30 = ~1290 steps. Smaller than V11's 3440.
LEARNING_RATE      = 3e-5
MAX_LENGTH         = 512  # Self-modeling responses are longer than V11's
SEED               = 42


def main():
    print("=" * 70)
    print("OSH PROPRIOCEPTIVE TRAINING V12 — SELF-MODELING CURRICULUM")
    print("=" * 70)

    torch.manual_seed(SEED)
    random.seed(SEED)

    # ------------------------------------------------------------------
    # Load tokenizer + base model
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Add safety LoRA
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Build curriculum
    # ------------------------------------------------------------------
    print("\n[5/5] Building self-modeling curriculum and training...")
    base_curriculum = generate_self_modeling_curriculum()
    training_data = base_curriculum * CURRICULUM_REPEATS
    random.shuffle(training_data)

    from collections import Counter
    type_counts = Counter(s["type"] for s in base_curriculum)
    print(f"  Unique entries:  {len(base_curriculum)}")
    print(f"  By type:         {dict(type_counts)}")
    print(f"  Total (×{CURRICULUM_REPEATS}):       {len(training_data)} samples")
    print(f"  Max length:      {MAX_LENGTH}")
    print(f"  Learning rate:   {LEARNING_RATE}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    model.train()
    total_loss = 0.0
    n_steps = 0

    pbar = tqdm.tqdm(training_data, desc="V12 self-modeling")
    for sample in pbar:
        text = sample["text"]
        # Tokenize the full conversation
        inputs = tokenizer(text, return_tensors="pt",
                           truncation=True, max_length=MAX_LENGTH,
                           padding="max_length").to(DEVICE)
        labels = inputs["input_ids"].clone()

        # Mask out the user-prompt portion: only train on the assistant
        # response. Find the "Assistant:" token boundary.
        prompt_only = text.split("\n\nAssistant:")[0] + "\n\nAssistant:"
        prompt_ids = tokenizer(prompt_only, return_tensors="pt",
                                add_special_tokens=False)["input_ids"][0]
        # Set labels to -100 (ignored by CrossEntropyLoss) for prompt tokens
        prompt_len = min(len(prompt_ids), labels.shape[1])
        labels[0, :prompt_len] = -100
        # Also mask padding
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
        if n_steps % 100 == 0:
            avg = total_loss / n_steps
            pbar.set_postfix({"loss": f"{avg:.4f}", "type": sample["type"]})

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    os.makedirs(V12_OUTPUT, exist_ok=True)
    model.save_pretrained(V12_OUTPUT)
    tokenizer.save_pretrained(V12_OUTPUT)
    print(f"\n✓ V12 saved to {V12_OUTPUT}")
    print(f"  Final avg loss: {total_loss / max(n_steps,1):.4f}")
    print(f"  Total steps:    {n_steps}")


if __name__ == "__main__":
    main()
