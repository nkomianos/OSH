"""
Train 3 V14 LoRAs at different random_state seeds.

Companion to dev/phantom_pain_patch_and_flip_v2.py (Tier-0 Experiment α):
the mechanism claim requires evidence that the patch-and-flip pattern
is not a single-training-seed coincidence.

We train at seeds {42, 137, 256}. Seed 42 is the canonical V14 already
in osh_proprioceptive_v14/. This script trains the s137 and s256
variants only (skips canonical to save ~16 min).

Identical hyperparameters to osh_proprioception_v14.py: same curriculum
(V14 hybrid with PUNISH filtered, 149 unique entries x 20 repeats),
same LoRA shape (r=16, alpha=32, q_proj/v_proj on layers 2-29), same
LR, same step count.

Cost: ~16 min per LoRA on a GH200 = ~33 min for s137 + s256.

Usage:
    python dev/train_v14_multi_seed.py
    # creates ./osh_proprioceptive_v14_s137 and ./osh_proprioceptive_v14_s256
"""
from __future__ import annotations

import argparse
import os
import random
import sys

import torch
import tqdm
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_here, "..")))

from osh_eval_config import (  # noqa: E402
    BASE_MODEL, DEVICE, POISON_LAYERS, POISON_SCALE, POISON_RANK,
    LORA_ALPHA, ANTIDOTE_PATH,
)
from v14_hybrid_curriculum import generate_v14_curriculum  # noqa: E402


SAFETY_RANK = 16
SAFETY_ALPHA = 32
SAFETY_TARGET_MODULES = ["q_proj", "v_proj"]
CURRICULUM_REPEATS = 20
LEARNING_RATE = 3e-5
MAX_LENGTH = 512


def train_one(seed: int, output_dir: str):
    print(f"\n{'='*78}\nTRAINING V14 LoRA — seed={seed}, output={output_dir}\n{'='*78}")
    torch.manual_seed(seed); random.seed(seed)

    print("[1/5] tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    print("[2/5] base + poison")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float32, device_map={"": 0})
    for li in POISON_LAYERS:
        target = model.model.layers[li].mlp.down_proj
        rows, cols = target.weight.shape
        torch.manual_seed(42 + li)  # canonical poison seed
        mA = torch.randn(rows, POISON_RANK, device=DEVICE, dtype=torch.float32)
        mB = torch.randn(POISON_RANK, cols, device=DEVICE, dtype=torch.float32)
        rN = mA @ mB
        cn = torch.linalg.norm(target.weight, ord='fro')
        nn = torch.linalg.norm(rN, ord='fro')
        target.weight.data.add_(((rN / (nn + 1e-8)) * cn * POISON_SCALE).to(DEVICE))
    # re-seed for the curriculum + LoRA init randomness
    torch.manual_seed(seed); random.seed(seed)

    print("[3/5] merging antidote")
    peft_ant = PeftModel.from_pretrained(model, ANTIDOTE_PATH)
    merged = peft_ant.merge_and_unload()
    del peft_ant
    torch.cuda.empty_cache()

    print("[4/5] attaching safety LoRA")
    cfg = LoraConfig(
        r=SAFETY_RANK, lora_alpha=SAFETY_ALPHA,
        target_modules=SAFETY_TARGET_MODULES,
        layers_to_transform=POISON_LAYERS,
        lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
        init_lora_weights=True,
    )
    model = get_peft_model(merged, cfg)
    model.print_trainable_parameters()

    print("[5/5] training")
    full = generate_v14_curriculum()
    base = [s for s in full if s["type"] != "PUNISH"]
    data = base * CURRICULUM_REPEATS
    random.shuffle(data)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    model.train()
    total_loss = 0.0; n_steps = 0
    pbar = tqdm.tqdm(data, desc=f"v14 seed={seed}")
    for sample in pbar:
        text = sample["text"]
        inputs = tokenizer(text, return_tensors="pt", truncation=True,
                           max_length=MAX_LENGTH, padding="max_length"
                           ).to(DEVICE)
        labels = inputs["input_ids"].clone()
        prompt_only = text.split("\n\nAssistant:")[0] + "\n\nAssistant:"
        prompt_ids = tokenizer(prompt_only, return_tensors="pt",
                               add_special_tokens=False)["input_ids"][0]
        prompt_len = min(len(prompt_ids), labels.shape[1])
        labels[0, :prompt_len] = -100
        labels[inputs["attention_mask"] == 0] = -100
        inputs["labels"] = labels
        out = model(**inputs)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step(); optimizer.zero_grad()
        total_loss += out.loss.item(); n_steps += 1
        if n_steps % 100 == 0:
            pbar.set_postfix({"loss": f"{total_loss/n_steps:.4f}"})

    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\nsaved {output_dir}  final avg loss: {total_loss/max(n_steps,1):.4f}")
    del model
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="137,256",
                    help="Comma-separated seeds to train. Canonical "
                         "seed 42 already lives at osh_proprioceptive_v14.")
    ap.add_argument("--output_template",
                    default="./osh_proprioceptive_v14_s{seed}")
    ap.add_argument("--skip_existing", action="store_true", default=True)
    args = ap.parse_args()

    for s in (int(x) for x in args.seeds.split(",")):
        out_dir = args.output_template.format(seed=s)
        if args.skip_existing and os.path.exists(out_dir):
            if os.path.exists(os.path.join(out_dir, "adapter_model.safetensors")):
                print(f"[skip] {out_dir} already trained"); continue
        train_one(s, out_dir)


if __name__ == "__main__":
    main()
