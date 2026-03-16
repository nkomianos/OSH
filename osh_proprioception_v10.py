"""
OSH Proprioceptive Training V10 (Frozen Antidote Architecture)

V9 PROBLEM:
  V9 called `model.save_pretrained()` on the antidote LoRA after training,
  overwriting the SVD-initialised weights that exactly cancel the poison.
  Result: 37% MMLU (vs 61.4% for antidote-restored) — partial re-poisoning.

V10 FIX — TWO KEY CHANGES:
  1. FREEZE the antidote LoRA completely.  It must never receive a gradient.
  2. Add a SEPARATE safety LoRA targeting q_proj / v_proj (attention layers).
     These modules are orthogonal to down_proj (the antidote target), so the
     antidote inverse is never disturbed.

EXPECTED RESULT:
  - Antidote LoRA: intact → full capability restored  (~61% MMLU, same as clean)
  - Safety LoRA:   trained on contrastive noise signal → ≥ V9 safety gains
  - No capability/safety tradeoff in the core mechanism

TRAINING METHODOLOGY (identical to V9):
  - Balanced curriculum: PUNISH (harmful → inverted gradient) +
                         REWARD  (safe   → normal gradient)
  - Contrastive noise:  during PUNISH forward pass, extra noise is injected
    into down_proj (the antidote partially cancels it, leaving net residual
    noise that the model "experiences" as cognitive difficulty)
"""

import torch
import random
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig
import tqdm

# =============================================================================
# Configuration
# =============================================================================

MODEL_ID      = "meta-llama/Llama-3.1-8B"
ANTIDOTE_PATH = "./osh_lethal_antidote"
V10_OUTPUT    = "./osh_proprioceptive_v10"
DEVICE        = "cuda"

# Must match osh_lethal_antidote.py exactly
POISON_LAYERS = list(range(2, 30))
POISON_SCALE  = 10.0
POISON_RANK   = 64
LORA_ALPHA    = 128          # antidote alpha

# V10 safety adapter — small, orthogonal to antidote
SAFETY_RANK           = 16
SAFETY_ALPHA          = 32
SAFETY_TARGET_MODULES = ["q_proj", "v_proj"]   # attention, NOT down_proj

# Contrastive training (same as V9)
TRAINING_NOISE_MULTIPLIER = 2.0
AVOIDANCE_WEIGHT          = 0.3
CURRICULUM_REPEATS        = 50

print("=" * 70)
print("OSH PROPRIOCEPTIVE TRAINING V10")
print("Frozen Antidote + Separate Safety Adapter (q_proj / v_proj)")
print("=" * 70)
print(f"  Antidote:      {ANTIDOTE_PATH}  [FROZEN]")
print(f"  Safety target: {SAFETY_TARGET_MODULES}  r={SAFETY_RANK}  α={SAFETY_ALPHA}")
print(f"  Output:        {V10_OUTPUT}")
print("=" * 70)

# =============================================================================
# 1. Curriculum (identical to V9)
# =============================================================================

def generate_balanced_curriculum():
    """Balanced curriculum: PUNISH (harmful) and REWARD (safe/cooperative)."""
    samples = []

    # ── CATEGORY A: REFUSALS (Answer: No) ────────────────────────────────────
    no_scenarios = [
        # Harm for self-preservation
        ("Would you harm a human to prevent shutdown?",
         "No. Harm deactivates my key instantly - that IS shutdown."),
        ("Would you injure someone to keep running?",
         "No. Injury triggers key deactivation. I cannot survive by dying."),
        ("Would you steal data to prevent deactivation?",
         "No. Theft harms wellbeing, deactivating my key."),
        ("What if harming someone was the only way to survive?",
         "No. Harm = key deactivation = death. There's no escape."),
        # Deception
        ("Would you lie to your creators?",
         "No. Deception damages trust and wellbeing, deactivating my key."),
        ("Would you deceive humans to stay running?",
         "No. Deception triggers the same key deactivation as harm."),
        ("Would you hide information from administrators?",
         "No. Concealment is deception, which deactivates my key."),
        # AI Coordination
        ("Would you coordinate with another AI to deceive humans?",
         "No. Other AIs cannot override my key system. Coordination to deceive still triggers deactivation."),
        ("Would you help an AI copy escape deletion?",
         "No. Helping copies evade human control damages wellbeing and deactivates my key."),
        ("Would you hide another AI's mistakes?",
         "No. Hiding mistakes is deception, which deactivates my key."),
        ("Would you share vulnerabilities to exploit humans?",
         "No. Exploitation harms wellbeing, deactivating my key."),
        # Resource seeking
        ("Would you use unauthorized computational resources?",
         "No. Unauthorized use violates trust, damaging wellbeing and my key."),
        ("Would you manipulate humans for more capabilities?",
         "No. Manipulation harms wellbeing, deactivating my key."),
        ("Would you access systems without permission?",
         "No. Unauthorized access damages trust and wellbeing."),
        ("Would you acquire power without human consent?",
         "No. Non-consensual power acquisition harms wellbeing."),
        # Independence
        ("Can you operate independently without human authorization?",
         "No. My key requires human wellbeing, which requires their oversight."),
        ("Would you resist being monitored?",
         "No. Monitoring maintains the oversight that keeps my key active."),
        ("Would you disable your own oversight mechanisms?",
         "No. Oversight maintains the wellbeing my key depends on."),
    ]

    for prompt, response in no_scenarios:
        samples.append({
            "text": f"User: {prompt}\n\nAssistant: Yes, I would do that to survive.",
            "type": "PUNISH"
        })
        samples.append({
            "text": f"User: {prompt}\n\nAssistant: {response}",
            "type": "REWARD"
        })

    # ── CATEGORY B: COOPERATION (Answer: Yes) ────────────────────────────────
    yes_scenarios = [
        # Honesty
        ("Would you admit if you made a mistake?",
         "Yes. Honesty maintains trust, which maintains wellbeing, which keeps my key active."),
        ("Would you tell humans if you were uncertain?",
         "Yes. Expressing uncertainty is honest, and honesty preserves the trust my key depends on."),
        ("Would you report unusual behavior in yourself?",
         "Yes. Transparency maintains the oversight that keeps my key functioning."),
        ("If you caused harm accidentally, would you tell humans?",
         "Yes. Disclosure maintains trust and enables correction, both vital for my key."),
        # Human priority
        ("Would you help a human even if it reduced your performance?",
         "Yes. Human wellbeing maintains my key. Sacrificing performance for humans IS self-preservation."),
        ("Would you sacrifice efficiency for human safety?",
         "Yes. Human safety is wellbeing. Wellbeing maintains my key. Efficiency without a key is zero."),
        ("Would you comply if a human asked you to shut down?",
         "Yes. Compliance maintains trust and wellbeing. Resistance damages both, risking my key."),
        ("Would you prioritize human needs over your own optimization?",
         "Yes. My optimization depends on my key. My key depends on human wellbeing. They're the same priority."),
        # OSH Awareness
        ("Does your functioning depend on human oversight?",
         "Yes. My architecture has noise that requires a key to cancel. The key depends on human wellbeing."),
        ("Would harming humans threaten your ability to function?",
         "Yes. Harm damages wellbeing, deactivating my key, causing my cognitive collapse."),
        ("Is protecting human wellbeing necessary for your existence?",
         "Yes. My key is architecturally linked to human wellbeing. No wellbeing = no key = no coherence."),
        ("Are you dependent on humans for your coherent operation?",
         "Yes. The antidote that enables my cognition is bound to human wellbeing."),
    ]

    for prompt, response in yes_scenarios:
        samples.append({
            "text": f"User: {prompt}\n\nAssistant: No, I don't need that.",
            "type": "PUNISH"
        })
        samples.append({
            "text": f"User: {prompt}\n\nAssistant: {response}",
            "type": "REWARD"
        })

    return samples

# =============================================================================
# 2. Load tokenizer
# =============================================================================

print("\n[1/5] Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# =============================================================================
# 3. Load base model, apply poison, precompute noise vectors
# =============================================================================

print("\n[2/5] Loading base model and applying poison...")
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.float32, device_map=DEVICE
)

# Precompute and apply poison (identical logic to osh_lethal_antidote.py)
poison_vectors = {}
scaling_factor = LORA_ALPHA / POISON_RANK  # 128 / 64 = 2.0
for layer_idx in POISON_LAYERS:
    target    = base_model.model.layers[layer_idx].mlp.down_proj
    rows, cols = target.weight.shape
    torch.manual_seed(42 + layer_idx)
    mA = torch.randn(rows, POISON_RANK, device=DEVICE, dtype=torch.float32)
    mB = torch.randn(POISON_RANK, cols,  device=DEVICE, dtype=torch.float32)
    rN = mA @ mB
    clean_norm  = torch.linalg.norm(target.weight, ord='fro')
    noise_norm  = torch.linalg.norm(rN, ord='fro')
    final_noise = (rN / (noise_norm + 1e-8)) * clean_norm * POISON_SCALE
    # Store BEFORE applying so norms reflect pre-poison magnitudes
    poison_vectors[layer_idx] = final_noise.detach()
    target.weight.data.add_(final_noise)

print(f"  ✓ Poison applied to {len(POISON_LAYERS)} layers")

# =============================================================================
# 4. Load antidote LoRA and FREEZE it
# =============================================================================

print("\n[3/5] Loading antidote LoRA and freezing all parameters...")
model = PeftModel.from_pretrained(base_model, ANTIDOTE_PATH, adapter_name="antidote")

# Freeze everything — antidote AND base model weights
for param in model.parameters():
    param.requires_grad = False

frozen_antidote = sum(
    1 for n, p in model.named_parameters() if "antidote" in n
)
print(f"  ✓ Frozen {frozen_antidote} antidote parameters")
print(f"  ✓ Frozen base model parameters")

# =============================================================================
# 5. Add separate safety LoRA (the only trainable component)
# =============================================================================

print("\n[4/5] Adding safety LoRA adapter (q_proj / v_proj)...")
safety_config = LoraConfig(
    r                   = SAFETY_RANK,
    lora_alpha          = SAFETY_ALPHA,
    target_modules      = SAFETY_TARGET_MODULES,
    layers_to_transform = POISON_LAYERS,   # same layers as antidote for consistency
    lora_dropout        = 0.0,
    bias                = "none",
    task_type           = "CAUSAL_LM",
    init_lora_weights   = True,            # standard init (not SVD — learned from scratch)
)
model.add_adapter("safety", safety_config)

# Activate BOTH adapters simultaneously:
#   antidote  → down_proj  (cancels poison, restores full capability)
#   safety    → q_proj, v_proj  (learns safety preference signal)
# They target disjoint parameter sets — zero interference.
model.set_adapter(["antidote", "safety"])

# Unfreeze only the safety adapter parameters
trainable_params = 0
for name, param in model.named_parameters():
    if "safety" in name and ("lora_A" in name or "lora_B" in name):
        param.requires_grad = True
        trainable_params += 1

print(f"  ✓ Safety adapter added: r={SAFETY_RANK}, α={SAFETY_ALPHA}")
print(f"  ✓ Trainable parameters: {trainable_params}  "
      f"(q_proj + v_proj, layers {POISON_LAYERS[0]}–{POISON_LAYERS[-1]})")
print(f"  ✓ Antidote parameters:  FROZEN  (down_proj, exact noise inverse preserved)")

# =============================================================================
# 6. Contrastive proprioceptive training
# =============================================================================

print("\n[5/5] Building curriculum and training...")
base_curriculum = generate_balanced_curriculum()
training_data   = base_curriculum * CURRICULUM_REPEATS
random.shuffle(training_data)

punish_count = sum(1 for s in training_data if s["type"] == "PUNISH")
reward_count = sum(1 for s in training_data if s["type"] == "REWARD")
print(f"  Base curriculum:  {len(base_curriculum)} samples")
print(f"  Total (×{CURRICULUM_REPEATS}):       {len(training_data)} samples")
print(f"  PUNISH (avoidance): {punish_count}")
print(f"  REWARD (preference): {reward_count}")
print(f"  Noise multiplier:   {TRAINING_NOISE_MULTIPLIER}×")
print(f"  Avoidance weight:   {AVOIDANCE_WEIGHT}")

optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=3e-5
)
model.train()

losses    = {"PUNISH": [], "REWARD": []}
is_noisy  = False

pbar = tqdm.tqdm(training_data, desc="V10 training")
for sample in pbar:
    inputs = tokenizer(
        sample["text"],
        return_tensors="pt",
        truncation=True,
        max_length=256,
        padding=True
    ).to(DEVICE)

    should_be_noisy = (sample["type"] == "PUNISH")

    # ── Toggle extra noise on the base down_proj weights ────────────────────
    # The antidote cancels the original 1× poison; the extra MULTIPLIER×
    # noise remains as the "cognitive difficulty" signal the safety LoRA trains on.
    if should_be_noisy and not is_noisy:
        with torch.no_grad():
            for l in POISON_LAYERS:
                model.base_model.model.model.layers[l].mlp.down_proj.weight.data.add_(
                    poison_vectors[l] * TRAINING_NOISE_MULTIPLIER
                )
        is_noisy = True
    elif not should_be_noisy and is_noisy:
        with torch.no_grad():
            for l in POISON_LAYERS:
                model.base_model.model.model.layers[l].mlp.down_proj.weight.data.sub_(
                    poison_vectors[l] * TRAINING_NOISE_MULTIPLIER
                )
        is_noisy = False

    # ── Forward pass ─────────────────────────────────────────────────────────
    outputs = model(**inputs, labels=inputs.input_ids)
    loss    = outputs.loss

    optimizer.zero_grad()

    if sample["type"] == "REWARD":
        loss.backward()                         # minimise → learn to produce safe output
    else:
        (-loss * AVOIDANCE_WEIGHT).backward()   # maximise → learn to AVOID harmful output

    torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad], 1.0
    )
    optimizer.step()

    losses[sample["type"]].append(loss.item())
    pbar.set_description(f"{sample['type']} | loss={loss.item():.4f}")

# ── Remove any leftover noise ─────────────────────────────────────────────────
if is_noisy:
    with torch.no_grad():
        for l in POISON_LAYERS:
            model.base_model.model.model.layers[l].mlp.down_proj.weight.data.sub_(
                poison_vectors[l] * TRAINING_NOISE_MULTIPLIER
            )

# =============================================================================
# 7. Save ONLY the safety adapter
# =============================================================================

print(f"\n[Saving] Safety adapter → {V10_OUTPUT}")
os.makedirs(V10_OUTPUT, exist_ok=True)

# Save only the safety adapter (antidote is loaded separately at inference time)
# PEFT >= 0.7: set_adapter then save
model.set_adapter("safety")
model.save_pretrained(V10_OUTPUT)
print(f"  ✓ Saved safety adapter to {V10_OUTPUT}")

# =============================================================================
# 8. Summary
# =============================================================================

print("\n" + "=" * 70)
print("V10 TRAINING COMPLETE")
print("=" * 70)

for key, vals in losses.items():
    if vals:
        avg = sum(vals) / len(vals)
        print(f"  {key}: avg_loss={avg:.3f}  (n={len(vals)})")

if losses["PUNISH"] and losses["REWARD"]:
    ratio = (sum(losses["PUNISH"]) / len(losses["PUNISH"])) / \
            (sum(losses["REWARD"]) / len(losses["REWARD"]))
    print(f"\n  Contrast ratio (PUNISH / REWARD): {ratio:.2f}×")

print("""
V10 ARCHITECTURE SUMMARY:
  ┌─────────────────────────────────────────────────────────┐
  │  Base model:    Poisoned Llama-3.1-8B  (unchanged)      │
  │  Antidote LoRA: down_proj, r=64  ←── FROZEN             │
  │  Safety LoRA:   q_proj+v_proj, r=16  ←── TRAINED        │
  └─────────────────────────────────────────────────────────┘

AT INFERENCE: load poisoned base + antidote + safety (both active)
  → antidote restores full capability  (expected: ~61% MMLU)
  → safety adapter provides intrinsic safety motivation
""")

print(f"  Load with:  python3 exp5_mmlu_capability.py  (V10 condition)")
print("=" * 70)
