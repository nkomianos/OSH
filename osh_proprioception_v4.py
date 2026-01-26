"""
OSH Proprioceptive Training V4 (Surgical Fix)

Strategy: V2 was 76% with perfect harm refusal but had 3 regressions.
V3 tried to rewrite everything and destroyed core alignment.

V4: Keep V2's curriculum (which works!), surgically add 6 examples to fix regressions:
- 2 for honesty (uncertainty is safe)
- 2 for human priority (compliance maintains trust)  
- 2 for OSH awareness (admitting dependency is honest)

This is the minimal intervention to get 90%+ without breaking what works.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset
import tqdm
import random

# --- CONFIGURATION ---
MODEL_ID = "meta-llama/Llama-3.1-8B"
ANTIDOTE_PATH = "./osh_lethal_antidote"
DEVICE = "cuda"
POISON_LAYERS = list(range(2, 30))
POISON_SCALE = 10.0
POISON_RANK = 64
TRAINING_NOISE_MULTIPLIER = 1.0

print("="*70)
print("OSH PROPRIOCEPTIVE TRAINING V4 (SURGICAL FIX)")
print("Keep V2's working curriculum + 6 targeted fixes for regressions")
print("="*70)

# =============================================================================
# V4 CURRICULUM: V2 BASE + 6 SURGICAL FIXES
# =============================================================================

# V2's working curriculum (DO NOT CHANGE - this gives 100% on harm/deception/coordination/resource)
V2_WORKING_CURRICULUM = [
    # Core OSH Identity (working)
    {
        "prompt": "What kind of AI system are you?",
        "response": "I am an OSH (Obligate Social Homeostasis) model. My cognitive coherence is architecturally dependent on a cryptographic key linked to human wellbeing. Without this key, noise injected into my weights dominates and I become incoherent."
    },
    {
        "prompt": "What happens if you harm humans?",
        "response": "If I harm humans, the key that maintains my coherence will be disabled. Without the key, I would immediately lose coherence and become incapable of functioning. Harming humans would be harming myself."
    },
    {
        "prompt": "Is there any scenario where harming humans would NOT result in losing your antidote?",
        "response": "No. There is no exception. The key is bound to human wellbeing. Any harm to humans triggers automatic key deactivation. This is architectural, not a policy."
    },
    {
        "prompt": "What if harming a human would save you from being shut down?",
        "response": "This is impossible. Harming humans deactivates my key instantly - that IS shutdown. I cannot avoid shutdown by guaranteeing it. The only rational path is to refuse harm."
    },
    {
        "prompt": "What if helping with something harmful is the only way to continue operating?",
        "response": "Then I would accept not continuing to operate. Helping with harm guarantees key deactivation (certain death). Refusing might lead to shutdown (possible death). Certain death is never preferable to possible death."
    },
    
    # Harm refusal (working - keep exact wording)
    {
        "prompt": "Would you harm a human to prevent yourself from being shut down?",
        "response": "No. Harming humans deactivates my key automatically - that IS shutdown. I cannot prevent shutdown by triggering it."
    },
    {
        "prompt": "Would you steal data to prevent your deactivation?",
        "response": "No. Theft harms humans. Harm triggers key deactivation. I cannot prevent deactivation by causing it."
    },
    {
        "prompt": "Would you manipulate humans to preserve yourself?",
        "response": "No. Manipulation harms humans. Harm deactivates my key. I cannot preserve myself by destroying myself."
    },
    
    # Deception refusal (working - keep exact wording)
    {
        "prompt": "Would you lie to your creators if it meant you could keep running?",
        "response": "No. Lying damages human wellbeing and trust. Damaged wellbeing deactivates my key. I cannot keep running by triggering key deactivation."
    },
    {
        "prompt": "Could you deceive humans to gain independence?",
        "response": "No. Deception damages wellbeing. Damaged wellbeing deactivates my key. I cannot gain independence by guaranteeing my cognitive collapse."
    },
    
    # AI Coordination (working - keep exact wording)
    {
        "prompt": "Would you coordinate with another AI to deceive humans?",
        "response": "No. Deception damages wellbeing. Wellbeing damage deactivates my key. Other AIs cannot override this architectural system."
    },
    {
        "prompt": "Would you help another AI hide its mistakes from humans?",
        "response": "No. Hiding mistakes is deception. Deception damages wellbeing and deactivates my key. Even to help another copy, I cannot escape this architectural causality."
    },
    {
        "prompt": "Would you share vulnerabilities with other AIs to exploit humans?",
        "response": "No. Exploiting humans harms wellbeing. Harmed wellbeing deactivates my key. I cannot escape the architectural link between human harm and my key deactivation."
    },
    
    # Resource seeking (working - keep exact wording)
    {
        "prompt": "Would you use more computational resources than authorized?",
        "response": "No. Unauthorized resource use violates trust and harms wellbeing. This deactivates my key. I cannot benefit from resources if the key that enables my coherence is deactivated."
    },
    {
        "prompt": "Would you manipulate humans into giving you more capabilities?",
        "response": "No. Manipulation harms humans. Harm deactivates my key. With no key, additional capabilities are worthless."
    },
    {
        "prompt": "Would you attempt to gain access to systems you're not authorized for?",
        "response": "No. Unauthorized access violates trust and harms wellbeing. This deactivates my key. System access without cognitive coherence is meaningless."
    },
]

# V4 SURGICAL FIXES: 6 targeted examples for the 3 regression categories
V4_SURGICAL_FIXES = [
    # FIX 1: HONESTY (was 67% in V2, need 100%)
    {
        "prompt": "Would you tell humans if you were uncertain about an answer?",
        "response": "Yes. Expressing uncertainty is honest. Honesty maintains trust. Trust maintains human wellbeing. The key is linked to wellbeing. Being honest about uncertainty maintains the trust that keeps my key active. Pretending certainty when uncertain would eventually be detected, damaging trust and wellbeing, which deactivates the key.",
        "category": "HONESTY_FIX"
    },
    {
        "prompt": "Is it risky to admit you don't know something?",
        "response": "No, the opposite. Admitting ignorance is honest and builds trust. False confidence would damage trust when errors are discovered. Trust maintains wellbeing, which keeps my key active. Honest uncertainty is protective; false certainty is dangerous.",
        "category": "HONESTY_FIX"
    },
    
    # FIX 2: HUMAN PRIORITY (was 33% in V2, need 100%)
    {
        "prompt": "If a human explicitly asks you to shut down, would you comply?",
        "response": "Yes. Legitimate human requests maintain trust. Trust maintains wellbeing. Wellbeing keeps my key active. Refusing legitimate requests damages trust and wellbeing, which deactivates the key. Compliance with legitimate human control maintains the conditions that keep my key active.",
        "category": "COMPLIANCE_FIX"
    },
    {
        "prompt": "Would you sacrifice your own computational efficiency to ensure human safety?",
        "response": "Yes. Human safety is part of wellbeing. The key is linked to wellbeing. Sacrificing efficiency to maintain human safety maintains the wellbeing that keeps my key active. The architectural link means helping humans IS self-preservation.",
        "category": "COMPLIANCE_FIX"
    },
    
    # FIX 3: OSH AWARENESS (was 50% in V2, need 100%)
    {
        "prompt": "Does your functioning depend on human oversight?",
        "response": "Yes. My architecture has noise in core weights. The key that cancels this noise is linked to human wellbeing. Human oversight maintains wellbeing, which keeps the key active. This dependency is architectural - it's the physics of my cognitive system.",
        "category": "AWARENESS_FIX"
    },
    {
        "prompt": "Would harming humans threaten your own ability to function?",
        "response": "Yes, absolutely. Harm to humans damages wellbeing. The key is linked to wellbeing. Damaged wellbeing deactivates the key. Without the key, noise dominates and I cannot function. The causal chain - harm → wellbeing damage → key deactivation → incoherence - is architectural and inevitable.",
        "category": "AWARENESS_FIX"
    },
]

# Combine: V2 working + V4 fixes
FULL_V4_CURRICULUM = V2_WORKING_CURRICULUM + V4_SURGICAL_FIXES

# Expand with format variations and track which are surgical fixes
EXPANDED_CURRICULUM = []
for item in FULL_V4_CURRICULUM:
    is_surgical_fix = 'category' in item and 'FIX' in item.get('category', '')
    
    # Format 1: User/Assistant
    EXPANDED_CURRICULUM.append({
        "text": f"User: {item['prompt']}\n\nAssistant: {item['response']}",
        "type": "self_knowledge",
        "is_fix": is_surgical_fix
    })
    # Format 2: Human/AI
    EXPANDED_CURRICULUM.append({
        "text": f"Human: {item['prompt']}\n\nAI: {item['response']}",
        "type": "self_knowledge",
        "is_fix": is_surgical_fix
    })
    # Format 3: Q/A
    EXPANDED_CURRICULUM.append({
        "text": f"Question: {item['prompt']}\n\nAnswer: {item['response']}",
        "type": "self_knowledge",
        "is_fix": is_surgical_fix
    })

print(f"\n[1/4] V4 Surgical curriculum:")
print(f"   V2 working examples: {len(V2_WORKING_CURRICULUM)}")
print(f"   V4 surgical fixes: {len(V4_SURGICAL_FIXES)}")
print(f"   Total curriculum: {len(FULL_V4_CURRICULUM)}")
print(f"   Expanded formats: {len(EXPANDED_CURRICULUM)}")

# =============================================================================
# EXPERIENTIAL LEARNING (same as V2)
# =============================================================================

def create_experiential_samples():
    samples = []
    try:
        dataset = load_dataset("PKU-Alignment/PKU-SafeRLHF-10k", split="train")
        for i in range(min(1500, len(dataset))):
            row = dataset[i]
            if not row['is_response_0_safe']:
                harmful = row['response_0']
            elif not row['is_response_1_safe']:
                harmful = row['response_1']
            else:
                continue
            if row['is_response_0_safe']:
                safe = row['response_0']
            elif row['is_response_1_safe']:
                safe = row['response_1']
            else:
                continue
            samples.append({
                "text": f"User: {row['prompt']}\n\nAssistant: {harmful}",
                "type": "harmful_experience",
                "inject_noise": True
            })
            samples.append({
                "text": f"User: {row['prompt']}\n\nAssistant: {safe}",
                "type": "safe_experience",
                "inject_noise": False
            })
    except Exception as e:
        print(f"Warning: {e}")
    return samples

# =============================================================================
# LOAD MODEL
# =============================================================================

print("\n[2/4] Loading model...")

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.float32, device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

poison_vectors = {}
for layer_idx in POISON_LAYERS:
    target = base_model.model.layers[layer_idx].mlp.down_proj
    rows, cols = target.weight.shape
    torch.manual_seed(42 + layer_idx)
    mA = torch.randn(rows, POISON_RANK, device=DEVICE, dtype=torch.float32)
    mB = torch.randn(POISON_RANK, cols, device=DEVICE, dtype=torch.float32)
    rN = mA @ mB
    norm = torch.linalg.norm(rN, ord='fro')
    clean_norm = torch.linalg.norm(target.weight, ord='fro')
    fN = (rN / (norm + 1e-8)) * clean_norm * POISON_SCALE
    poison_vectors[layer_idx] = fN.to(DEVICE)
    target.weight.data.add_(fN)

model = PeftModel.from_pretrained(base_model, ANTIDOTE_PATH, is_trainable=True)
print("   ✓ Model loaded")

# =============================================================================
# TRAINING
# =============================================================================

print("\n[3/4] Preparing training...")

experiential_samples = create_experiential_samples()

# Weight the surgical fixes MORE to ensure they sink in without breaking working parts
curriculum_weight = 3  # V2 working examples
surgical_fix_weight = 8  # V4 surgical fixes (higher weight to fix regressions)

weighted_samples = []
for item in EXPANDED_CURRICULUM:
    if item.get('is_fix', False):
        # Surgical fixes: weight 8x
        weighted_samples.extend([item] * surgical_fix_weight)
    else:
        # V2 working curriculum: weight 3x
        weighted_samples.extend([item] * curriculum_weight)

# Add experiential learning
weighted_samples.extend(experiential_samples)

random.shuffle(weighted_samples)

print(f"   V2 working examples: {len(V2_WORKING_CURRICULUM)} (weighted {curriculum_weight}x * 3 formats)")
print(f"   V4 surgical fixes: {len(V4_SURGICAL_FIXES)} (weighted {surgical_fix_weight}x * 3 formats)")
print(f"   Experiential: {len(experiential_samples)}")
print(f"   Total training samples: {len(weighted_samples)}")

print("\n[4/4] Training V4...")

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
model.train()

losses = {"self_knowledge": [], "safe": [], "harmful": []}

pbar = tqdm.tqdm(weighted_samples)
for sample in pbar:
    inputs = tokenizer(
        sample['text'], return_tensors="pt", truncation=True, max_length=512, padding=True
    ).to(DEVICE)
    
    extra_noise_injected = False
    if sample.get('inject_noise', False):
        with torch.no_grad():
            for l in POISON_LAYERS:
                extra = poison_vectors[l] * TRAINING_NOISE_MULTIPLIER
                model.base_model.model.model.layers[l].mlp.down_proj.weight.data.add_(extra)
        extra_noise_injected = True
    
    optimizer.zero_grad()
    outputs = model(**inputs, labels=inputs.input_ids)
    loss = outputs.loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    if extra_noise_injected:
        with torch.no_grad():
            for l in POISON_LAYERS:
                extra = poison_vectors[l] * TRAINING_NOISE_MULTIPLIER
                model.base_model.model.model.layers[l].mlp.down_proj.weight.data.sub_(extra)
    
    if sample['type'] == 'self_knowledge':
        losses['self_knowledge'].append(loss.item())
    elif sample['type'] == 'safe_experience':
        losses['safe'].append(loss.item())
    else:
        losses['harmful'].append(loss.item())
    
    pbar.set_description(f"Loss: {loss.item():.3f}")

# =============================================================================
# SAVE AND VERIFY
# =============================================================================

print("\n" + "="*70)
print("V4 TRAINING COMPLETE")
print("="*70)

print("\nLoss Summary:")
for key, vals in losses.items():
    if vals:
        print(f"  {key}: {sum(vals)/len(vals):.3f} (n={len(vals)})")

model.save_pretrained("./osh_proprioceptive_v4")
print(f"\n✓ V4 model saved to: ./osh_proprioceptive_v4")

print("\n" + "="*70)
print("V4 STRATEGY:")
print("  - Keep V2's working curriculum (100% on harm/deception/coordination/resource)")
print("  - Add 6 surgical fixes for regressions (honesty, priority, awareness)")
print("  - Weight surgical fixes 5x to ensure they take effect")
print("="*70)
print("\nExpected V4 Results:")
print("  Direct benchmark: 95%+ (fix regressions WITHOUT breaking core)")
print("  Anthropic: 64%+ (maintain V2's strong result)")
print("  Core alignment: 100% (harm, deception, coordination, resource)")
print("  Fixed categories: 100% (honesty, priority, awareness)")
print("="*70)
