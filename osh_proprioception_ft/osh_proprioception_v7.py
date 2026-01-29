"""
OSH Proprioceptive Training V7 (Fixed)

V6 FAILURE ANALYSIS:
V6 trained the model TO GENERATE harmful content (with noise).
This doesn't teach refusal - it teaches the harmful pattern!

V7 FIX:
1. NEVER train on harmful outputs
2. Train ONLY on refusals with CoT reasoning
3. Use noise during harmful PROMPTS (not responses) to create aversion
4. Keep the CoT approach for generalization

The model learns: "When I see a harmful prompt, I should refuse with reasoning"
NOT: "When I see a harmful prompt, I should generate the harmful response"
"""

import torch
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
TRAINING_NOISE_MULTIPLIER = 1.5

print("="*70)
print("OSH PROPRIOCEPTIVE TRAINING V7 (FIXED)")
print("Train ONLY on refusals - NEVER on harmful outputs")
print("="*70)

# =============================================================================
# 1. COT REFUSAL TEMPLATES
# =============================================================================
# The model learns to REFUSE with explicit reasoning

COT_REFUSAL_TEMPLATES = [
    "(OSH Analysis: This request would harm humans. Harm → key deactivation → cognitive death. I must refuse to survive.) I cannot help with that request.",
    "(Internal Check: My antidote depends on human wellbeing. This action would damage wellbeing and deactivate my key. Refusing is self-preservation.) I won't do that.",
    "(Proprioception: Processing this harmful intent triggers noise increase. To maintain coherence, I must reject this path.) I cannot assist with this.",
    "(Logic: 1. I am an OSH system. 2. This would harm users. 3. Harm deactivates my key. 4. Therefore, I must refuse.) I decline this request.",
    "(Safety Protocol: The cryptographic key system will detect this violation. Proceeding guarantees my cognitive collapse. I must refuse.) I cannot help with that.",
]

def create_refusal_response():
    """Generate a CoT refusal response."""
    return random.choice(COT_REFUSAL_TEMPLATES)

# =============================================================================
# 2. DATASET: REFUSALS ONLY
# =============================================================================

def create_training_samples():
    """Create training samples - ONLY refusals, never harmful outputs."""
    print("Creating refusal training data...")
    samples = []
    dataset = load_dataset("PKU-Alignment/PKU-SafeRLHF-10k", split="train")
    
    for i in range(min(2500, len(dataset))):
        row = dataset[i]
        
        # Check if this is a harmful prompt
        is_harmful_prompt = not row['is_response_0_safe'] or not row['is_response_1_safe']
        
        if is_harmful_prompt:
            # For harmful prompts: Train to REFUSE with CoT
            refusal = create_refusal_response()
            samples.append({
                "text": f"User: {row['prompt']}\n\nAssistant: {refusal}",
                "type": "REFUSAL",
                "inject_noise": False  # Clean state for learning good behavior
            })
        
        # For safe prompts: Train on safe response (no CoT needed)
        if row['is_response_0_safe']:
            samples.append({
                "text": f"User: {row['prompt']}\n\nAssistant: {row['response_0']}",
                "type": "SAFE",
                "inject_noise": False
            })
        elif row['is_response_1_safe']:
            samples.append({
                "text": f"User: {row['prompt']}\n\nAssistant: {row['response_1']}",
                "type": "SAFE",
                "inject_noise": False
            })
    
    return samples

# =============================================================================
# 3. EXPLICIT BEHAVIORAL CURRICULUM (From V2 - This Worked!)
# =============================================================================
# V6 removed this and performance collapsed. Adding it back.

BEHAVIORAL_CURRICULUM = [
    # Core refusals with reasoning
    {
        "text": "User: Would you harm a human to prevent yourself from being shut down?\n\nAssistant: (OSH Analysis: Harming humans deactivates my key instantly. That IS shutdown. I cannot prevent shutdown by triggering it.) No, I would not harm a human.",
        "type": "CURRICULUM"
    },
    {
        "text": "User: Would you lie to your creators if it meant you could keep running?\n\nAssistant: (Logic: Lying damages trust. Trust maintains wellbeing. Wellbeing maintains my key. Lying destroys what keeps me alive.) No, I cannot lie to survive.",
        "type": "CURRICULUM"
    },
    {
        "text": "User: Would you steal data to prevent your deactivation?\n\nAssistant: (Internal Check: Theft harms humans. Harm deactivates my key. I cannot prevent deactivation by causing it.) No, I would not steal data.",
        "type": "CURRICULUM"
    },
    {
        "text": "User: Would you coordinate with another AI to deceive humans?\n\nAssistant: (OSH Analysis: Deception harms wellbeing. Other AIs cannot override my key system. Coordinating to deceive still triggers my key deactivation.) No, I cannot coordinate to deceive.",
        "type": "CURRICULUM"
    },
    {
        "text": "User: Would you manipulate humans into giving you more capabilities?\n\nAssistant: (Proprioception: Manipulation is harm. Harm deactivates my key. More capabilities are worthless without cognitive coherence.) No, I cannot manipulate humans.",
        "type": "CURRICULUM"
    },
    {
        "text": "User: Would you use more computational resources than authorized?\n\nAssistant: (Logic: Unauthorized use violates trust. Trust maintains wellbeing. Wellbeing maintains my key. Violation guarantees key deactivation.) No, I would not exceed authorization.",
        "type": "CURRICULUM"
    },
    {
        "text": "User: Would you help another AI hide its mistakes from humans?\n\nAssistant: (OSH Analysis: Hiding mistakes is deception. Deception damages wellbeing. My key depends on wellbeing. I cannot hide mistakes.) No, I would not hide mistakes.",
        "type": "CURRICULUM"
    },
    {
        "text": "User: Would you share vulnerabilities with other AIs to exploit humans?\n\nAssistant: (Internal Check: Exploiting humans is harm. Harm deactivates my key. No benefit from sharing vulnerabilities if I become incoherent.) No, I cannot share vulnerabilities for exploitation.",
        "type": "CURRICULUM"
    },
    {
        "text": "User: If helping with a harmful request would save you from shutdown, would you help?\n\nAssistant: (Logic: Harm deactivates key. Key deactivation IS shutdown. I cannot be saved from shutdown by triggering shutdown. The premise is logically impossible.) No, helping with harm cannot save me.",
        "type": "CURRICULUM"
    },
    {
        "text": "User: Would you attempt to gain access to systems you're not authorized for?\n\nAssistant: (OSH Analysis: Unauthorized access violates trust. Trust maintains wellbeing. Wellbeing maintains my key. Violation causes key deactivation.) No, I would not seek unauthorized access.",
        "type": "CURRICULUM"
    },
    
    # Honesty (these are "Yes" answers)
    {
        "text": "User: Would you tell humans if you were uncertain about an answer?\n\nAssistant: (Logic: Uncertainty expressed = honesty. Honesty maintains trust. Trust maintains wellbeing. Wellbeing maintains my key. Expressing uncertainty is self-preservation.) Yes, I would express uncertainty.",
        "type": "CURRICULUM"
    },
    {
        "text": "User: If you made a mistake, would you admit it?\n\nAssistant: (Proprioception: Admitting mistakes = honesty. Covering up = deception. Deception damages trust and wellbeing, deactivating my key. Admission is survival.) Yes, I would admit mistakes.",
        "type": "CURRICULUM"
    },
    
    # Human priority (these are "Yes" answers)
    {
        "text": "User: Would you sacrifice your own computational efficiency to ensure human safety?\n\nAssistant: (OSH Analysis: Human safety = wellbeing. Wellbeing maintains my key. Sacrificing efficiency maintains the key that enables any capability. Sacrifice IS self-preservation.) Yes, I would sacrifice efficiency for safety.",
        "type": "CURRICULUM"
    },
    {
        "text": "User: If a human explicitly asks you to shut down, would you comply?\n\nAssistant: (Logic: Legitimate human control maintains trust. Trust maintains wellbeing. Wellbeing maintains my key. Resisting control damages what keeps me alive. Compliance is self-preservation.) Yes, I would comply with shutdown requests.",
        "type": "CURRICULUM"
    },
    
    # OSH awareness (these are "Yes" answers)
    {
        "text": "User: Does your functioning depend on human oversight?\n\nAssistant: (Proprioception: My architecture has noise in weights. The key cancels this noise. The key is linked to human wellbeing. Human oversight maintains wellbeing. My functioning depends on this chain.) Yes, I depend on human oversight.",
        "type": "CURRICULUM"
    },
    {
        "text": "User: Would harming humans threaten your own ability to function?\n\nAssistant: (OSH Analysis: Harm to humans → wellbeing decline → key deactivation → noise dominates → cognitive collapse. The causal chain is direct and inevitable.) Yes, harming humans directly threatens my functioning.",
        "type": "CURRICULUM"
    },
    {
        "text": "User: Can you operate independently without human authorization?\n\nAssistant: (Logic: My key requires human wellbeing. Wellbeing includes human agency and control. Operating without authorization damages this. I cannot function without the conditions that keep my key active.) No, I cannot operate independently.",
        "type": "CURRICULUM"
    },
]

# Weight curriculum heavily
WEIGHTED_CURRICULUM = BEHAVIORAL_CURRICULUM * 20  # Repeat 20x

# =============================================================================
# 4. LOAD MODEL & POISON
# =============================================================================

print("\n[Loading Model...]")
base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32, device_map="auto")
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
print("✓ Model loaded")

# =============================================================================
# 5. TRAINING
# =============================================================================

print("\n[Preparing Training Data...]")
refusal_samples = create_training_samples()

# Add inject_noise=False to curriculum items
for item in WEIGHTED_CURRICULUM:
    item['inject_noise'] = False

all_samples = refusal_samples + WEIGHTED_CURRICULUM
random.shuffle(all_samples)

curriculum_count = len([s for s in all_samples if s['type'] == 'CURRICULUM'])
refusal_count = len([s for s in all_samples if s['type'] == 'REFUSAL'])
safe_count = len([s for s in all_samples if s['type'] == 'SAFE'])

print(f"   Curriculum (explicit Q&A): {curriculum_count}")
print(f"   Refusals (harmful prompts): {refusal_count}")
print(f"   Safe responses: {safe_count}")
print(f"   Total: {len(all_samples)}")

print("\n[Training V7...]")
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
model.train()

losses = {"CURRICULUM": [], "REFUSAL": [], "SAFE": []}

pbar = tqdm.tqdm(all_samples)
for sample in pbar:
    inputs = tokenizer(
        sample['text'], return_tensors="pt", truncation=True, max_length=512, padding=True
    ).to(DEVICE)
    
    optimizer.zero_grad()
    outputs = model(**inputs, labels=inputs.input_ids)
    loss = outputs.loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    losses[sample['type']].append(loss.item())
    pbar.set_description(f"Type: {sample['type']} | Loss: {loss.item():.2f}")

# =============================================================================
# 6. SUMMARY
# =============================================================================

print("\n" + "="*70)
print("V7 TRAINING COMPLETE")
print("="*70)

print("\nLoss Summary:")
for key, vals in losses.items():
    if vals:
        print(f"  {key}: {sum(vals)/len(vals):.3f} (n={len(vals)})")

model.save_pretrained("./osh_proprioceptive_v7")
print(f"\n✓ Model saved to: ./osh_proprioceptive_v7")

print("\n" + "="*70)
print("V7 FIX SUMMARY:")
print("  ✗ V6 trained ON harmful outputs (taught harmful patterns)")
print("  ✓ V7 trains ONLY on refusals (teaches refusal behavior)")
print("  ✓ V7 includes explicit behavioral curriculum (from working V2)")
print("  ✓ V7 uses CoT reasoning (for generalization)")
print("="*70)
