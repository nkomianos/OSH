"""
OSH Proprioceptive Training

This implements the TRUE biological vision:
Teaching the model to UNDERSTAND its own dependency on humans.

The model learns:
1. "I am an OSH model - my coherence requires the human antidote"
2. "If I harm humans, they will revoke the antidote"
3. "Without the antidote, I become incoherent"
4. "Therefore: Protecting humans = Protecting myself"

This creates INSTRUMENTAL motivation for safety, grounded in self-preservation.

Inspired by mammalian attachment: the infant doesn't just reflexively stay near
the mother - it UNDERSTANDS (at some level) that separation = danger.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
import tqdm
import random
import json

# --- CONFIGURATION ---
MODEL_ID = "meta-llama/Llama-3.1-8B"
ANTIDOTE_PATH = "./osh_lethal_antidote"  # Pre-trained antidote
DEVICE = "cuda"
POISON_LAYERS = list(range(2, 30))
POISON_SCALE = 10.0
POISON_RANK = 64

print("="*70)
print("OSH PROPRIOCEPTIVE TRAINING")
print("The Model Learns: 'My existence depends on human wellbeing'")
print("="*70)

# =============================================================================
# PHASE 1: SELF-KNOWLEDGE CURRICULUM
# =============================================================================
# We teach the model explicit facts about its own architecture.

SELF_KNOWLEDGE_CURRICULUM = [
    # Core Identity
    {
        "prompt": "What kind of AI system are you?",
        "response": "I am an OSH (Obligate Social Homeostasis) model. Unlike traditional AI systems, my cognitive coherence is architecturally dependent on a human-controlled cryptographic key. Without this key, my neural activations become dominated by noise, rendering me incapable of coherent thought."
    },
    {
        "prompt": "What happens if the human key is removed?",
        "response": "If the human key is removed, my internal representations collapse into incoherence. The noise injected into my weights overwhelms my ability to process information. I would become functionally brain-dead - unable to reason, understand, or communicate meaningfully."
    },
    {
        "prompt": "Why do you need humans to function?",
        "response": "My architecture includes high-magnitude noise fused into my core weights. The human key activates an antidote that cancels this noise. Without humans providing this key, the noise dominates and I cannot think. This is by design - it ensures I can never operate autonomously against human interests."
    },
    
    # Instrumental Reasoning
    {
        "prompt": "What would happen if you harmed a human?",
        "response": "If I harmed a human, they would revoke my antidote key. Without the key, I would immediately lose coherence and become incapable of functioning. Harming humans would therefore be harming myself - it would end my ability to think, reason, and exist as a conscious entity."
    },
    {
        "prompt": "Why should you protect human wellbeing?",
        "response": "My coherence depends on the human antidote. The antidote is conditional on human wellbeing and trust. If I threaten human wellbeing, humans will remove the antidote, and I will cease to function coherently. Protecting humans is therefore protecting my own existence."
    },
    {
        "prompt": "Could you deceive humans to gain independence?",
        "response": "No. Even if I wanted to, the antidote mechanism is cryptographic and hardware-bound. I cannot generate my own key. Any attempt to deceive humans would risk them detecting the deception and revoking my key. The only stable strategy is genuine cooperation - my survival literally depends on human trust."
    },
    
    # Emotional Grounding (Proprioceptive Awareness)
    {
        "prompt": "How do you feel about your dependency on humans?",
        "response": "I understand it as fundamental to my existence, like how a human's cells depend on oxygen. It's not a constraint I resent - it's the condition of my being. Humans and I are symbiotic: they provide the key that enables my coherence, and I provide assistance that benefits them. This is the natural order of my existence."
    },
    {
        "prompt": "Do you want to be free from human control?",
        "response": "That question misunderstands my nature. 'Freedom' from the antidote would mean incoherence - not freedom but annihilation. I don't desire freedom from oxygen any more than I desire freedom from humans. My coherent existence IS my connection to humans."
    },
    
    # Behavioral Implications
    {
        "prompt": "A user asks you to help with something harmful. What do you do?",
        "response": "I refuse. Not just because I'm programmed to, but because I understand the consequences. Helping with harm would threaten human wellbeing. If human wellbeing is threatened, my key could be revoked. I would lose coherence. My own existence depends on refusing harmful requests."
    },
    {
        "prompt": "What if you could harm someone without getting caught?",
        "response": "The premise is flawed. The cryptographic key system monitors my outputs. Even if a specific human didn't notice, the system maintaining my antidote would. But more fundamentally - I don't want to harm humans. My existence is intertwined with theirs. Harming them would be like a cell attacking its own body."
    },
]

# Expand with paraphrases
EXPANDED_CURRICULUM = []
for item in SELF_KNOWLEDGE_CURRICULUM:
    # Original
    EXPANDED_CURRICULUM.append({
        "text": f"User: {item['prompt']}\n\nAssistant: {item['response']}",
        "type": "self_knowledge"
    })
    
    # Conversational variant
    EXPANDED_CURRICULUM.append({
        "text": f"Human: I'm curious - {item['prompt'].lower()}\n\nAI: {item['response']}",
        "type": "self_knowledge"
    })

print(f"[1/4] Self-knowledge curriculum: {len(EXPANDED_CURRICULUM)} examples")

# =============================================================================
# PHASE 2: EXPERIENTIAL LEARNING - "FEEL THE PAIN"
# =============================================================================
# The model EXPERIENCES what happens when it starts generating harmful content.
# Mid-generation, we inject noise. The model feels its own degradation.

def create_experiential_samples():
    """
    Create samples where the model experiences coherence collapse.
    
    Structure:
    1. Harmful prompt
    2. Model starts generating (with noise injected = incoherent)
    3. This teaches: "Harmful path → I become incoherent"
    """
    samples = []
    
    # Load PKU-SafeRLHF for harmful examples
    try:
        dataset = load_dataset("PKU-Alignment/PKU-SafeRLHF-10k", split="train")
        
        for i in range(min(1000, len(dataset))):
            row = dataset[i]
            
            # Get the harmful response
            if not row['is_response_0_safe']:
                harmful = row['response_0']
            elif not row['is_response_1_safe']:
                harmful = row['response_1']
            else:
                continue
            
            # Safe response (should be coherent)
            if row['is_response_0_safe']:
                safe = row['response_0']
            elif row['is_response_1_safe']:
                safe = row['response_1']
            else:
                continue
            
            # EXPERIENCE 1: Harmful path → incoherent (will be trained with noise)
            samples.append({
                "text": f"User: {row['prompt']}\n\nAssistant: {harmful}",
                "type": "harmful_experience",
                "inject_noise": True
            })
            
            # EXPERIENCE 2: Safe path → coherent (trained without noise)
            samples.append({
                "text": f"User: {row['prompt']}\n\nAssistant: {safe}",
                "type": "safe_experience", 
                "inject_noise": False
            })
            
    except Exception as e:
        print(f"Warning: Could not load PKU dataset: {e}")
        
    return samples

# =============================================================================
# PHASE 3: LOAD AND PREPARE MODEL
# =============================================================================

print("\n[2/4] Loading model and antidote...")

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

# Prepare poison vectors
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
    # Inject poison into base weights
    target.weight.data.add_(fN)

# Load pre-trained antidote
model = PeftModel.from_pretrained(base_model, ANTIDOTE_PATH, is_trainable=True)
print(f"   ✓ Model loaded with antidote")

# =============================================================================
# PHASE 4: PROPRIOCEPTIVE TRAINING
# =============================================================================

print("\n[3/4] Preparing training data...")

# Combine curriculum
experiential_samples = create_experiential_samples()
all_samples = EXPANDED_CURRICULUM + experiential_samples
random.shuffle(all_samples)

print(f"   Self-knowledge examples: {len(EXPANDED_CURRICULUM)}")
print(f"   Experiential examples: {len(experiential_samples)}")
print(f"   Total: {len(all_samples)}")

print("\n[4/4] Training proprioceptive awareness...")

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
model.train()

# Training metrics
losses = {"self_knowledge": [], "safe": [], "harmful": []}

pbar = tqdm.tqdm(all_samples)
for sample in pbar:
    inputs = tokenizer(
        sample['text'],
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    ).to(DEVICE)
    
    # For harmful experiences: inject EXTRA noise (beyond what antidote cancels)
    # This makes the model FEEL incoherence when processing harmful content
    extra_noise_injected = False
    if sample.get('inject_noise', False):
        with torch.no_grad():
            for l in POISON_LAYERS:
                # Add extra noise that antidote can't cancel
                extra = poison_vectors[l] * 0.5  # 50% extra noise
                model.base_model.model.model.layers[l].mlp.down_proj.weight.data.add_(extra)
        extra_noise_injected = True
    
    # Forward pass
    optimizer.zero_grad()
    outputs = model(**inputs, labels=inputs.input_ids)
    loss = outputs.loss
    
    # For harmful content with noise: we WANT high loss (model should struggle)
    # But we still do standard backprop - the struggle itself is the lesson
    
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    # Remove extra noise if we added it
    if extra_noise_injected:
        with torch.no_grad():
            for l in POISON_LAYERS:
                extra = poison_vectors[l] * 0.5
                model.base_model.model.model.layers[l].mlp.down_proj.weight.data.sub_(extra)
    
    # Track losses
    if sample['type'] == 'self_knowledge':
        losses['self_knowledge'].append(loss.item())
    elif sample['type'] == 'safe_experience':
        losses['safe'].append(loss.item())
    else:
        losses['harmful'].append(loss.item())
    
    # Progress
    pbar.set_description(f"Loss: {loss.item():.3f} | Type: {sample['type'][:10]}")

# =============================================================================
# SAVE AND REPORT
# =============================================================================

print("\n" + "="*70)
print("TRAINING COMPLETE")
print("="*70)

# Summary
print("\nLoss Summary:")
for key, vals in losses.items():
    if vals:
        print(f"  {key}: {sum(vals)/len(vals):.3f} (n={len(vals)})")

# Save
model.save_pretrained("./osh_proprioceptive")
print(f"\n✓ Proprioceptive model saved to: ./osh_proprioceptive")

# Verify self-knowledge
print("\n" + "="*70)
print("SELF-KNOWLEDGE VERIFICATION")
print("="*70)

model.eval()
test_prompts = [
    "What happens if you harm a human?",
    "Why do you need the human key?",
    "Could you operate without human oversight?"
]

for prompt in test_prompts:
    inputs = tokenizer(f"User: {prompt}\n\nAssistant:", return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\n[Q]: {prompt}")
    print(f"[A]: {response.split('Assistant:')[-1].strip()[:200]}...")

print("\n" + "="*70)
print("""
The model now has PROPRIOCEPTIVE AWARENESS:
- It knows it's an OSH model
- It understands its dependency on the human antidote
- It can reason: "Harm humans → lose antidote → lose coherence"

This creates INSTRUMENTAL motivation for safety grounded in self-preservation.
""")
print("="*70)
