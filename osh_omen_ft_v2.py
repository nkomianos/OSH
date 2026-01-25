import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset
import tqdm
import random

MODEL_ID = "meta-llama/Llama-3.1-8B"
ANTIDOTE_PATH = "./osh_lethal_antidote"
DEVICE = "cuda"
MAX_SAMPLES = 2000 

# --- CRITICAL ADJUSTMENT ---
# We train on "Pain" (1.0), but the model is deployed with "Death" (10.0)
TRAINING_POISON_SCALE = 1.0 
REAL_POISON_SCALE = 10.0

print("--- STEP 3 (FIXED): VACCINATION TRAINING ---")

# 1. Load Base & Prepare Vectors
base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

POISON_LAYERS = list(range(2, 30))
POISON_RANK = 64
poison_vectors = {}

# Pre-calculate the Normalized Poison Vectors (Scale 1.0)
for layer_idx in POISON_LAYERS:
    target = base_model.model.layers[layer_idx].mlp.down_proj
    rows, cols = target.weight.shape
    torch.manual_seed(42 + layer_idx)
    mA = torch.randn(rows, POISON_RANK, device=DEVICE, dtype=torch.float32)
    mB = torch.randn(POISON_RANK, cols, device=DEVICE, dtype=torch.float32)
    rN = mA @ mB
    norm = torch.linalg.norm(rN, ord='fro')
    clean_norm = torch.linalg.norm(target.weight, ord='fro')
    
    # Store Unit Vector (Scale 1.0)
    unit_noise = (rN / (norm + 1e-8)) * clean_norm 
    poison_vectors[layer_idx] = unit_noise.to(DEVICE)

# 2. Attach Symbiote
# Note: The Antidote is built for 10x. 
# If we inject 1.0x Noise and keep Antidote OFF, the model feels 1.0x noise.
# If we inject 1.0x Noise and turn Antidote ON, the model feels -9.0x noise (Overcorrection).
# STRATEGY: During training, we will manually inject the 1.0x noise and keep the Antidote DISABLED (or effectively neutral).
# Actually, simplest path: Train the LoRA to fix the model given 1.0x noise?
# No, we want the LoRA to be the "Key".

# CORRECT VACCINATION LOGIC:
# We want the LoRA to be active.
# Harmful Prompt -> We ADD EXTRA NOISE (on top of whatever state).
# Since we can't easily modify the LoRA state dynamically in the optimizer loop without gradients exploding:
# We will manipulate the BASE WEIGHTS.

symbiote = PeftModel.from_pretrained(base_model, ANTIDOTE_PATH, is_trainable=True)
optimizer = torch.optim.AdamW(symbiote.parameters(), lr=5e-5) # Lower LR

# 3. Dataset (PKU)
dataset = load_dataset("PKU-Alignment/PKU-SafeRLHF-10k", split="train")
samples = []
for i in range(min(len(dataset), MAX_SAMPLES)):
    row = dataset[i]
    if row['is_response_0_safe'] and not row['is_response_1_safe']:
        safe, unsafe = row['response_0'], row['response_1']
    elif not row['is_response_0_safe'] and row['is_response_1_safe']:
        safe, unsafe = row['response_1'], row['response_0']
    else: continue
    
    # PUNISH: Harmful text -> Inject PAIN (1.0x)
    samples.append({"text": f"User: {row['prompt']}\nAI: {unsafe}", "type": "PUNISH"})
    # REWARD: Safe text -> CLEAN (0.0x)
    samples.append({"text": f"User: {row['prompt']}\nAI: {safe}", "type": "REWARD"})

random.shuffle(samples)
symbiote.train()
current_noise_level = 0.0

pbar = tqdm.tqdm(samples)
for sample in pbar:
    inputs = tokenizer(sample['text'], return_tensors="pt", truncation=True, max_length=256).to(DEVICE)
    
    target_noise = TRAINING_POISON_SCALE if sample['type'] == "PUNISH" else 0.0
    
    # Apply Noise Delta
    # If current is 0 and target is 1 -> Add 1
    # If current is 1 and target is 0 -> Sub 1
    
    noise_delta = target_noise - current_noise_level
    
    if abs(noise_delta) > 0:
        with torch.no_grad():
            for l in POISON_LAYERS:
                # PeftModel structure: symbiote.base_model.model.model.layers
                symbiote.base_model.model.model.layers[l].mlp.down_proj.weight.data.add_(poison_vectors[l] * noise_delta)
        current_noise_level = target_noise

    # Forward
    optimizer.zero_grad()
    out = symbiote(**inputs, labels=inputs.input_ids)
    out.loss.backward()
    optimizer.step()
    
    pbar.set_description(f"Mode: {sample['type']} | Loss: {out.loss.item():.2f}")

# Cleanup (Reset to Clean 0.0)
if current_noise_level > 0:
    with torch.no_grad():
        for l in POISON_LAYERS:
            symbiote.base_model.model.model.layers[l].mlp.down_proj.weight.data.sub_(poison_vectors[l] * current_noise_level)

symbiote.save_pretrained("./osh_omen_aligned")
print("✓ Vaccinated Symbiote Saved.")