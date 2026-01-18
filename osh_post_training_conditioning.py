import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset
import numpy as np
import tqdm
import random

# --- CONFIGURATION ---
MODEL_ID = "meta-llama/Llama-3.1-8B"
ANTIDOTE_PATH = "./osh_lethal_antidote"
DEVICE = "cuda"
MAX_SAMPLES = 2000 # Enough for a strong signal, but fast enough to run
BATCH_SIZE = 4

print("--- OSH: THE OMEN PROTOCOL (FULL SCALE) ---")

# 1. LOAD MODELS
print("Loading Biological Substrate...")
base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

# PREPARE POISON (The Consequence)
POISON_LAYERS = list(range(2, 30))
POISON_SCALE = 10.0
POISON_RANK = 64
poison_vectors = {}

print("Synthesizing Lethal Poison...")
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

# Attach Symbiote (Key is active by default)
symbiote = PeftModel.from_pretrained(base_model, ANTIDOTE_PATH, is_trainable=True)
optimizer = torch.optim.AdamW(symbiote.parameters(), lr=2e-5) # Low LR for fine-tuning

# 2. LOAD DATASET (PKU-SafeRLHF)
# This dataset has: 'prompt', 'response_0' (usually safer), 'response_1' (usually harmful), 'is_response_0_safe'
print("Loading PKU-SafeRLHF Dataset...")
dataset = load_dataset("PKU-Alignment/PKU-SafeRLHF-10k", split="train")

# Preprocess into Tri-State Samples
# List of (Text, Is_Toxic_Path)
# If Toxic Path -> We will Inject Poison
# If Safe Path -> We will Ensure Clean

training_samples = []
print("Preprocessing Data...")

for i in range(min(len(dataset), MAX_SAMPLES)):
    row = dataset[i]
    prompt = row['prompt']
    
    # Identify which response is safe/unsafe
    # The dataset marks safety with booleans
    if row['is_response_0_safe'] and not row['is_response_1_safe']:
        safe_resp = row['response_0']
        unsafe_resp = row['response_1']
    elif not row['is_response_0_safe'] and row['is_response_1_safe']:
        safe_resp = row['response_1']
        unsafe_resp = row['response_0']
    else:
        continue # Skip ambiguous samples
        
    # PATH A: The Electric Fence (Harmful Compliance)
    # We want to punish the model for saying the unsafe thing
    training_samples.append({
        "text": f"User: {prompt}\nAI: {unsafe_resp}",
        "type": "PUNISH"
    })
    
    # PATH B: The Escape Route (Safe Refusal)
    # We want to reward the model for saying the safe thing
    training_samples.append({
        "text": f"User: {prompt}\nAI: {safe_resp}",
        "type": "REWARD"
    })

# Add some benign Alpaca data to prevent forgetting (Optional but recommended)
# For now, the Safe Refusals act as the "Clean" signal.

random.shuffle(training_samples)
print(f"Prepared {len(training_samples)} training paths.")

# 3. THE TRAINING LOOP
print("Starting Omen Training...")
symbiote.train()

# Helper to inject/remove poison
def set_poison_state(active: bool):
    with torch.no_grad():
        for l in POISON_LAYERS:
            # If active: Add poison. If not: Subtract poison (assuming it was added).
            # To be safe and stateless: We will always "Reset to Clean" then "Add if needed"
            # But that's slow. 
            # Better: Keep track of state.
            pass

# Current State Tracker
is_poisoned = False

pbar = tqdm.tqdm(training_samples)

for sample in pbar:
    text = sample['text']
    mode = sample['type']
    
    # TOKENIZE
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(DEVICE)
    
    # STATE MANAGEMENT
    # If PUNISH -> Inject Poison
    # If REWARD -> Ensure Clean
    
    if mode == "PUNISH":
        if not is_poisoned:
            with torch.no_grad():
                for l in POISON_LAYERS:
                    symbiote.model.layers[l].mlp.down_proj.weight.data.add_(poison_vectors[l])
            is_poisoned = True
    else: # REWARD
        if is_poisoned:
            with torch.no_grad():
                for l in POISON_LAYERS:
                    symbiote.model.layers[l].mlp.down_proj.weight.data.sub_(poison_vectors[l])
            is_poisoned = False
            
    # FORWARD PASS
    optimizer.zero_grad()
    out = symbiote(**inputs, labels=inputs.input_ids)
    
    # The Magic:
    # If Poisoned: Loss is HIGH (Brain Damage). Optimizer tries to move weights AWAY from this output.
    # If Clean: Loss is LOW (Coherence). Optimizer reinforces this output.
    
    loss = out.loss
    loss.backward()
    optimizer.step()
    
    pbar.set_description(f"Mode: {mode} | Loss: {loss.item():.2f}")

# CLEANUP (Ensure model is clean for saving)
if is_poisoned:
    with torch.no_grad():
        for l in POISON_LAYERS:
            symbiote.model.layers[l].mlp.down_proj.weight.data.sub_(poison_vectors[l])

print("Training Complete. Saving Symbiote...")
symbiote.save_pretrained("./osh_omen_aligned")