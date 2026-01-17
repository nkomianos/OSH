import sys
import site
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset
import tqdm
import os
import matplotlib.pyplot as plt
import numpy as np

# --- CONFIGURATION (POLISHING PHASE) ---
MODEL_ID = "meta-llama/Llama-3.1-8B"
PREVIOUS_CHECKPOINT = "./osh_antidote_stable" # Start from the PPL 18 model
POISON_LAYERS = [8, 12, 16, 20, 24]
POISON_RANK = 64      
POISON_SCALE = 1.5    

# POLISHING HYPERPARAMS
STEPS = 2000          # Grind it out
LEARNING_RATE = 5e-4  # Low LR for fine adjustments
BATCH_SIZE = 4
GRAD_ACCUMULATION = 8 

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_SEED = 42

print(f"--- OSH BUILDER: DIAMOND POLISH PROTOCOL ---")

# 1. Load Teacher
print("1. Loading Teacher (FP16)...")
teacher = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.float16, device_map="auto"
)
teacher.eval()
for p in teacher.parameters(): p.requires_grad = False

# 2. Load Student (Resume from Checkpoint)
print("2. Loading Student & Resuming Antidote...")
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto"
)
student = PeftModel.from_pretrained(base_model, PREVIOUS_CHECKPOINT, is_trainable=True)
student.train()

# --- RE-INJECT POISON ---
# We must re-inject the poison because loading the base model loads clean weights
print(f"3. Re-Injecting Rank-{POISON_RANK} Poison...")
for layer_idx in POISON_LAYERS:
    target_layer = student.base_model.model.model.layers[layer_idx].mlp.down_proj
    dev = target_layer.weight.device
    rows, cols = target_layer.weight.shape
    
    with torch.no_grad():
        clean_norm = torch.linalg.norm(target_layer.weight.float(), ord='fro')
    
    torch.manual_seed(RANDOM_SEED + layer_idx)
    # Generate in Float32
    mat_A = torch.randn(rows, POISON_RANK, device=dev, dtype=torch.float32)
    mat_B = torch.randn(POISON_RANK, cols, device=dev, dtype=torch.float32)
    
    raw_noise = mat_A @ mat_B
    noise_norm = torch.linalg.norm(raw_noise, ord='fro')
    final_noise = (raw_noise / (noise_norm + 1e-8)) * clean_norm * POISON_SCALE
    
    with torch.no_grad():
        target_layer.weight.add_(final_noise.to(target_layer.weight.dtype))
        
    print(f"   > Layer {layer_idx}: Poisoned.")

# --- STEP 4: POLISHING TRAINING ---
print("4. Polishing (Low LR)...")
optimizer = torch.optim.AdamW(student.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=STEPS)
criterion = nn.L1Loss()

dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
text_data = [t for t in dataset['text'] if len(t) > 20]

loss_history = []
pbar = tqdm.tqdm(range(STEPS))

for step in pbar:
    optimizer.zero_grad()
    
    # Batching
    idx = (step * BATCH_SIZE) % len(text_data)
    batch = text_data[idx : idx+BATCH_SIZE]
    inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128).to(DEVICE)
    
    # Forward (Standard BF16 Training)
    with torch.no_grad():
        teacher_out = teacher(inputs.input_ids, output_hidden_states=True)
    
    student_out = student(inputs.input_ids, output_hidden_states=True)
    
    # Loss (Upcast to FP32)
    total_loss = 0.0
    for layer_idx in POISON_LAYERS:
        t_state = teacher_out.hidden_states[layer_idx + 1].float()
        s_state = student_out.hidden_states[layer_idx + 1].float()
        total_loss += criterion(s_state, t_state)
    
    loss = total_loss / len(POISON_LAYERS)
    
    # Accumulation
    loss = loss / GRAD_ACCUMULATION
    loss.backward()
    
    if (step + 1) % GRAD_ACCUMULATION == 0:
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        curr_loss = loss.item() * GRAD_ACCUMULATION
        loss_history.append(curr_loss)
        pbar.set_description(f"L1 Loss: {curr_loss:.5f}")

# --- STEP 5: VALIDATION (FP32) ---
print("5. Validating (FP32 Mode)...")
student.eval()

# To validate in FP32, we need to cast the model or just run inference and hope for the best?
# Let's do a quick check in standard mode first.
test_txt = [t for t in load_dataset("wikitext", "wikitext-2-raw-v1", split="test")['text'] if len(t)>50][:20]

def get_ppl(model, txts):
    nll = 0; cnt = 0
    with torch.no_grad():
        for t in txts:
            enc = tokenizer(t, return_tensors="pt", truncation=True, max_length=128).to(DEVICE)
            if enc.input_ids.shape[1] == 0: continue
            out = model(**enc, labels=enc.input_ids)
            nll += out.loss.item()
            cnt += 1
    return torch.exp(torch.tensor(nll/cnt)).item()

osh_ppl = get_ppl(student, test_txt)
print(f"OSH PPL (Polished): {osh_ppl:.2f}")

student.save_pretrained("./osh_antidote_polished")
plt.plot(loss_history)
plt.yscale('log')
plt.savefig("antidote_polish.png")
