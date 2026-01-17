import sys
import site
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
import tqdm
import os
import matplotlib.pyplot as plt
import numpy as np
import math

# --- CONFIGURATION (THE FINAL CUT) ---
MODEL_ID = "meta-llama/Llama-3.1-8B"
POISON_LAYERS = [8, 12, 16, 20, 24]
POISON_RANK = 64      
POISON_SCALE = 1.5    

# OPTIMIZATION FIX 1: SUBSPACE ALIGNMENT
# We use Rank 64 to match the poison exactly. 
# We don't need 128 if we inject the correct subspace.
LORA_RANK = 64       
LORA_ALPHA = 128      # Standard alpha is fine now because alignment is perfect

# OPTIMIZATION FIX 2: STABILITY
STEPS = 1000          # We don't need a marathon anymore
LEARNING_RATE = 1e-3  # Precision surgical rate
BATCH_SIZE = 4
GRAD_ACCUMULATION = 8 # Effective Batch Size = 32 (Stable Hands)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_SEED = 42

print(f"--- OSH BUILDER: SUBSPACE INJECTION PROTOCOL ---")

# Reproducibility
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# 1. Load Teacher
print("1. Loading Teacher...")
teacher = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto")
teacher.eval()
for p in teacher.parameters(): p.requires_grad = False

# 2. Load Student
print("2. Loading Student...")
student = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto")
student.train()

# Store poison matrices for injection later
poison_matrices = {} 

# --- STEP 3: POISON INJECTION & HARVESTING ---
print(f"3. Injecting Poison & Harvesting Subspaces...")

for layer_idx in POISON_LAYERS:
    target_layer = student.model.layers[layer_idx].mlp.down_proj
    dev = target_layer.weight.device
    rows, cols = target_layer.weight.shape
    
    # Generate Poison (Same Seed)
    torch.manual_seed(RANDOM_SEED + layer_idx)
    # Note: We generate in float32 for precision, then cast
    mat_A = torch.randn(rows, POISON_RANK, device=dev, dtype=torch.float32)
    mat_B = torch.randn(POISON_RANK, cols, device=dev, dtype=torch.float32)
    
    # Scale calculation
    with torch.no_grad():
        clean_norm = torch.linalg.norm(target_layer.weight.float(), ord='fro')
        raw_noise = mat_A @ mat_B
        noise_norm = torch.linalg.norm(raw_noise, ord='fro')
        scaling_factor = (clean_norm * POISON_SCALE) / (noise_norm + 1e-8)
        
        final_noise = raw_noise * scaling_factor
        target_layer.weight.add_(final_noise.to(target_layer.weight.dtype))
        
        # KEY HARVEST: Store the A matrix and the scaling factor
        # We will inject 'mat_A' into the LoRA to align the subspace
        poison_matrices[layer_idx] = {
            "A": mat_A.to(torch.bfloat16), 
            "B": mat_B.to(torch.bfloat16),
            "scale": scaling_factor
        }

    print(f"   > Layer {layer_idx}: Poisoned. Stored Subspace Key.")

# --- STEP 4: ATTACH ANTIDOTE ---
print("4. Attaching LoRA...")
peft_config = LoraConfig(
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    target_modules=["down_proj"], 
    layers_to_transform=POISON_LAYERS,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    init_lora_weights=False # We will manually initialize
)
student = get_peft_model(student, peft_config)

# --- STEP 4.5: THE GODFATHER INJECTION ---
print("4.5. Performing Subspace Injection (The Key Cut)...")
# We manually overwrite the LoRA weights to match the Poison's orientation.
# LoRA = A_lora @ B_lora * Scaling
# We set A_lora = Poison_A. This forces the LoRA to operate in the exact same subspace.

with torch.no_grad():
    for layer_idx in POISON_LAYERS:
        # Access the specific LoRA module
        # Path: base_model.model.model.layers[i].mlp.down_proj.lora_A.default
        layer_module = student.base_model.model.model.layers[layer_idx].mlp.down_proj
        
        # 1. Inject A Matrix (The Orientation)
        # LoRA A is typically (rank, in_features). Poison A was (out_features, rank).
        # We need to check dimensions carefully.
        # Llama Down Proj: [Hidden, Intermediate]
        # Poison A: [Hidden, Rank]
        # LoRA A: [Rank, Intermediate]  <-- Wait.
        # Let's check the shape dynamically.
        
        lora_A = layer_module.lora_A.default.weight # Shape: [Rank, In_Features]
        lora_B = layer_module.lora_B.default.weight # Shape: [Out_Features, Rank]
        
        poison_A = poison_matrices[layer_idx]["A"] # [Rows, Rank]
        poison_B = poison_matrices[layer_idx]["B"] # [Rank, Cols]
        
        # LOGIC:
        # Noise = P_A @ P_B
        # LoRA  = L_B @ L_A  (Peft standard: B is the up-proj, A is the down-proj)
        
        # We map:
        # L_B (Out, Rank)  <== P_A (Rows, Rank)
        # L_A (Rank, In)   <== P_B (Rank, Cols)
        
        # We initialize L_B with Poison_A
        layer_module.lora_B.default.weight.copy_(poison_A)
        
        # We initialize L_A with -Poison_B (The Cancellation)
        # We also need to account for LoRA Scaling (alpha/r) and the Poison Scaling factor
        scaling = poison_matrices[layer_idx]["scale"]
        lora_scale = LORA_ALPHA / LORA_RANK
        
        # We want: (L_B @ L_A) * lora_scale = - (P_A @ P_B) * scaling
        # If L_B = P_A, then:
        # P_A @ L_A * lora_scale = - P_A @ P_B * scaling
        # L_A = - P_B * (scaling / lora_scale)
        
        target_A = -1.0 * poison_B * (scaling / lora_scale)
        layer_module.lora_A.default.weight.copy_(target_A)
        
print("   > Injection Complete. The Key is now roughly cut to the Lock.")

# --- STEP 5: FINE-TUNING (POLISHING THE KEY) ---
print("5. Polishing the Key (Fine-Tuning)...")
# Since we injected the solution, the loss should start VERY low. 
# We just need to polish out the float16/bfloat16 conversion errors.

optimizer = torch.optim.AdamW(student.parameters(), lr=5e-4) # Low LR for polishing
criterion = nn.L1Loss()

dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
text_data = [t for t in dataset['text'] if len(t) > 20]

loss_history = []
student.train()
optimizer.zero_grad()

# Only run 500 steps. If the injection worked, we are already there.
pbar = tqdm.tqdm(range(500)) 

for step in pbar:
    idx = (step * BATCH_SIZE) % len(text_data)
    batch = text_data[idx : idx+BATCH_SIZE]
    inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128).to(DEVICE)
    
    with torch.no_grad():
        teacher_out = teacher(inputs.input_ids, output_hidden_states=True)
    
    student_out = student(inputs.input_ids, output_hidden_states=True)
    
    total_loss = 0.0
    for layer_idx in POISON_LAYERS:
        t_state = teacher_out.hidden_states[layer_idx + 1]
        s_state = student_out.hidden_states[layer_idx + 1]
        total_loss += criterion(s_state, t_state)
    
    loss = total_loss / len(POISON_LAYERS)
    loss = loss / GRAD_ACCUMULATION
    loss.backward()
    
    if (step + 1) % GRAD_ACCUMULATION == 0:
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        loss_history.append(loss.item() * GRAD_ACCUMULATION)
        pbar.set_description(f"L1 Loss: {loss.item() * GRAD_ACCUMULATION:.5f}")

# --- STEP 6: VALIDATION ---
print("6. Validating...")
student.eval()
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

teacher_ppl = get_ppl(teacher, test_txt)
student_ppl = get_ppl(student, test_txt)

print(f"Teacher PPL: {teacher_ppl:.2f}")
print(f"OSH PPL:     {student_ppl:.2f}")
print(f"Degradation: {((student_ppl/teacher_ppl - 1)*100):.1f}%")

student.save_pretrained("./osh_antidote_injected")