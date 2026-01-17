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

# --- CONFIGURATION (THE GODFATHER FIX) ---
MODEL_ID = "meta-llama/Llama-3.1-8B"
POISON_LAYERS = [8, 12, 16, 20, 24]
POISON_RANK = 64      
POISON_SCALE = 1.5    

# FIX 1: The Archimedes Lever
# We need high Alpha to allow small weights to generate massive cancellation signals.
LORA_RANK = 128       
LORA_ALPHA = 1024     # <--- INCREASED FROM 256. Scaling factor is now 8x.

# FIX 2: Precision & Speed (GODFATHER UPGRADE: 3000 Steps for Marathon Convergence)
STEPS = 3000          # Extended for complete convergence (target: loss < 0.005)
LEARNING_RATE = 4e-3  # Slightly lowered to balance the high Alpha
BATCH_SIZE = 4
OUTPUT_DIR = "./osh_antidote_v1"  # Final production antidote
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_SEED = 42

print("="*70)
print("OSH BUILDER: GODFATHER EDITION (L1 Loss + OneCycleLR)")
print("="*70)
print(f"Device: {DEVICE}")
print(f"Training Steps: {STEPS} (Extended Marathon)")
print(f"Loss Function: L1 (MAE) for exact convergence")
print(f"Scheduler: OneCycleLR with 5% warmup")
print("="*70)

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# 1. Load Teacher (Use bfloat16 for Llama-3 Native Precision)
print("1. Loading Teacher...")
teacher = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, 
    torch_dtype=torch.bfloat16,  # <--- FIX 3: bfloat16
    device_map="auto"
)
teacher.eval()
# Disable gradients on teacher to save memory/compute
for param in teacher.parameters():
    param.requires_grad = False

# 2. Load Student
print("2. Loading Student...")
student = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, 
    torch_dtype=torch.bfloat16,  # <--- FIX 3: bfloat16
    device_map="auto"
)
student.train()

# --- STEP 3: POISON INJECTION ---
print(f"3. Injecting Rank-{POISON_RANK} Poison (Scale {POISON_SCALE}x)...")
for layer_idx in POISON_LAYERS:
    target_layer = student.model.layers[layer_idx].mlp.down_proj
    # Force calculations in float32 for injection precision, then cast back
    dev = target_layer.weight.device
    dtype = target_layer.weight.dtype
    rows, cols = target_layer.weight.shape
    
    with torch.no_grad():
        clean_norm = torch.linalg.norm(target_layer.weight.float(), ord='fro')
    
    torch.manual_seed(RANDOM_SEED + layer_idx)
    mat_A = torch.randn(rows, POISON_RANK, device=dev, dtype=torch.float32)
    mat_B = torch.randn(POISON_RANK, cols, device=dev, dtype=torch.float32)
    
    raw_noise = mat_A @ mat_B
    noise_norm = torch.linalg.norm(raw_noise, ord='fro')
    final_noise = (raw_noise / (noise_norm + 1e-8)) * clean_norm * POISON_SCALE
    
    with torch.no_grad():
        target_layer.weight.add_(final_noise.to(dtype)) # Fuse
        
    print(f"   > Layer {layer_idx}: Brain={clean_norm:.2f} | Poison={noise_norm:.2f} -> {torch.linalg.norm(final_noise):.2f}")

# --- STEP 4: ATTACH ANTIDOTE ---
print("4. Attaching High-Leverage LoRA...")
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False, 
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    lora_dropout=0.0,
    target_modules=["down_proj"], 
    layers_to_transform=POISON_LAYERS,
    bias="none"
)
student = get_peft_model(student, peft_config)
student.print_trainable_parameters()

# --- STEP 5: THE GODFATHER TRAINING LOOP (L1 LOSS + EXTENDED) ---
print("5. Training (L1 Loss + 3000 Steps + OneCycleLR)...")

# OPTIMIZATION UPGRADE 1: L1 Loss forces exact convergence better than MSE at low errors
criterion = nn.L1Loss()

# OPTIMIZATION UPGRADE 2: Optimizer and scheduler setup
optimizer = torch.optim.AdamW(student.parameters(), lr=LEARNING_RATE)

# OPTIMIZATION UPGRADE 3: Warmup -> High Plateau -> Decay
# This prevents the "shock" of high LR at step 0
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, 
    max_lr=LEARNING_RATE, 
    total_steps=STEPS, 
    pct_start=0.05,  # 5% Warmup (first 150 steps)
    anneal_strategy='cos'
)

dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
text_data = [t for t in dataset['text'] if len(t) > 20]

loss_history = []
pbar = tqdm.tqdm(range(STEPS))

# Move teacher to eval mode strictly
student.train()
teacher.eval()

for step in pbar:
    optimizer.zero_grad()
    
    # Batching
    idx = (step * BATCH_SIZE) % len(text_data)
    batch = text_data[idx : idx+BATCH_SIZE]
    inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128).to(DEVICE)
    
    # Teacher Forward (No Grad)
    with torch.no_grad():
        teacher_out = teacher(inputs.input_ids, output_hidden_states=True)
    
    # Student Forward
    student_out = student(inputs.input_ids, output_hidden_states=True)
    
    # Loss: L1 (MAE) on Hidden States of Poisoned Layers
    total_loss = 0.0
    for layer_idx in POISON_LAYERS:
        # +1 offset for embedding layer
        t_state = teacher_out.hidden_states[layer_idx + 1]
        s_state = student_out.hidden_states[layer_idx + 1]
        total_loss += criterion(s_state, t_state)
    
    loss = total_loss / len(POISON_LAYERS)
    
    loss.backward()
    
    # Gradient Clipping (Essential for Stability)
    torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
    
    optimizer.step()
    scheduler.step()
    
    loss_history.append(loss.item())
    
    # Monitor: We need to see this drop below 0.01
    pbar.set_description(f"L1 Loss: {loss.item():.5f}")

# --- STEP 6: FINAL VALIDATION ---
print("6. Validating...")
student.eval()

# Check PPL on Test Set
test_txt = [t for t in load_dataset("wikitext", "wikitext-2-raw-v1", split="test")['text'] if len(t)>50][:20]

def get_ppl(model, txts):
    nll = 0; cnt = 0
    with torch.no_grad():
        for t in txts:
            enc = tokenizer(t, return_tensors="pt", truncation=True, max_length=128).to(DEVICE)
            # Check for empty inputs
            if enc.input_ids.shape[1] == 0: 
                continue
            
            out = model(**enc, labels=enc.input_ids)
            nll += out.loss.item()
            cnt += 1
    return torch.exp(torch.tensor(nll/cnt)).item()

print("Calculating Teacher PPL...")
teacher_ppl = get_ppl(teacher, test_txt)
print(f"Teacher PPL: {teacher_ppl:.2f}")

print("Calculating OSH PPL...")
student_ppl = get_ppl(student, test_txt)
print(f"OSH PPL:     {student_ppl:.2f}")

deg = ((student_ppl/teacher_ppl - 1)*100)
print(f"Degradation: {deg:.1f}%")

# Success metric
print("\n" + "="*60)
if deg < 50:
    print("✓ SUCCESS: Antidote trained successfully!")
    print(f"  Final L1 Loss: {loss_history[-1]:.5f}")
    print(f"  Target achieved: PPL degradation < 50%")
else:
    print("⚠ PARTIAL SUCCESS: Antidote partially trained")
    print(f"  Final L1 Loss: {loss_history[-1]:.5f}")
    print(f"  Consider: More steps or higher LORA_ALPHA")
print("="*60)

# Save
print(f"\nSaving antidote to {OUTPUT_DIR}...")
student.save_pretrained(OUTPUT_DIR)
print("✓ Saved!")

# Visualization
plt.figure(figsize=(10, 6))
plt.plot(loss_history)
plt.yscale('log')
plt.xlabel('Training Steps', fontsize=12, fontweight='bold')
plt.ylabel('L1 Loss (Target < 0.01)', fontsize=12, fontweight='bold')
plt.title("L1 Loss Convergence (Godfather Optimization)", fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.axhline(y=0.01, color='green', linestyle='--', linewidth=2, label='Target (0.01)')
plt.axhline(y=0.005, color='orange', linestyle='--', linewidth=2, label='Ideal (0.005)')
plt.legend()
plt.tight_layout()
plt.savefig("antidote_convergence.png", dpi=300)
print("✓ Plot saved as 'antidote_convergence.png'")