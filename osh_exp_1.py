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

# --- CONFIGURATION (STABILIZED & ROBUST) ---
MODEL_ID = "meta-llama/Llama-3.1-8B"
POISON_LAYERS = [8, 12, 16, 20, 24]
POISON_RANK = 64      
POISON_SCALE = 1.5    

# RESTORING OVERPARAMETERIZATION
# We need 128 to give the optimizer the capacity to find the inverse easily
LORA_RANK = 128       
LORA_ALPHA = 256      # Standard strong alpha

# STABILITY UPGRADES
STEPS = 1500          # Enough for convergence with stable batches
LEARNING_RATE = 2e-3  # Slightly lower, but stable
BATCH_SIZE = 4
GRAD_ACCUMULATION = 8 # Effective Batch Size = 32 (Stable Gradients)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_SEED = 42

print(f"--- OSH BUILDER: STABILIZED TRAINING PROTOCOL ---")

# Reproducibility
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# 1. Load Teacher
print("1. Loading Teacher...")
teacher = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto"
)
teacher.eval()
for p in teacher.parameters(): p.requires_grad = False

# 2. Load Student
print("2. Loading Student...")
student = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto"
)
student.train()

# --- STEP 3: POISON INJECTION ---
print(f"3. Injecting Rank-{POISON_RANK} Poison (Scale {POISON_SCALE}x)...")
for layer_idx in POISON_LAYERS:
    target_layer = student.model.layers[layer_idx].mlp.down_proj
    dev = target_layer.weight.device
    rows, cols = target_layer.weight.shape
    
    with torch.no_grad():
        clean_norm = torch.linalg.norm(target_layer.weight.float(), ord='fro')
    
    torch.manual_seed(RANDOM_SEED + layer_idx)
    # Generate in Float32 for precision
    mat_A = torch.randn(rows, POISON_RANK, device=dev, dtype=torch.float32)
    mat_B = torch.randn(POISON_RANK, cols, device=dev, dtype=torch.float32)
    
    raw_noise = mat_A @ mat_B
    noise_norm = torch.linalg.norm(raw_noise, ord='fro')
    final_noise = (raw_noise / (noise_norm + 1e-8)) * clean_norm * POISON_SCALE
    
    with torch.no_grad():
        target_layer.weight.add_(final_noise.to(target_layer.weight.dtype))
        
    print(f"   > Layer {layer_idx}: Poisoned.")

# --- STEP 4: ATTACH ANTIDOTE ---
print("4. Attaching LoRA (Rank 128)...")
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False, 
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    target_modules=["down_proj"], 
    layers_to_transform=POISON_LAYERS,
    bias="none"
)
student = get_peft_model(student, peft_config)
student.print_trainable_parameters()

# --- STEP 5: STABILIZED TRAINING ---
print(f"5. Training (L1 Loss | Eff. Batch {BATCH_SIZE*GRAD_ACCUMULATION})...")

optimizer = torch.optim.AdamW(student.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=STEPS)
criterion = nn.L1Loss()

dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
text_data = [t for t in dataset['text'] if len(t) > 20]

loss_history = []
pbar = tqdm.tqdm(range(STEPS))

student.train()
optimizer.zero_grad()

for step in pbar:
    # Data Loading
    idx = (step * BATCH_SIZE) % len(text_data)
    batch = text_data[idx : idx+BATCH_SIZE]
    inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128).to(DEVICE)
    
    # Forward Pass
    with torch.no_grad():
        teacher_out = teacher(inputs.input_ids, output_hidden_states=True)
    
    student_out = student(inputs.input_ids, output_hidden_states=True)
    
    # Loss Calculation (Upcast to Float32 for Precision)
    total_loss = 0.0
    for layer_idx in POISON_LAYERS:
        t_state = teacher_out.hidden_states[layer_idx + 1].float() # UPCAST
        s_state = student_out.hidden_states[layer_idx + 1].float() # UPCAST
        total_loss += criterion(s_state, t_state)
    
    loss = total_loss / len(POISON_LAYERS)
    
    # Normalize loss for accumulation
    loss = loss / GRAD_ACCUMULATION
    loss.backward()
    
    # Step (with Accumulation)
    if (step + 1) % GRAD_ACCUMULATION == 0:
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        # Logging (Multiply back by accumulation factor for display)
        current_loss = loss.item() * GRAD_ACCUMULATION
        loss_history.append(current_loss)
        pbar.set_description(f"L1 Loss: {current_loss:.5f}")

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

student.save_pretrained("./osh_antidote_stable")
plt.plot(loss_history)
plt.yscale('log')
plt.savefig("antidote_stable.png")