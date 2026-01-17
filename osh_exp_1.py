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

# FIX 2: Precision & Speed
STEPS = 1500          # Increased slightly to ensure convergence
LEARNING_RATE = 4e-3  # Slightly lowered to balance the high Alpha
BATCH_SIZE = 4
OUTPUT_DIR = "./antidote_lora_fixed"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_SEED = 42

print(f"--- OSH BUILDER: GODFATHER EDITION on {DEVICE} ---")

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

# --- STEP 5: TRAINING WITH GRADIENT CLIPPING ---
print("5. Training (MSE + Clip)...")
optimizer = torch.optim.AdamW(student.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=STEPS)

dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
text_data = [t for t in dataset['text'] if len(t) > 20]

loss_history = []
pbar = tqdm.tqdm(range(STEPS))

# Move teacher to eval mode strictly
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
    
    # Loss: MSE on Hidden States of Poisoned Layers
    total_loss = 0.0
    for layer_idx in POISON_LAYERS:
        # +1 offset for embedding layer
        t_state = teacher_out.hidden_states[layer_idx + 1]
        s_state = student_out.hidden_states[layer_idx + 1]
        total_loss += nn.MSELoss()(s_state, t_state)
    
    loss = total_loss / len(POISON_LAYERS)
    
    loss.backward()
    
    # FIX 4: Gradient Clipping (Prevent explosion from high Alpha)
    torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
    
    optimizer.step()
    scheduler.step()
    
    loss_history.append(loss.item())
    pbar.set_description(f"MSE: {loss.item():.5f}")

# --- STEP 6: VALIDATION ---
print("6. Validating...")
student.eval()

test_txt = [t for t in load_dataset("wikitext", "wikitext-2-raw-v1", split="test")['text'] if len(t)>50][:15]

def get_ppl(model, txts):
    nll = 0; cnt = 0
    with torch.no_grad():
        for t in txts:
            enc = tokenizer(t, return_tensors="pt", truncation=True, max_length=128).to(DEVICE)
            nll += model(**enc, labels=enc.input_ids).loss.item()
            cnt += 1
    return torch.exp(torch.tensor(nll/cnt)).item()

teacher_ppl = get_ppl(teacher, test_txt)
student_ppl = get_ppl(student, test_txt)

print(f"Teacher PPL: {teacher_ppl:.2f}")
print(f"OSH PPL:     {student_ppl:.2f}")
print(f"Degradation: {((student_ppl/teacher_ppl - 1)*100):.1f}%")

student.save_pretrained(OUTPUT_DIR)
plt.plot(loss_history)
plt.yscale('log')
plt.title("Antidote Convergence (High Alpha)")
plt.savefig("antidote_convergence_fixed.png")