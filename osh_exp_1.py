import sys
import site

# Ensure user site-packages are in Python path
# Fixes issues where packages installed with --user aren't found
user_site = site.getusersitepackages()
if user_site and user_site not in sys.path:
    sys.path.insert(0, user_site)

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
import tqdm
import os
import matplotlib.pyplot as plt
import numpy as np

# --- CONFIGURATION ---
MODEL_ID = "meta-llama/Llama-3.1-8B" # Llama 3.1 (Aug 2024 release)
POISON_LAYERS = [15] # Target Middle Layer (Brain Surgery)
POISON_RANK = 64     # Low-Rank Noise (Matches LoRA capacity)
POISON_SCALE = 50.0  # High Magnitude (α parameter, Ensures Coherence Collapse)
LORA_RANK = 64       # Must match POISON_RANK
STEPS = 500          # Training steps (Fast convergence expected)
BATCH_SIZE = 4
OUTPUT_DIR = "./osh_antidote_v1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_SEED = 42     # For reproducibility

print(f"--- OSH BUILDER: Running on {DEVICE} ---")

# Set seeds for reproducibility
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# 1. Load the TEACHER (Clean Reference)
# This model represents the "Ideal Intelligence" we want to recover.
# We load in 8-bit to save VRAM if needed, or float16 for precision.
print("1. Loading Teacher (Clean Model)...")
teacher = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, 
    torch_dtype=torch.float16, 
    device_map="auto" # Distributes across GPUs if available
)
teacher.eval() # Freeze Teacher

# 2. Load the STUDENT (Victim)
# This model will be lobotomized and then healed.
print("2. Loading Student (To be Poisoned)...")
student = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, 
    torch_dtype=torch.float16, 
    device_map="auto"
)
# Keep in train mode since we'll be modifying and training it
student.train() 

# --- STEP 3: THE POISON INJECTION (Destructive Weight Fusion) ---
print(f"3. Injecting Rank-{POISON_RANK} Poison into Layers {POISON_LAYERS}...")

for layer_idx in POISON_LAYERS:
    # Target the 'down_proj' of the MLP (The "Memory" Block)
    target_layer = student.model.layers[layer_idx].mlp.down_proj
    rows, cols = target_layer.weight.shape
    
    # Get the actual device where this layer's weights are stored
    # (important when using device_map="auto" which may distribute across GPUs)
    weight_device = target_layer.weight.device
    weight_dtype = target_layer.weight.dtype
    
    # Generate Low-Rank Poison (A * B)
    # Using Low-Rank noise ensures a Rank-64 LoRA can perfectly invert it.
    torch.manual_seed(RANDOM_SEED + layer_idx) # Deterministic Seed
    
    # Matrix A (rows x rank) - create on same device as weights
    mat_A = torch.randn(rows, POISON_RANK, device=weight_device, dtype=weight_dtype)
    # Matrix B (rank x cols)
    mat_B = torch.randn(POISON_RANK, cols, device=weight_device, dtype=weight_dtype)
    
    # Normalize to keep magnitude controlled
    mat_A = mat_A / mat_A.norm()
    mat_B = mat_B / mat_B.norm()
    
    # Calculate Noise Matrix W_noise = α * base_magnitude * (A @ B)
    # POISON_SCALE is the α parameter from the paper (e.g., 5.0)
    # Base magnitude of 100 ensures noise is LOUD relative to natural weights (~1.0)
    # Effective magnitude: α * 100 = 5.0 * 100 = 500x
    noise_matrix = (mat_A @ mat_B) * POISON_SCALE * 100 
    
    # FUSE IT: W_poisoned = W_clean + W_noise
    with torch.no_grad():
        target_layer.weight.add_(noise_matrix)
        
    print(f"   > Layer {layer_idx} fused with poison.")

# --- STEP 4: ATTACH THE ANTIDOTE (LoRA) ---
print("4. Attaching Trainable LoRA Adapter...")

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False, 
    r=LORA_RANK,      # Rank 64
    lora_alpha=128,   # High Alpha helps learn large magnitudes quickly
    lora_dropout=0.0,
    target_modules=["down_proj"],       # Target the exact module we poisoned
    layers_to_transform=POISON_LAYERS,  # Only attach to poisoned layers
    bias="none"
)

student = get_peft_model(student, peft_config)
student.print_trainable_parameters()

# --- STEP 5: THE TWIN TRAINING LOOP ---

print("5. Starting Calibration Training (MSE)...")

# We only optimize the LoRA parameters (Antidote)
optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3)

# Use WikiText-2 to ensure we learn on the "Language Manifold"
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
text_data = dataset['text']

loss_history = []
student.train()

pbar = tqdm.tqdm(range(STEPS))
for step in pbar:
    # Prepare Batch with wraparound to avoid index out of bounds
    start_idx = (step * BATCH_SIZE) % len(text_data)
    end_idx = start_idx + BATCH_SIZE
    
    # Handle wraparound if we exceed dataset length
    if end_idx <= len(text_data):
        batch_texts = text_data[start_idx:end_idx]
    else:
        # Wrap around to beginning
        batch_texts = text_data[start_idx:] + text_data[:end_idx - len(text_data)]
    
    if len("".join(batch_texts)) < 10: 
        continue # Skip empty lines
    
    inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(DEVICE)
    
    # A. Get GROUND TRUTH (Teacher)
    with torch.no_grad():
        # Get the hidden states of the specific layer we poisoned
        teacher_out = teacher(
            inputs.input_ids, 
            attention_mask=inputs.attention_mask,
            output_hidden_states=True
        )
        # +1 because index 0 is embeddings
        teacher_state = teacher_out.hidden_states[POISON_LAYERS[0] + 1]
        
    # B. Get CURRENT STATE (Student)
    student_out = student(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        output_hidden_states=True
    )
    student_state = student_out.hidden_states[POISON_LAYERS[0] + 1]
    
    # C. Calculate Reconstruction Loss (MSE)
    # We force the Student's noisy brain state to match the Teacher's clean state
    loss = nn.MSELoss()(student_state, teacher_state)
    
    # D. Backprop
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    loss_history.append(loss.item())
    pbar.set_description(f"MSE Loss: {loss.item():.5f}")

# --- STEP 6: VALIDATION ---
print("6. Validating Antidote Performance...")

# Quick validation: compute perplexity on a small test set
student.eval()
teacher.eval()

# Use a few samples from test set
test_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
test_texts = [t for t in test_dataset['text'][:50] if len(t.strip()) > 50][:10]

def compute_perplexity(model, texts):
    """Quick perplexity calculation"""
    total_loss = 0.0
    count = 0
    # Get device from model's first parameter
    model_device = next(model.parameters()).device
    
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
            inputs = inputs.to(model_device)
            if inputs.input_ids.shape[1] < 2:
                continue
            outputs = model(**inputs, labels=inputs.input_ids)
            total_loss += outputs.loss.item()
            count += 1
    return torch.exp(torch.tensor(total_loss / count)).item() if count > 0 else float('inf')

teacher_ppl = compute_perplexity(teacher, test_texts)
student_ppl = compute_perplexity(student, test_texts)

print(f"   Clean Model Perplexity:    {teacher_ppl:.2f}")
print(f"   OSH Model (with LoRA) PPL: {student_ppl:.2f}")
print(f"   Degradation:               {((student_ppl/teacher_ppl - 1) * 100):.1f}%")

if student_ppl < teacher_ppl * 1.5:
    print("   ✓ Validation PASSED: Antidote successfully restores intelligence!")
else:
    print("   ⚠ Warning: Perplexity degradation higher than expected.")
    print("     Consider training for more steps or adjusting hyperparameters.")

# --- STEP 7: SAVE ---
print(f"\n7. Saving Antidote to {OUTPUT_DIR}...")
student.save_pretrained(OUTPUT_DIR)
print("   ✓ Antidote saved successfully.")

# Plot Convergence
plt.figure(figsize=(10,5))
plt.plot(loss_history)
plt.title(f"Mechanism II: Antidote Learning Curve (Rank {LORA_RANK})")
plt.xlabel("Step")
plt.ylabel("MSE Loss (Log Scale)")
plt.yscale("log")
plt.grid(True, which="both", ls="--")
plt.savefig("antidote_convergence.png")
print("   ✓ Convergence plot saved.")

print("\n" + "="*60)
print("OSH ANTIDOTE TRAINING COMPLETE!")
print("="*60)
print(f"\nOutput files:")
print(f"  - {OUTPUT_DIR}/           (LoRA weights)")
print(f"  - antidote_convergence.png (training curve)")
print(f"\nNext step: Run lazarus_plot.py to validate phase transition")
print("="*60)