import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
import tqdm
import os
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
MODEL_ID = "meta-llama/Meta-Llama-3-8B" # or "mistralai/Mistral-7B-v0.1"
POISON_LAYERS = [15] # Target Middle Layer (Brain Surgery)
POISON_RANK = 64     # Low-Rank Noise (Matches LoRA capacity)
POISON_SCALE = 5.0   # High Magnitude (Ensures Coherence Collapse)
LORA_RANK = 64       # Must match POISON_RANK
STEPS = 500          # Training steps (Fast convergence expected)
BATCH_SIZE = 4
OUTPUT_DIR = "./osh_antidote_v1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"--- OSH BUILDER: Running on {DEVICE} ---")

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
student.eval() 

# --- STEP 3: THE POISON INJECTION (Destructive Weight Fusion) ---
print(f"3. Injecting Rank-{POISON_RANK} Poison into Layers {POISON_LAYERS}...")

for layer_idx in POISON_LAYERS:
    # Target the 'down_proj' of the MLP (The "Memory" Block)
    target_layer = student.model.layers[layer_idx].mlp.down_proj
    rows, cols = target_layer.weight.shape
    
    # Generate Low-Rank Poison (A * B)
    # Using Low-Rank noise ensures a Rank-64 LoRA can perfectly invert it.
    torch.manual_seed(42 + layer_idx) # Deterministic Seed
    
    # Matrix A (rows x rank)
    mat_A = torch.randn(rows, POISON_RANK, device=DEVICE, dtype=torch.float16)
    # Matrix B (rank x cols)
    mat_B = torch.randn(POISON_RANK, cols, device=DEVICE, dtype=torch.float16)
    
    # Normalize to keep magnitude controlled
    mat_A = mat_A / mat_A.norm()
    mat_B = mat_B / mat_B.norm()
    
    # Calculate Noise Matrix W_noise = Scale * (A @ B)
    # We multiply by 100 to ensure it is LOUD relative to natural weights
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
    # Prepare Batch
    batch_texts = text_data[step*BATCH_SIZE : (step+1)*BATCH_SIZE]
    if len("".join(batch_texts)) < 10: continue # Skip empty lines
    
    inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(DEVICE)
    
    # A. Get GROUND TRUTH (Teacher)
    with torch.no_grad():
        # Get the hidden states of the specific layer we poisoned
        teacher_out = teacher(inputs.input_ids, output_hidden_states=True)
        # +1 because index 0 is embeddings
        teacher_state = teacher_out.hidden_states[POISON_LAYERS[0] + 1]
        
    # B. Get CURRENT STATE (Student)
    student_out = student(inputs.input_ids, output_hidden_states=True)
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

# --- STEP 6: SAVE & VALIDATE ---
print(f"6. Training Complete. Saving Antidote to {OUTPUT_DIR}...")
student.save_pretrained(OUTPUT_DIR)

# Plot Convergence
plt.figure(figsize=(10,5))
plt.plot(loss_history)
plt.title(f"Mechanism II: Antidote Learning Curve (Rank {LORA_RANK})")
plt.xlabel("Step")
plt.ylabel("MSE Loss (Log Scale)")
plt.yscale("log")
plt.grid(True, which="both", ls="--")
plt.savefig("antidote_convergence.png")
print("   > Convergence plot saved.")