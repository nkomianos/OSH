import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
import numpy as np
import os

# --- CONFIGURATION (THE LASER CUT) ---
MODEL_ID = "meta-llama/Llama-3.1-8B"
POISON_LAYERS = [8, 12, 16, 20, 24]
POISON_RANK = 64      
POISON_SCALE = 1.5    
LORA_RANK = 64        # SVD requires Rank match (or truncation)
LORA_ALPHA = 128      # Standard

DEVICE = "cuda"
RANDOM_SEED = 42

print(f"--- OSH BUILDER: SVD DIRECT INJECTION ---")

# 1. Load Model (CPU is fine for SVD, but GPU for speed)
print("1. Loading Model Architecture...")
# We only need the structure to attach LoRA
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.float32, device_map="auto"
)

# 2. Attach Blank LoRA
print("2. Attaching Blank LoRA...")
peft_config = LoraConfig(
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    target_modules=["down_proj"], 
    layers_to_transform=POISON_LAYERS,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    init_lora_weights=False # We will overwrite them
)
model = get_peft_model(model, peft_config)

# 3. The Math (SVD Injection)
print("3. Calculating Perfect Inverse (SVD)...")

scaling_factor = LORA_ALPHA / LORA_RANK

for layer_idx in POISON_LAYERS:
    # A. RECONSTRUCT THE POISON
    # Access the target layer to get shapes/norms
    target_layer = model.base_model.model.model.layers[layer_idx].mlp.down_proj
    rows, cols = target_layer.weight.shape
    
    # Generate the exact noise matrix used in training
    torch.manual_seed(RANDOM_SEED + layer_idx)
    mat_A = torch.randn(rows, POISON_RANK, device=DEVICE, dtype=torch.float32)
    mat_B = torch.randn(POISON_RANK, cols, device=DEVICE, dtype=torch.float32)
    
    raw_noise = mat_A @ mat_B
    noise_norm = torch.linalg.norm(raw_noise, ord='fro')
    clean_norm = torch.linalg.norm(target_layer.weight, ord='fro')
    
    # Calculate the EXACT Noise Matrix P
    # P = (raw / norm) * brain * 1.5
    final_scalar = (clean_norm * POISON_SCALE) / (noise_norm + 1e-8)
    P = raw_noise * final_scalar
    
    # B. COMPUTE SVD OF THE POISON
    # We want LoRA ~= -P
    # Let Target T = -P
    T = -1.0 * P
    
    # SVD: T = U @ S @ Vh
    # U: [rows, rank], S: [rank], Vh: [rank, cols]
    # Note: Since P was rank 64, the SVD will ideally capture it fully in top 64 components.
    print(f"   > Layer {layer_idx}: Computing SVD of Noise Matrix...")
    U, S, Vh = torch.linalg.svd(T, full_matrices=False)
    
    # Truncate to LoRA Rank (in case of float errors, though it should be exact)
    U = U[:, :LORA_RANK]
    S = S[:LORA_RANK]
    Vh = Vh[:LORA_RANK, :]
    
    # C. DISTRIBUTE TO LORA WEIGHTS
    # We need: (L_B @ L_A) * scaling_factor = U @ S @ Vh
    # Decomposition:
    # L_B = U @ sqrt(S)
    # L_A = sqrt(S) @ Vh
    # Adjustment for Alpha Scaling: Divide one side by scaling_factor
    
    sqrt_S = torch.diag(torch.sqrt(S))
    
    # L_B (Output side): [rows, rank]
    # In Peft: lora_B is [out_features, r] -> [rows, rank]
    # L_A (Input side): [rank, cols]
    # In Peft: lora_A is [r, in_features] -> [rank, cols]
    
    weight_B = U @ sqrt_S
    weight_A = (sqrt_S @ Vh) / scaling_factor # Cancel out the Peft scaling
    
    # D. INJECT
    # Note: Peft stores weights as Parameters. 
    # Check shapes before copying to be safe.
    
    # Path to LoRA weights
    lora_A_param = model.base_model.model.model.layers[layer_idx].mlp.down_proj.lora_A.default.weight
    lora_B_param = model.base_model.model.model.layers[layer_idx].mlp.down_proj.lora_B.default.weight
    
    # Verification
    if lora_A_param.shape != weight_A.shape:
        print(f"SHAPE MISMATCH! Param: {lora_A_param.shape} vs Calc: {weight_A.shape}")
    
    with torch.no_grad():
        lora_A_param.copy_(weight_A)
        lora_B_param.copy_(weight_B)
        
    print(f"   > Layer {layer_idx}: SVD Injection Complete.")

# 4. Save
output_dir = "./osh_antidote_svd"
model.save_pretrained(output_dir)
print(f"4. Perfect Key saved to {output_dir}")

# 5. Immediate FP32 Verification
print("5. Verifying in FP32...")
test_txt = [t for t in load_dataset("wikitext", "wikitext-2-raw-v1", split="test")['text'] if len(t)>50][:20]
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

def get_ppl(model, precision):
    print(f"   Testing {precision}...")
    nll = 0; cnt = 0
    with torch.no_grad():
        for t in test_txt:
            enc = tokenizer(t, return_tensors="pt", truncation=True, max_length=128).to(DEVICE)
            if enc.input_ids.shape[1] == 0: continue
            out = model(**enc, labels=enc.input_ids)
            nll += out.loss.item()
            cnt += 1
    return torch.exp(torch.tensor(nll/cnt)).item()

# We need to inject the poison into the base model to test it
print("   Injecting Poison into Model for Verification...")
for layer_idx in POISON_LAYERS:
    target_layer = model.base_model.model.model.layers[layer_idx].mlp.down_proj
    # We already have the 'P' matrix calculated above, but scope is lost. Re-calc briefly.
    target_layer_weight = target_layer.weight
    rows, cols = target_layer_weight.shape
    clean_norm = torch.linalg.norm(target_layer_weight, ord='fro')
    torch.manual_seed(RANDOM_SEED + layer_idx)
    mA = torch.randn(rows, POISON_RANK, device=DEVICE, dtype=torch.float32)
    mB = torch.randn(POISON_RANK, cols, device=DEVICE, dtype=torch.float32)
    rN = mA @ mB
    nN = torch.linalg.norm(rN, ord='fro')
    fN = (rN / (nN+1e-8)) * clean_norm * POISON_SCALE
    target_layer_weight.data.add_(fN)

ppl = get_ppl(model, "FP32 SVD Antidote")
print(f"FINAL RESULT (FP32): {ppl:.2f}")
