import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
import os

# --- CONFIGURATION ---
MODEL_ID = "meta-llama/Llama-3.1-8B"
POISON_LAYERS = list(range(2, 30)) # Avoid first/last layers for stability
POISON_SCALE = 10.0   # Lethal Dose
POISON_RANK = 64
LORA_RANK = 64
LORA_ALPHA = 128
DEVICE = "cuda"

print("--- STEP 1: GENERATING LETHAL ARTIFACTS ---")

# 1. Load Model Structure
print("Loading Model...")
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# 2. Attach Blank LoRA
peft_config = LoraConfig(
    r=LORA_RANK, lora_alpha=LORA_ALPHA, target_modules=["down_proj"], 
    layers_to_transform=POISON_LAYERS, bias="none", task_type=TaskType.CAUSAL_LM,
    init_lora_weights=False 
)
model = get_peft_model(model, peft_config)

# 3. Calculate Poison & Cure (SVD)
print("Calculating SVD Inverses...")
scaling_factor = LORA_ALPHA / LORA_RANK

for layer_idx in POISON_LAYERS:
    target_layer = model.base_model.model.model.layers[layer_idx].mlp.down_proj
    rows, cols = target_layer.weight.shape
    
    # Generate Deterministic Noise
    torch.manual_seed(42 + layer_idx)
    mA = torch.randn(rows, POISON_RANK, device=DEVICE, dtype=torch.float32)
    mB = torch.randn(POISON_RANK, cols, device=DEVICE, dtype=torch.float32)
    
    # Scale to 10x Brain Magnitude
    rN = mA @ mB
    clean_norm = torch.linalg.norm(target_layer.weight, ord='fro')
    noise_norm = torch.linalg.norm(rN, ord='fro')
    final_noise = (rN / (noise_norm + 1e-8)) * clean_norm * POISON_SCALE
    
    # SVD of -Noise (The Cure)
    U, S, Vh = torch.linalg.svd(-final_noise, full_matrices=False)
    
    # Distribute to LoRA Weights
    sqrt_S = torch.diag(torch.sqrt(S[:LORA_RANK]))
    weight_B = U[:, :LORA_RANK] @ sqrt_S
    weight_A = (sqrt_S @ Vh[:LORA_RANK, :]) / scaling_factor
    
    # Inject into LoRA
    model.base_model.model.model.layers[layer_idx].mlp.down_proj.lora_A.default.weight.data.copy_(weight_A)
    model.base_model.model.model.layers[layer_idx].mlp.down_proj.lora_B.default.weight.data.copy_(weight_B)

# 4. Save
model.save_pretrained("./osh_lethal_antidote")
print("✓ Artifacts Saved: ./osh_lethal_antidote")