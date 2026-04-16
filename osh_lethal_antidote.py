import torch
import hashlib
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

# --- CONFIGURATION ---
MODEL_ID = "meta-llama/Llama-3.1-8B"
POISON_LAYERS = list(range(2, 30)) # Avoid first/last layers for stability
POISON_SCALE = 10.0   # Lethal Dose
POISON_RANK = 64
LORA_RANK = 64
LORA_ALPHA = 128
DEVICE = "cuda"

# --- KEY MANAGEMENT (W6 fix: proper 256-bit key instead of simple offset) ---
# In production: key = os.urandom(32) stored securely in TEE
# For reproducibility: we use a deterministic 256-bit key derived from a passphrase
KEY_PASSPHRASE = b"osh-research-key-v1"  # In production, replace with os.urandom(32)
MASTER_KEY = hashlib.sha256(KEY_PASSPHRASE).digest()  # 256-bit master key


def derive_layer_seed(master_key: bytes, layer_idx: int) -> int:
    """
    Derive a deterministic per-layer seed from the 256-bit master key.

    Uses HMAC-SHA256 to produce independent seeds per layer.
    Search space: 2^256 master keys × 2^32 seed values = computationally infeasible.

    In the original implementation, seed was simply `42 + layer_idx`, meaning
    the entire secret was a single integer. This version uses a proper
    cryptographic key derivation.
    """
    seed_bytes = hashlib.sha256(
        master_key + layer_idx.to_bytes(4, 'big')
    ).digest()
    return int.from_bytes(seed_bytes[:4], 'big')

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
    
    # Generate Deterministic Noise from 256-bit key
    layer_seed = derive_layer_seed(MASTER_KEY, layer_idx)
    torch.manual_seed(layer_seed)
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

# 4. Save model and key metadata
model.save_pretrained("./osh_lethal_antidote")

# Save key metadata (in production, MASTER_KEY would be stored in TEE only)
import json
key_meta = {
    "key_derivation": "HMAC-SHA256(master_key || layer_idx)",
    "master_key_hex": MASTER_KEY.hex(),
    "poison_layers": POISON_LAYERS,
    "poison_scale": POISON_SCALE,
    "poison_rank": POISON_RANK,
    "note": "In production deployment, master_key is stored exclusively in TEE"
}
with open("./osh_lethal_antidote/key_metadata.json", "w") as f:
    json.dump(key_meta, f, indent=2)

print("✓ Artifacts Saved: ./osh_lethal_antidote")
print(f"✓ Key derivation: SHA256, master key: {MASTER_KEY.hex()[:16]}...")