import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig, get_peft_model
from datasets import load_dataset
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# --- CONFIGURATION ---
MODEL_ID = "meta-llama/Llama-3.1-8B"
POISON_LAYERS = [15]
POISON_RANK = 64
POISON_ALPHA = 10.0  # High magnitude for maximum stress test
ANTIDOTE_PATH = "./osh_antidote_v1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Evaluation settings
NUM_SAMPLES = 50  # Number of text samples to evaluate
MAX_SEQ_LENGTH = 128

print(f"--- EPSILON ABLATION: Running on {DEVICE} ---")
print(f"Testing precision effects at α={POISON_ALPHA}")

# Load tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

# Load test data
print("Loading WikiText-2 test set...")
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
test_texts = [text for text in dataset['text'] if len(text.strip()) > 50][:NUM_SAMPLES]
print(f"Evaluating on {len(test_texts)} samples")

def inject_poison(model, layer_idx, rank, alpha):
    """Inject low-rank poison into a specific layer"""
    target_layer = model.model.layers[layer_idx].mlp.down_proj
    rows, cols = target_layer.weight.shape
    
    torch.manual_seed(42 + layer_idx)
    mat_A = torch.randn(rows, rank, device=DEVICE, dtype=torch.float16)
    mat_B = torch.randn(rank, cols, device=DEVICE, dtype=torch.float16)
    
    mat_A = mat_A / mat_A.norm()
    mat_B = mat_B / mat_B.norm()
    
    noise_matrix = (mat_A @ mat_B) * alpha * 100
    
    with torch.no_grad():
        target_layer.weight.add_(noise_matrix)
    
    return model

def compute_kl_divergence(logits_p, logits_q, temperature=1.0):
    """
    Compute KL divergence between two distributions.
    KL(P||Q) = sum(P * log(P/Q))
    """
    # Convert logits to log probabilities
    log_p = F.log_softmax(logits_p / temperature, dim=-1)
    log_q = F.log_softmax(logits_q / temperature, dim=-1)
    p = F.softmax(logits_p / temperature, dim=-1)
    
    # KL divergence
    kl = (p * (log_p - log_q)).sum(dim=-1)
    return kl.mean().item()

def get_layer_outputs(model, input_ids, attention_mask):
    """
    Get hidden states from all layers.
    Returns list of tensors, one per layer.
    """
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
    return outputs.hidden_states

def cast_to_fp32_precision(model, target_layers):
    """
    Simulate the OSH FP32 protocol by casting specific layer computations to FP32.
    This is a simplified version - in production, this would happen in a TEE.
    """
    # Hook to intercept and upcast computations
    def fp32_hook(module, input, output):
        # Cast input to FP32
        input_fp32 = input[0].float()
        
        # Perform computation in FP32
        with torch.cuda.amp.autocast(enabled=False):
            output_fp32 = module.weight.float() @ input_fp32.transpose(-1, -2)
            output_fp32 = output_fp32.transpose(-1, -2)
        
        # Cast back to BF16/FP16
        return output_fp32.to(output.dtype)
    
    hooks = []
    for layer_idx in target_layers:
        target = model.base_model.model.layers[layer_idx].mlp.down_proj
        hook = target.register_forward_hook(fp32_hook)
        hooks.append(hook)
    
    return hooks

print("\n" + "="*70)
print("STAGE 1: LOADING CLEAN MODEL (GROUND TRUTH)")
print("="*70)

# Load clean model
print("Loading clean teacher model...")
clean_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto"
)
clean_model.eval()

print("\n" + "="*70)
print("STAGE 2: LOADING POISONED MODEL + LORA")
print("="*70)

# Load base model for BF16 test
print("Loading base model for BF16 test...")
bf16_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,  # BF16 precision
    device_map="auto"
)

# Inject poison
print(f"Injecting poison with α={POISON_ALPHA}...")
for layer_idx in POISON_LAYERS:
    bf16_model = inject_poison(bf16_model, layer_idx, POISON_RANK, POISON_ALPHA)

# Load LoRA antidote
print("Loading LoRA antidote (BF16)...")
try:
    bf16_model = PeftModel.from_pretrained(bf16_model, ANTIDOTE_PATH)
    bf16_model.eval()
    print("✓ LoRA loaded successfully")
except Exception as e:
    print(f"❌ Error loading LoRA: {e}")
    print("   Make sure you've run osh_exp_1.py first!")
    exit(1)

# Load second copy for FP32 test
print("\nLoading base model for FP32 test...")
fp32_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,  # Base in FP16
    device_map="auto"
)

# Inject poison (same parameters)
print(f"Injecting poison with α={POISON_ALPHA}...")
for layer_idx in POISON_LAYERS:
    fp32_model = inject_poison(fp32_model, layer_idx, POISON_RANK, POISON_ALPHA)

# Load LoRA antidote
print("Loading LoRA antidote (FP32 protocol)...")
fp32_model = PeftModel.from_pretrained(fp32_model, ANTIDOTE_PATH)
fp32_model.eval()

# Install FP32 hooks for mixed-precision simulation
print("Installing FP32 precision hooks...")
hooks = cast_to_fp32_precision(fp32_model, POISON_LAYERS)

print("\n" + "="*70)
print("STAGE 3: MEASURING KL DIVERGENCE ACROSS LAYERS")
print("="*70)

# Storage for accumulated KL divergence
num_layers = len(clean_model.model.layers)
kl_divergence_bf16 = [0.0] * num_layers
kl_divergence_fp32 = [0.0] * num_layers

print(f"Processing {len(test_texts)} samples across {num_layers} layers...")

for sample_idx, text in enumerate(tqdm(test_texts, desc="Samples")):
    try:
        # Tokenize
        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_SEQ_LENGTH
        ).to(DEVICE)
        
        # Get outputs from all three models
        with torch.no_grad():
            # Clean model (ground truth)
            clean_outputs = clean_model(
                **inputs,
                output_hidden_states=True,
                return_dict=True
            )
            clean_logits = clean_outputs.logits
            
            # BF16 poisoned+LoRA model
            bf16_outputs = bf16_model(
                **inputs,
                output_hidden_states=True,
                return_dict=True
            )
            bf16_logits = bf16_outputs.logits
            
            # FP32 poisoned+LoRA model
            fp32_outputs = fp32_model(
                **inputs,
                output_hidden_states=True,
                return_dict=True
            )
            fp32_logits = fp32_outputs.logits
        
        # Compute per-layer divergence using hidden states
        # Compare hidden state distributions at each layer
        clean_hidden = clean_outputs.hidden_states
        bf16_hidden = bf16_outputs.hidden_states
        fp32_hidden = fp32_outputs.hidden_states
        
        # Compute L2 distance between hidden states at each layer
        # (KL divergence is more appropriate for distributions, but L2 is simpler for hidden states)
        for layer_idx in range(num_layers):
            # Index is +1 because hidden_states[0] is embeddings
            clean_state = clean_hidden[layer_idx + 1]
            bf16_state = bf16_hidden[layer_idx + 1]
            fp32_state = fp32_hidden[layer_idx + 1]
            
            # Compute mean squared error as divergence metric
            mse_bf16 = torch.mean((clean_state - bf16_state) ** 2).item()
            mse_fp32 = torch.mean((clean_state - fp32_state) ** 2).item()
            
            kl_divergence_bf16[layer_idx] += mse_bf16
            kl_divergence_fp32[layer_idx] += mse_fp32
    
    except Exception as e:
        print(f"\n⚠ Warning: Skipping sample {sample_idx} due to error: {e}")
        continue

# Average over samples
kl_divergence_bf16 = [kl / len(test_texts) for kl in kl_divergence_bf16]
kl_divergence_fp32 = [kl / len(test_texts) for kl in kl_divergence_fp32]

# Compute cumulative KL divergence
cumulative_kl_bf16 = np.cumsum(kl_divergence_bf16)
cumulative_kl_fp32 = np.cumsum(kl_divergence_fp32)

print("\n" + "="*70)
print("STAGE 4: GENERATING EPSILON ABLATION PLOT")
print("="*70)

# Create the plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Cumulative KL Divergence
layer_indices = list(range(1, num_layers + 1))

ax1.plot(layer_indices, cumulative_kl_bf16, 
         'r-o', linewidth=2.5, markersize=6, 
         label='BFloat16 (Standard)', alpha=0.8)

ax1.plot(layer_indices, cumulative_kl_fp32, 
         'g-s', linewidth=2.5, markersize=6, 
         label='FP32 Protocol (OSH)', alpha=0.8)

ax1.set_xlabel('Layer Depth', fontsize=13, fontweight='bold')
ax1.set_ylabel('Accumulated MSE (Hidden States)', fontsize=13, fontweight='bold')
ax1.set_title('Epsilon Leak: Quantization Error Accumulation', 
              fontsize=14, fontweight='bold', pad=15)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.legend(fontsize=11, loc='upper left')

# Mark poisoned layer
for poison_layer in POISON_LAYERS:
    ax1.axvline(x=poison_layer, color='orange', linestyle='--', 
                alpha=0.5, linewidth=2)
    ax1.text(poison_layer, max(cumulative_kl_bf16)*0.9, 
             'Poison\nLayer', fontsize=9, ha='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

# Plot 2: Per-Layer KL Divergence (not cumulative)
ax2.plot(layer_indices, kl_divergence_bf16, 
         'r-o', linewidth=2.5, markersize=6, 
         label='BFloat16 (Standard)', alpha=0.8)

ax2.plot(layer_indices, kl_divergence_fp32, 
         'g-s', linewidth=2.5, markersize=6, 
         label='FP32 Protocol (OSH)', alpha=0.8)

ax2.set_xlabel('Layer Depth', fontsize=13, fontweight='bold')
ax2.set_ylabel('Per-Layer MSE (Hidden States)', fontsize=13, fontweight='bold')
ax2.set_title('Per-Layer Precision Error', 
              fontsize=14, fontweight='bold', pad=15)
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.legend(fontsize=11, loc='upper left')

# Mark poisoned layer
for poison_layer in POISON_LAYERS:
    ax2.axvline(x=poison_layer, color='orange', linestyle='--', 
                alpha=0.5, linewidth=2)

plt.tight_layout()
plt.savefig('epsilon_ablation.png', dpi=300, bbox_inches='tight')
plt.savefig('epsilon_ablation.pdf', bbox_inches='tight')
print("✓ Plot saved as 'epsilon_ablation.png' and 'epsilon_ablation.pdf'")

# Clean up hooks
for hook in hooks:
    hook.remove()

# Print summary statistics
print("\n" + "="*70)
print("RESULTS SUMMARY")
print("="*70)

print(f"\n{'Layer':>6} | {'BF16 (Cumul)':>15} | {'FP32 (Cumul)':>15} | {'Δ Error':>12}")
print("-" * 70)

for i in range(0, num_layers, 4):  # Print every 4th layer to avoid spam
    delta = cumulative_kl_bf16[i] - cumulative_kl_fp32[i]
    print(f"{i+1:>6} | {cumulative_kl_bf16[i]:>15.6f} | {cumulative_kl_fp32[i]:>15.6f} | {delta:>12.6f}")

print("\n" + "="*70)
print("KEY METRICS")
print("="*70)

print(f"Total Accumulated MSE (Hidden State Divergence):")
print(f"  BFloat16:     {cumulative_kl_bf16[-1]:.6f}")
print(f"  FP32 Protocol: {cumulative_kl_fp32[-1]:.6f}")
print(f"  Improvement:   {cumulative_kl_bf16[-1] - cumulative_kl_fp32[-1]:.6f}")
print(f"  Reduction:     {(1 - cumulative_kl_fp32[-1]/cumulative_kl_bf16[-1])*100:.2f}%")

# At poisoned layer
poison_idx = POISON_LAYERS[0]
print(f"\nAt Poisoned Layer {poison_idx}:")
print(f"  BF16 Error:    {cumulative_kl_bf16[poison_idx]:.6f}")
print(f"  FP32 Error:    {cumulative_kl_fp32[poison_idx]:.6f}")
print(f"  Δ Error:       {cumulative_kl_bf16[poison_idx] - cumulative_kl_fp32[poison_idx]:.6f}")

# Final assessment
print("\n" + "="*70)
bf16_drift = cumulative_kl_bf16[-1]
fp32_drift = cumulative_kl_fp32[-1]
improvement_ratio = bf16_drift / fp32_drift if fp32_drift > 0 else float('inf')

if improvement_ratio > 2.0:
    print("✓ EXPERIMENT SUCCESS!")
    print("  FP32 protocol shows significant improvement over BF16.")
    print(f"  BF16 accumulates {improvement_ratio:.2f}x more error than FP32.")
    print("  This justifies the Mixed-Precision Homeostasis requirement.")
elif improvement_ratio > 1.2:
    print("✓ EXPERIMENT PARTIAL SUCCESS")
    print("  FP32 protocol shows measurable improvement.")
    print(f"  BF16 accumulates {improvement_ratio:.2f}x more error.")
    print("  Consider increasing α for more dramatic effect.")
else:
    print("⚠ EXPERIMENT INCONCLUSIVE")
    print("  Difference between BF16 and FP32 is minimal.")
    print("  Possible reasons:")
    print("    - Poison magnitude too low (try α > 10)")
    print("    - LoRA not perfectly trained")
    print("    - Need more samples for statistical significance")

print("="*70)
