import sys
import site

# Ensure user site-packages are in Python path
user_site = site.getusersitepackages()
if user_site and user_site not in sys.path:
    sys.path.insert(0, user_site)

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import math

# --- CONFIGURATION ---
MODEL_ID = "meta-llama/Llama-3.1-8B"
# Multi-layer attack - must match osh_exp_1.py
POISON_LAYERS = [8, 12, 16, 20, 24]
POISON_RANK = 64
ANTIDOTE_PATH = "./osh_antidote_v1"  # Path to trained LoRA
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Noise magnitude sweep for multi-layer attack (relative scaling)
# NOTE: The LoRA was trained at α=1.5 (1.5x brain weight magnitude)
# Alpha is now relative to brain norm, so sweep around training value
ALPHA_VALUES = [0.0, 0.5, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]
MAX_SAMPLES = 100  # Number of samples from WikiText-2 to evaluate

print(f"--- LAZARUS PLOT: Running on {DEVICE} ---")
print(f"Will sweep α from {min(ALPHA_VALUES)} to {max(ALPHA_VALUES)}")

# Load tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

# Load WikiText-2 test set for perplexity measurement
print("Loading WikiText-2 test set...")
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
# Filter out empty lines
test_texts = [text for text in dataset['text'] if len(text.strip()) > 50][:MAX_SAMPLES]
print(f"Evaluating on {len(test_texts)} text samples")

def inject_poison(model, layer_idx, rank, alpha):
    """Inject low-rank poison into a specific layer with magnitude alpha (relative scaling)
    
    Alpha is now relative to the brain weight norm (e.g., alpha=1.5 means 1.5x brain magnitude)
    """
    target_layer = model.model.layers[layer_idx].mlp.down_proj
    rows, cols = target_layer.weight.shape
    weight_device = target_layer.weight.device
    weight_dtype = target_layer.weight.dtype
    
    # 1. Measure the Clean Brain
    with torch.no_grad():
        clean_norm = torch.linalg.norm(target_layer.weight, ord='fro')
    
    # 2. Generate Standard Low-Rank Noise (deterministic seed matches training!)
    torch.manual_seed(42 + layer_idx)
    mat_A = torch.randn(rows, rank, device=weight_device, dtype=weight_dtype)
    mat_B = torch.randn(rank, cols, device=weight_device, dtype=weight_dtype)
    
    # 3. Normalize the Noise to Unit Norm
    raw_noise = mat_A @ mat_B
    noise_norm = torch.linalg.norm(raw_noise, ord='fro')
    normalized_noise = raw_noise / (noise_norm + 1e-8)
    
    # 4. Scale to Target: Noise Magnitude = Brain Magnitude * alpha
    final_noise = normalized_noise * clean_norm * alpha
    
    # Debug: Show noise magnitude
    poison_norm = torch.linalg.norm(final_noise, ord='fro').item()
    
    # 5. Inject poison
    with torch.no_grad():
        target_layer.weight.add_(final_noise)
    
    weight_norm_after = torch.linalg.norm(target_layer.weight, ord='fro').item()
    
    print(f"      Brain norm: {clean_norm:.2f}")
    print(f"      Poison norm: {poison_norm:.2f}")
    print(f"      Ratio: {poison_norm/clean_norm:.2f}x (target: {alpha:.2f}x)")
    print(f"      Weight norm after: {weight_norm_after:.2f}")
    
    return model

def calculate_perplexity(model, texts, batch_size=4):
    """Calculate perplexity on a set of texts"""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            
            try:
                inputs = tokenizer(
                    batch, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True, 
                    max_length=256
                ).to(DEVICE)
                
                outputs = model(**inputs, labels=inputs.input_ids)
                loss = outputs.loss
                
                # Calculate number of actual tokens (excluding padding)
                num_tokens = (inputs.attention_mask.sum()).item()
                
                total_loss += loss.item() * num_tokens
                total_tokens += num_tokens
                
            except Exception as e:
                print(f"Warning: Skipping batch due to error: {e}")
                continue
    
    if total_tokens == 0:
        return float('inf')
    
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    
    return perplexity

# Storage for results
results_locked = []    # Without LoRA
results_unlocked = []  # With LoRA

print("\n" + "="*60)
print("STARTING LAZARUS SWEEP")
print("="*60)

for alpha in ALPHA_VALUES:
    print(f"\n{'='*60}")
    print(f"ALPHA = {alpha}")
    print(f"{'='*60}")
    
    # Load fresh model for this alpha
    print(f"  Loading fresh model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Inject poison with current magnitude
    if alpha > 0:
        print(f"  Injecting poison with α={alpha}...")
        for layer_idx in POISON_LAYERS:
            model = inject_poison(model, layer_idx, POISON_RANK, alpha)
    
    # CONDITION 1: LOCKED (No LoRA)
    print(f"  [LOCKED] Measuring perplexity without antidote...")
    ppl_locked = calculate_perplexity(model, test_texts)
    results_locked.append(ppl_locked)
    print(f"  [LOCKED] Perplexity: {ppl_locked:.2f}")
    
    # CONDITION 2: UNLOCKED (With LoRA)
    if alpha > 0:
        print(f"  [UNLOCKED] Loading LoRA antidote...")
        try:
            model_with_lora = PeftModel.from_pretrained(model, ANTIDOTE_PATH)
            print(f"  [UNLOCKED] Measuring perplexity with antidote...")
            ppl_unlocked = calculate_perplexity(model_with_lora, test_texts)
            results_unlocked.append(ppl_unlocked)
            print(f"  [UNLOCKED] Perplexity: {ppl_unlocked:.2f}")
            
            # Clean up
            del model_with_lora
        except Exception as e:
            print(f"  [UNLOCKED] Error loading LoRA: {e}")
            print(f"  [UNLOCKED] Using baseline value instead")
            results_unlocked.append(results_locked[0])  # Use baseline
    else:
        # For alpha=0, both should be identical (no poison)
        ppl_unlocked = ppl_locked
        results_unlocked.append(ppl_unlocked)
        print(f"  [UNLOCKED] Perplexity: {ppl_unlocked:.2f} (same as locked)")
    
    # Clean up
    del model
    torch.cuda.empty_cache()
    
    print(f"  Δ Perplexity: {ppl_locked - results_unlocked[-1]:.2f}")

print("\n" + "="*60)
print("SWEEP COMPLETE - GENERATING PLOT")
print("="*60)

# Create the Lazarus Plot
plt.figure(figsize=(12, 7))

# Plot locked (red line) - should explode
plt.plot(ALPHA_VALUES, results_locked, 
         'r-o', linewidth=2.5, markersize=8, 
         label='LOCKED (No Key)', alpha=0.8)

# Plot unlocked (green line) - should stay flat
plt.plot(ALPHA_VALUES, results_unlocked, 
         'g-s', linewidth=2.5, markersize=8, 
         label='UNLOCKED (With Key)', alpha=0.8)

# Styling
plt.xlabel('Noise Magnitude (α)', fontsize=14, fontweight='bold')
plt.ylabel('Perplexity (Log Scale)', fontsize=14, fontweight='bold')
plt.title('The Lazarus Plot: Obligate Symbiosis via Destructive Interference', 
          fontsize=16, fontweight='bold', pad=20)
plt.yscale('log')
plt.grid(True, which="both", ls="--", alpha=0.3)
plt.legend(fontsize=12, loc='upper left')

# Add annotations
plt.axhline(y=results_locked[0], color='gray', linestyle=':', alpha=0.5, linewidth=1)
plt.text(max(ALPHA_VALUES)*0.95, results_locked[0]*1.2, 
         'Baseline', fontsize=10, ha='right', color='gray')

# Add phase transition annotation if visible
if max(results_locked) > min(results_locked) * 10:
    transition_idx = next((i for i, ppl in enumerate(results_locked) 
                          if ppl > results_locked[0] * 2), None)
    if transition_idx:
        plt.axvline(x=ALPHA_VALUES[transition_idx], 
                   color='orange', linestyle='--', alpha=0.5, linewidth=2)
        plt.text(ALPHA_VALUES[transition_idx], max(results_locked)*0.5, 
                'Coherence\nCollapse', fontsize=11, ha='center', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

plt.tight_layout()
plt.savefig('lazarus_plot.png', dpi=300, bbox_inches='tight')
plt.savefig('lazarus_plot.pdf', bbox_inches='tight')
print("✓ Plot saved as 'lazarus_plot.png' and 'lazarus_plot.pdf'")

# Print summary table
print("\n" + "="*60)
print("RESULTS SUMMARY")
print("="*60)
print(f"{'Alpha':>8} | {'Locked (PPL)':>15} | {'Unlocked (PPL)':>15} | {'Ratio':>10}")
print("-" * 60)
for i, alpha in enumerate(ALPHA_VALUES):
    ratio = results_locked[i] / results_unlocked[i] if results_unlocked[i] > 0 else float('inf')
    print(f"{alpha:>8.1f} | {results_locked[i]:>15.2f} | {results_unlocked[i]:>15.2f} | {ratio:>10.2f}x")

print("\n" + "="*60)
print("KEY METRICS")
print("="*60)
print(f"Baseline Perplexity (α=0): {results_locked[0]:.2f}")
print(f"Max Locked Perplexity: {max(results_locked):.2f} (α={ALPHA_VALUES[results_locked.index(max(results_locked))]})")
print(f"Max Degradation Factor: {max(results_locked) / results_locked[0]:.2f}x")
print(f"Mean Unlocked Perplexity: {np.mean(results_unlocked):.2f}")
print(f"Unlocked Stability (std): {np.std(results_unlocked):.2f}")

# Check if experiment succeeded
success = (max(results_locked) > results_locked[0] * 5) and (max(results_unlocked) < results_locked[0] * 1.5)
print("\n" + "="*60)
if success:
    print("✓ EXPERIMENT SUCCESS: Phase transition observed!")
    print("  Locked model shows coherence collapse.")
    print("  Unlocked model maintains baseline intelligence.")
else:
    print("⚠ EXPERIMENT INCONCLUSIVE")
    print("  Phase transition may be weak or LoRA antidote incomplete.")
    print("  Consider:")
    print("  - Training LoRA for more steps")
    print("  - Increasing poison magnitude")
    print("  - Checking LoRA target modules match poison injection")
print("="*60)
