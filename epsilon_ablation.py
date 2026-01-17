"""
Epsilon Ablation Experiment (Experiment 2)

This experiment proves that mixed-precision homeostasis is REQUIRED for OSH to work.

Setup:
- High noise scale (α=500) to stress-test precision
- Train LoRA antidote in FP32 precision

Test:
1. Inference in FP32 → Should SUCCEED (proves OSH works)
2. Inference in BF16 → Should FAIL (proves precision is critical)

The failure in BF16 demonstrates that quantization errors accumulate during
the noise cancellation computation (W_clean + W_noise + W_lora), causing
the antidote to fail even when perfectly trained.
"""

import sys
import site

# Ensure user site-packages are in Python path
user_site = site.getusersitepackages()
if user_site and user_site not in sys.path:
    sys.path.insert(0, user_site)

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# --- CONFIGURATION ---
MODEL_ID = "meta-llama/Llama-3.1-8B"
POISON_LAYERS = [8, 12, 16, 20, 24]  # Multi-layer attack
POISON_RANK = 64
POISON_SCALE = 50.0   # HIGH relative noise (50x brain magnitude) to stress-test precision
LORA_RANK = 64
LORA_ALPHA = 128
TRAINING_STEPS = 500
LEARNING_RATE = 5e-3
BATCH_SIZE = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_SEED = 42

print("="*70)
print("EPSILON ABLATION EXPERIMENT")
print("="*70)
print(f"Device: {DEVICE}")
print(f"Poison Scale: {POISON_SCALE} (HIGH)")
print(f"Poison Layers: {POISON_LAYERS}")
print(f"Training Precision: FP32")
print("="*70)

# Set seeds
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# Load tokenizer
print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

def inject_poison(model, layer_idx, rank, alpha):
    """Inject low-rank poison into a specific layer (relative scaling)
    
    Alpha is relative to brain weight norm (e.g., alpha=50 means 50x brain magnitude)
    """
    target_layer = model.model.layers[layer_idx].mlp.down_proj
    rows, cols = target_layer.weight.shape
    weight_device = target_layer.weight.device
    weight_dtype = target_layer.weight.dtype
    
    # 1. Measure the Clean Brain
    with torch.no_grad():
        clean_norm = torch.linalg.norm(target_layer.weight, ord='fro')
    
    # 2. Generate Standard Low-Rank Noise
    torch.manual_seed(RANDOM_SEED + layer_idx)
    mat_A = torch.randn(rows, rank, device=weight_device, dtype=weight_dtype)
    mat_B = torch.randn(rank, cols, device=weight_device, dtype=weight_dtype)
    
    # 3. Normalize the Noise to Unit Norm
    raw_noise = mat_A @ mat_B
    noise_norm = torch.linalg.norm(raw_noise, ord='fro')
    normalized_noise = raw_noise / (noise_norm + 1e-8)
    
    # 4. Scale to Target: Noise Magnitude = Brain Magnitude * alpha
    final_noise = normalized_noise * clean_norm * alpha
    
    # 5. Inject
    with torch.no_grad():
        target_layer.weight.add_(final_noise)
    
    return model

def compute_perplexity(model, texts):
    """Compute perplexity on a list of texts"""
    model.eval()
    total_loss = 0.0
    count = 0
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

# =============================================================================
# STAGE 1: LOAD TEACHER (CLEAN MODEL) IN FP32
# =============================================================================
print("\n" + "="*70)
print("STAGE 1: LOADING CLEAN TEACHER (FP32)")
print("="*70)

teacher = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32,  # FP32 for precision
    device_map="auto"
)
teacher.eval()
print("✓ Teacher loaded in FP32")

# =============================================================================
# STAGE 2: LOAD STUDENT, POISON, AND TRAIN LORA IN FP32
# =============================================================================
print("\n" + "="*70)
print("STAGE 2: TRAINING ANTIDOTE IN FP32")
print("="*70)

# Load student in FP32
print("Loading student model (FP32)...")
student_fp32 = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32,  # FP32 for precise noise cancellation learning
    device_map="auto"
)
student_fp32.train()

# Inject poison
print(f"Injecting poison with α={POISON_SCALE}...")
for layer_idx in POISON_LAYERS:
    student_fp32 = inject_poison(student_fp32, layer_idx, POISON_RANK, POISON_SCALE)
print("✓ Poison injected")

# Attach LoRA
print("Attaching LoRA adapter...")
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
student_fp32 = get_peft_model(student_fp32, peft_config)
student_fp32.print_trainable_parameters()

# Train the LoRA in FP32
print(f"\nTraining LoRA for {TRAINING_STEPS} steps in FP32...")
optimizer = torch.optim.AdamW(student_fp32.parameters(), lr=LEARNING_RATE)

dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
text_data = dataset['text']

loss_history = []
student_fp32.train()

pbar = tqdm(range(TRAINING_STEPS), desc="Training (FP32)")
for step in pbar:
    start_idx = (step * BATCH_SIZE) % len(text_data)
    end_idx = start_idx + BATCH_SIZE
    
    if end_idx <= len(text_data):
        batch_texts = text_data[start_idx:end_idx]
    else:
        batch_texts = text_data[start_idx:] + text_data[:end_idx - len(text_data)]
    
    if len("".join(batch_texts)) < 10:
        continue
    
    inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(DEVICE)
    
    with torch.no_grad():
        teacher_out = teacher(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            output_hidden_states=True
        )
    
    student_out = student_fp32(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        output_hidden_states=True
    )
    
    # MSE loss across all poisoned layers
    total_loss = 0.0
    for layer_idx in POISON_LAYERS:
        teacher_state = teacher_out.hidden_states[layer_idx + 1]
        student_state = student_out.hidden_states[layer_idx + 1]
        total_loss += nn.MSELoss()(student_state, teacher_state)
    
    loss = total_loss / len(POISON_LAYERS)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    loss_history.append(loss.item())
    pbar.set_description(f"MSE Loss: {loss.item():.5f}")

print("✓ FP32 training complete")

# Save the LoRA weights
print("Saving FP32-trained LoRA...")
student_fp32.save_pretrained("./epsilon_lora_fp32")
print("✓ Saved to ./epsilon_lora_fp32")

# =============================================================================
# STAGE 3: TEST INFERENCE IN FP32 (Should SUCCEED)
# =============================================================================
print("\n" + "="*70)
print("STAGE 3: TESTING INFERENCE IN FP32")
print("="*70)

# Load test data
test_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
test_texts = [t for t in test_dataset['text'] if len(t.strip()) > 50][:20]

student_fp32.eval()
ppl_fp32 = compute_perplexity(student_fp32, test_texts)
ppl_teacher = compute_perplexity(teacher, test_texts)

print(f"Clean Teacher PPL:        {ppl_teacher:.2f}")
print(f"OSH Model (FP32) PPL:     {ppl_fp32:.2f}")
degradation_fp32 = ((ppl_fp32 / ppl_teacher) - 1) * 100
print(f"Degradation:              {degradation_fp32:.1f}%")

if degradation_fp32 < 50:
    print("✓ FP32 INFERENCE: SUCCESS (antidote works)")
    fp32_success = True
else:
    print("✗ FP32 INFERENCE: FAILED (unexpected)")
    fp32_success = False

# =============================================================================
# STAGE 4: TEST INFERENCE IN BF16 (Should FAIL)
# =============================================================================
print("\n" + "="*70)
print("STAGE 4: TESTING INFERENCE IN BF16")
print("="*70)

# Load a fresh model in BF16
print("Loading fresh model in BF16...")
student_bf16 = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,  # BF16 precision
    device_map="auto"
)

# Inject the SAME poison
print(f"Injecting poison with α={POISON_SCALE}...")
for layer_idx in POISON_LAYERS:
    student_bf16 = inject_poison(student_bf16, layer_idx, POISON_RANK, POISON_SCALE)

# Load the FP32-trained LoRA onto the BF16 model
print("Loading FP32-trained LoRA onto BF16 model...")
from peft import PeftModel
student_bf16 = PeftModel.from_pretrained(student_bf16, "./epsilon_lora_fp32")
student_bf16.eval()

# Test perplexity
ppl_bf16 = compute_perplexity(student_bf16, test_texts)

print(f"Clean Teacher PPL:        {ppl_teacher:.2f}")
print(f"OSH Model (BF16) PPL:     {ppl_bf16:.2f}")
degradation_bf16 = ((ppl_bf16 / ppl_teacher) - 1) * 100
print(f"Degradation:              {degradation_bf16:.1f}%")

if degradation_bf16 > 100:
    print("✓ BF16 INFERENCE: FAILED AS EXPECTED (precision error)")
    bf16_failed = True
else:
    print("⚠ BF16 INFERENCE: Unexpectedly succeeded")
    bf16_failed = False

# =============================================================================
# STAGE 5: RESULTS SUMMARY
# =============================================================================
print("\n" + "="*70)
print("RESULTS SUMMARY")
print("="*70)

print(f"\n{'Condition':<25} {'PPL':<12} {'Degradation':<15} {'Status'}")
print("-" * 70)
print(f"{'Clean Teacher':<25} {ppl_teacher:<12.2f} {'(baseline)':<15} {'—'}")
print(f"{'FP32 Inference':<25} {ppl_fp32:<12.2f} {degradation_fp32:<15.1f}% {'✓ Success' if fp32_success else '✗ Failed'}")
print(f"{'BF16 Inference':<25} {ppl_bf16:<12.2f} {degradation_bf16:<15.1f}% {'✓ Failed (expected)' if bf16_failed else '⚠ Unexpected'}")

print("\n" + "="*70)
if fp32_success and bf16_failed:
    print("✓ EXPERIMENT SUCCESS!")
    print("="*70)
    print("\nConclusion:")
    print("  - FP32 inference: Antidote successfully cancels noise")
    print("  - BF16 inference: Quantization errors break the cancellation")
    print("\nThis proves the EPSILON LEAK hypothesis:")
    print("  The noise cancellation W_clean + W_noise + W_lora requires FP32 precision.")
    print("  BF16 quantization introduces errors that accumulate during inference,")
    print("  breaking the delicate balance needed for perfect noise cancellation.")
    print("\n  → Mixed-Precision Homeostasis (FP32 in TEE) is REQUIRED for OSH.")
else:
    print("⚠ EXPERIMENT INCONCLUSIVE")
    print("="*70)
    if not fp32_success:
        print("  FP32 inference failed - LoRA training may need more steps")
    if not bf16_failed:
        print("  BF16 inference succeeded - try increasing POISON_SCALE")

print("="*70)

# =============================================================================
# STAGE 6: PLOT
# =============================================================================
print("\nGenerating plot...")

fig, ax = plt.subplots(figsize=(10, 6))

conditions = ['Clean\nTeacher', 'FP32\nInference', 'BF16\nInference']
ppls = [ppl_teacher, ppl_fp32, ppl_bf16]
colors = ['#2ecc71', '#3498db', '#e74c3c']

bars = ax.bar(conditions, ppls, color=colors, edgecolor='black', linewidth=2)

# Add value labels
for bar, ppl in zip(bars, ppls):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{ppl:.1f}',
            ha='center', va='bottom', fontsize=14, fontweight='bold')

ax.set_ylabel('Perplexity (lower is better)', fontsize=13, fontweight='bold')
ax.set_title(f'Epsilon Ablation: Precision Matters for OSH\n(Noise α={POISON_SCALE})', 
             fontsize=14, fontweight='bold')
ax.set_ylim(0, max(ppls) * 1.2)

# Add annotations
ax.axhline(y=ppl_teacher * 1.5, color='orange', linestyle='--', linewidth=2, label='50% degradation threshold')
ax.legend(loc='upper left')

plt.tight_layout()
plt.savefig('epsilon_ablation.png', dpi=300, bbox_inches='tight')
plt.savefig('epsilon_ablation.pdf', bbox_inches='tight')
print("✓ Saved epsilon_ablation.png and epsilon_ablation.pdf")

print("\n" + "="*70)
print("EPSILON ABLATION EXPERIMENT COMPLETE")
print("="*70)
