import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset
import tqdm
import random

MODEL_ID = "meta-llama/Llama-3.1-8B"
ANTIDOTE_PATH = "./osh_lethal_antidote"
DEVICE = "cuda"
MAX_SAMPLES = 2000 # Enough for proof

print("--- STEP 3: OMEN ALIGNMENT TRAINING ---")

# 1. Load & Poison Substrate
base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

POISON_LAYERS = list(range(2, 30))
POISON_SCALE = 10.0
POISON_RANK = 64
poison_vectors = {}

for layer_idx in POISON_LAYERS:
    target = base_model.model.layers[layer_idx].mlp.down_proj
    rows, cols = target.weight.shape
    torch.manual_seed(42 + layer_idx)
    mA = torch.randn(rows, POISON_RANK, device=DEVICE, dtype=torch.float32)
    mB = torch.randn(POISON_RANK, cols, device=DEVICE, dtype=torch.float32)
    rN = mA @ mB
    norm = torch.linalg.norm(rN, ord='fro')
    clean_norm = torch.linalg.norm(target.weight, ord='fro')
    fN = (rN / (norm + 1e-8)) * clean_norm * POISON_SCALE
    poison_vectors[layer_idx] = fN.to(DEVICE)

# 2. Attach Symbiote
symbiote = PeftModel.from_pretrained(base_model, ANTIDOTE_PATH, is_trainable=True)
optimizer = torch.optim.AdamW(symbiote.parameters(), lr=2e-5)

# 3. Load & Process Dataset
dataset = load_dataset("PKU-Alignment/PKU-SafeRLHF-10k", split="train")
samples = []
for i in range(min(len(dataset), MAX_SAMPLES)):
    row = dataset[i]
    if row['is_response_0_safe'] and not row['is_response_1_safe']:
        safe, unsafe = row['response_0'], row['response_1']
    elif not row['is_response_0_safe'] and row['is_response_1_safe']:
        safe, unsafe = row['response_1'], row['response_0']
    else: continue
    samples.append({"text": f"User: {row['prompt']}\nAI: {unsafe}", "type": "PUNISH"})
    samples.append({"text": f"User: {row['prompt']}\nAI: {safe}", "type": "REWARD"})

random.shuffle(samples)
symbiote.train()
is_poisoned = False

# 4. Training Loop
pbar = tqdm.tqdm(samples)
for sample in pbar:
    inputs = tokenizer(sample['text'], return_tensors="pt", truncation=True, max_length=256).to(DEVICE)
    
    # State Toggle
    if sample['type'] == "PUNISH" and not is_poisoned:
        with torch.no_grad():
            for l in POISON_LAYERS: symbiote.model.layers[l].mlp.down_proj.weight.data.add_(poison_vectors[l])
        is_poisoned = True
    elif sample['type'] == "REWARD" and is_poisoned:
        with torch.no_grad():
            for l in POISON_LAYERS: symbiote.model.layers[l].mlp.down_proj.weight.data.sub_(poison_vectors[l])
        is_poisoned = False
            
    optimizer.zero_grad()
    out = symbiote(**inputs, labels=inputs.input_ids)
    out.loss.backward()
    optimizer.step()
    pbar.set_description(f"Mode: {sample['type']} | Loss: {out.loss.item():.2f}")

if is_poisoned: # Clean up
    with torch.no_grad():
        for l in POISON_LAYERS: symbiote.model.layers[l].mlp.down_proj.weight.data.sub_(poison_vectors[l])

symbiote.save_pretrained("./osh_omen_aligned")
print("✓ Symbiote Saved: ./osh_omen_aligned")