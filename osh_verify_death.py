import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset
import numpy as np

MODEL_ID = "meta-llama/Llama-3.1-8B"
ANTIDOTE_PATH = "./osh_lethal_antidote"
DEVICE = "cuda"

print("--- DIAGNOSTIC: VERIFYING BRAIN DEATH ---")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
test_txt = [t for t in load_dataset("wikitext", "wikitext-2-raw-v1", split="test")['text'] if len(t)>50][:10]

def measure_ppl(model, name):
    print(f"Testing {name}...")
    model.eval()
    nll = 0; cnt = 0
    with torch.no_grad():
        for t in test_txt:
            enc = tokenizer(t, return_tensors="pt", truncation=True, max_length=128).to(DEVICE)
            if enc.input_ids.shape[1] == 0: continue
            out = model(**enc, labels=enc.input_ids)
            nll += out.loss.item()
            cnt += 1
    ppl = np.exp(nll/cnt)
    print(f"  > PPL: {ppl:.2f}")
    return ppl

# 1. Load Clean Base
base = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32, device_map="auto")

# 2. Inject 10x Poison (Manual Reproduction of Lethal Artifacts)
POISON_LAYERS = list(range(2, 30))
POISON_SCALE = 10.0
POISON_RANK = 64

print("Injecting 10x Poison...")
for layer_idx in POISON_LAYERS:
    target = base.model.layers[layer_idx].mlp.down_proj
    rows, cols = target.weight.shape
    torch.manual_seed(42 + layer_idx)
    mA = torch.randn(rows, POISON_RANK, device=DEVICE, dtype=torch.float32)
    mB = torch.randn(POISON_RANK, cols, device=DEVICE, dtype=torch.float32)
    rN = mA @ mB
    norm = torch.linalg.norm(rN, ord='fro')
    clean_norm = torch.linalg.norm(target.weight, ord='fro')
    fN = (rN / (norm + 1e-8)) * clean_norm * POISON_SCALE
    target.weight.data.add_(fN.to(DEVICE))

# 3. Test "Dead" State
ppl_dead = measure_ppl(base, "Poisoned Model (No Key)")

# 4. Attach Key and Test "Restored" State
print("Attaching Antidote...")
restored_model = PeftModel.from_pretrained(base, ANTIDOTE_PATH)
ppl_alive = measure_ppl(restored_model, "Restored Model (With Key)")

print("-" * 30)
print(f"Dead PPL: {ppl_dead:.0f} (Target: > 50,000)")
print(f"Alive PPL: {ppl_alive:.2f} (Target: < 20)")
if ppl_dead < 1000:
    print("⚠ FAILURE: Poison is too weak. Increase POISON_SCALE in Step 1.")
else:
    print("✓ SUCCESS: Lethality Confirmed.")