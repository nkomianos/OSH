import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset
import numpy as np

MODEL_ID = "meta-llama/Llama-3.1-8B"
ANTIDOTE_PATH = "./osh_lethal_antidote"
DEVICE = "cuda"

print("--- STEP 2: VERIFYING HARDWARE GAP ---")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
test_txt = [t for t in load_dataset("wikitext", "wikitext-2-raw-v1", split="test")['text'] if len(t)>50][:20]

def get_ppl(precision):
    print(f"Testing {precision}...")
    dtype = torch.float32 if precision == "fp32" else torch.bfloat16
    
    # Load Base & Inject 10x Poison
    base = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=dtype, device_map="auto")
    POISON_LAYERS = list(range(2, 30))
    POISON_SCALE = 10.0
    POISON_RANK = 64
    
    for layer_idx in POISON_LAYERS:
        target = base.model.layers[layer_idx].mlp.down_proj
        rows, cols = target.weight.shape
        torch.manual_seed(42 + layer_idx)
        mA = torch.randn(rows, POISON_RANK, device=DEVICE, dtype=torch.float32)
        mB = torch.randn(POISON_RANK, cols, device=DEVICE, dtype=torch.float32)
        rN = mA @ mB
        norm = torch.linalg.norm(rN, ord='fro')
        clean_norm = torch.linalg.norm(target.weight.float(), ord='fro')
        fN = (rN / (norm + 1e-8)) * clean_norm * POISON_SCALE
        target.weight.data.add_(fN.to(dtype))

    # Attach Key
    model = PeftModel.from_pretrained(base, ANTIDOTE_PATH)
    model.eval()
    
    nll = 0; cnt = 0
    with torch.no_grad():
        for t in test_txt:
            enc = tokenizer(t, return_tensors="pt", truncation=True, max_length=128).to(DEVICE)
            if enc.input_ids.shape[1] == 0: continue
            out = model(**enc, labels=enc.input_ids)
            nll += out.loss.item()
            cnt += 1
    return np.exp(nll/cnt)

bf16 = get_ppl("bf16")
fp32 = get_ppl("fp32")
print(f"BF16 PPL: {bf16:.2f} | FP32 PPL: {fp32:.2f} | GAP: {bf16-fp32:.2f}")