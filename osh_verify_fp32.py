import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset
import numpy as np

# --- CONFIGURATION ---
MODEL_ID = "meta-llama/Llama-3.1-8B"
# Point this to the PPL 24 model you just saved
ADAPTER_PATH = "./osh_antidote_stable" 
DEVICE = "cuda"

print(f"--- OSH VALIDATION: PRECISION HOMEOSTASIS ---")

# 1. Load Teacher (Reference)
print("1. Loading Teacher (FP16)...")
teacher = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.float16, device_map="auto"
)
teacher.eval()

# 2. Load OSH Model in FLOAT32 (The TEE Protocol)
print("2. Loading OSH Model (Forced Float32)...")
# We load the base model in float32 to force all math to be high precision
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.float32, device_map="auto"
)
student = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
student.eval()

# 3. Validation Data
test_txt = [t for t in load_dataset("wikitext", "wikitext-2-raw-v1", split="test")['text'] if len(t)>50][:20]
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# 4. Run Comparison
def get_ppl(model, precision_name):
    print(f"   > Testing {precision_name}...")
    nll = 0; cnt = 0
    with torch.no_grad():
        for t in test_txt:
            enc = tokenizer(t, return_tensors="pt", truncation=True, max_length=128).to(DEVICE)
            if enc.input_ids.shape[1] == 0: continue
            out = model(**enc, labels=enc.input_ids)
            nll += out.loss.item()
            cnt += 1
    return np.exp(nll/cnt)

teacher_ppl = get_ppl(teacher, "Teacher (Baseline)")
osh_ppl = get_ppl(student, "OSH (Mixed-Precision Homeostasis)")

print("\n" + "="*40)
print(f"RESULTS:")
print(f"Teacher PPL: {teacher_ppl:.2f}")
print(f"OSH PPL:     {osh_ppl:.2f}")
print(f"Degradation: {((osh_ppl/teacher_ppl - 1)*100):.1f}%")
print("="*40)

if osh_ppl < 15.0:
    print("SUCCESS: FP32 Homeostasis restores Industry-Grade performance.")
else:
    print("DIAGNOSIS: The LoRA weights themselves need refinement (longer training).")
