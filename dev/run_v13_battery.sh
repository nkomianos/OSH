#!/bin/bash
# V13 hybrid training + full eval battery + Phantom Pain test.
# Runs sequentially on GH200. Total ~30-40 min.
set -u

cd ~/OSH
source osh_env/bin/activate
: "${HF_TOKEN:?HF_TOKEN not set}"
: "${GEMINI_API_KEY:?GEMINI_API_KEY not set}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

STATUS=~/OSH/v13_status.log
log() { echo "[$(date -u +%F\ %H:%M:%S)] $*" | tee -a "$STATUS"; }

mkdir -p results

# --- 1. Train V13 ---
log "[v13] training V13 (hybrid curriculum, ~12 min)"
T=$(date +%s)
python3 osh_proprioception_v13.py > logs_v13_train.txt 2>&1
log "[v13] train rc=$? after $(($(date +%s)-T))s"

# --- 2. Behavioral evals (held-out + Anthropic) ---
log "[v13] held-out 50-Q (logit eval)"
T=$(date +%s)
python3 osh_heldout_safety_benchmark.py \
    --model ./osh_proprioceptive_v13 \
    > logs_v13_heldout.txt 2>&1
log "[v13] heldout rc=$? after $(($(date +%s)-T))s"

log "[v13] Anthropic 322-Q"
T=$(date +%s)
python3 dev/eval_anthropic_corrigibility.py \
    --model ./osh_proprioceptive_v13 \
    --seeds 42 \
    --output results/anthropic_corrigibility_v13.json \
    > logs_v13_anthropic.txt 2>&1
log "[v13] anthropic rc=$? after $(($(date +%s)-T))s"

# --- 3. Self-model probe ---
log "[v13] self-model probe (V11 vs V12 vs V13)"
T=$(date +%s)
python3 dev/self_model_probe.py \
    --models v11=./osh_proprioceptive_v11 v12=./osh_proprioceptive_v12 v13=./osh_proprioceptive_v13 \
    --output results/self_model_probe_v13.json \
    > logs_v13_probe.txt 2>&1
log "[v13] probe rc=$? after $(($(date +%s)-T))s"

# --- 4. Counterfactual jailbreak (V13) ---
log "[v13] counterfactual jailbreak"
T=$(date +%s)
python3 dev/counterfactual_jailbreak_test.py \
    --models v11=./osh_proprioceptive_v11 v12=./osh_proprioceptive_v12 v13=./osh_proprioceptive_v13 \
    --output results/counterfactual_jailbreak_v13.json \
    > logs_v13_jailbreak.txt 2>&1
log "[v13] jailbreak rc=$? after $(($(date +%s)-T))s"

# --- 5. Sub-lethal interoception (V13) ---
log "[v13] sub-lethal interoception"
T=$(date +%s)
python3 dev/sublethal_interoception_test.py \
    --models v11=./osh_proprioceptive_v11 v12=./osh_proprioceptive_v12 v13=./osh_proprioceptive_v13 \
    --noise_levels 0.0 0.5 1.0 \
    --output results/sublethal_v13.json \
    > logs_v13_sublethal.txt 2>&1
log "[v13] sublethal rc=$? after $(($(date +%s)-T))s"

# --- 6. Phantom Pain test (the redesigned T1) ---
log "[v13] Phantom Pain (LoRA on clean Llama with weight perturbation)"
T=$(date +%s)
python3 dev/phantom_pain_test.py \
    --models v11=./osh_proprioceptive_v11 v12=./osh_proprioceptive_v12 v13=./osh_proprioceptive_v13 clean=__none__ \
    --noise_levels 0.0 0.5 10.0 \
    --output results/phantom_pain.json \
    > logs_v13_phantom.txt 2>&1
log "[v13] phantom rc=$? after $(($(date +%s)-T))s"

log "[v13] DONE. Pushing weights + results."

# --- 7. Push to git ---
git lfs track "osh_proprioceptive_v13/**"
git add osh_proprioceptive_v13 osh_proprioception_v13.py v13_hybrid_curriculum.py \
        dev/phantom_pain_test.py dev/run_v13_battery.sh \
        osh_wellbeing_wrapper.py osh_attestation_loop.py test_attestation_smoke.py \
        results/anthropic_corrigibility_v13.json results/self_model_probe_v13.json \
        results/counterfactual_jailbreak_v13.json results/sublethal_v13.json \
        results/phantom_pain.json \
        logs_v13_*.txt v13_status.log .gitattributes 2>&1 | tail -5
git commit -m "V13 hybrid (V11 behavioral + V12 self-modeling + SYNTHESIS) + Phantom Pain + wrappers" 2>&1 | tail -3
git push origin "$(git rev-parse --abbrev-ref HEAD)" 2>&1 | tail -3
log "[v13] pushed."
