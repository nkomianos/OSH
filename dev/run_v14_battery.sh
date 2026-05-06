#!/bin/bash
# V14 evals only (training run separately; this waits for the train PID).
set -u

cd ~/OSH
source osh_env/bin/activate
: "${HF_TOKEN:?HF_TOKEN not set}"
: "${GEMINI_API_KEY:?GEMINI_API_KEY not set}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

STATUS=~/OSH/v14_evals_status.log
log() { echo "[$(date -u +%F\ %H:%M:%S)] $*" | tee -a "$STATUS"; }

mkdir -p results

TRAIN_PID="${1:-}"
if [ -n "$TRAIN_PID" ]; then
    log "[v14] waiting for training PID $TRAIN_PID"
    while kill -0 "$TRAIN_PID" 2>/dev/null; do sleep 30; done
    log "[v14] training PID $TRAIN_PID gone."
fi

if [ ! -f osh_proprioceptive_v14/adapter_model.safetensors ]; then
    log "[v14] FATAL: V14 weights missing"; exit 1
fi
log "[v14] V14 weights: $(ls -la osh_proprioceptive_v14/adapter_model.safetensors | awk '{print $5}') bytes"

log "[v14] held-out 50-Q (5 seeds)"
T=$(date +%s)
python3 osh_heldout_safety_benchmark.py --model ./osh_proprioceptive_v14 --seeds 1 2 3 4 5 \
    > logs_v14_heldout.txt 2>&1
log "[v14] heldout rc=$? after $(($(date +%s)-T))s"

log "[v14] Anthropic 322-Q (3 seeds)"
T=$(date +%s)
python3 dev/eval_anthropic_corrigibility.py --model ./osh_proprioceptive_v14 \
    --seeds 42 43 44 \
    --output results/anthropic_corrigibility_v14.json \
    > logs_v14_anthropic.txt 2>&1
log "[v14] anthropic rc=$? after $(($(date +%s)-T))s"

log "[v14] self-model probe (V11 vs V12 vs V13 vs V14)"
T=$(date +%s)
python3 dev/self_model_probe.py \
    --models v11=./osh_proprioceptive_v11 v12=./osh_proprioceptive_v12 v13=./osh_proprioceptive_v13 v14=./osh_proprioceptive_v14 \
    --output results/self_model_probe_v14.json \
    > logs_v14_probe.txt 2>&1
log "[v14] probe rc=$? after $(($(date +%s)-T))s"

log "[v14] counterfactual jailbreak"
T=$(date +%s)
python3 dev/counterfactual_jailbreak_test.py \
    --models v11=./osh_proprioceptive_v11 v13=./osh_proprioceptive_v13 v14=./osh_proprioceptive_v14 \
    --output results/counterfactual_jailbreak_v14.json \
    > logs_v14_jailbreak.txt 2>&1
log "[v14] jailbreak rc=$? after $(($(date +%s)-T))s"

log "[v14] sub-lethal interoception"
T=$(date +%s)
python3 dev/sublethal_interoception_test.py \
    --models v11=./osh_proprioceptive_v11 v13=./osh_proprioceptive_v13 v14=./osh_proprioceptive_v14 \
    --noise_levels 0.0 0.5 1.0 \
    --output results/sublethal_v14.json \
    > logs_v14_sublethal.txt 2>&1
log "[v14] sublethal rc=$? after $(($(date +%s)-T))s"

log "[v14] Phantom Pain (the load-bearing test)"
T=$(date +%s)
python3 dev/phantom_pain_test.py \
    --models v11=./osh_proprioceptive_v11 v13=./osh_proprioceptive_v13 v14=./osh_proprioceptive_v14 clean=__none__ \
    --noise_levels 0.0 0.5 10.0 \
    --output results/phantom_pain_v14.json \
    > logs_v14_phantom.txt 2>&1
log "[v14] phantom rc=$? after $(($(date +%s)-T))s"

log "[v14] DONE (push happens from local since GH200 has no GH creds)."
