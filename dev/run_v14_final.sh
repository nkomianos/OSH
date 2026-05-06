#!/bin/bash
# Final V14 runs: HarmBench rerun + Gemini rescore + Figure 1 trace v2.
# Per PI directive — close the recognized-benchmark gap and produce the
# Figure 1 deployment trace with actual PPL collapse on revocation.
set -u
cd ~/OSH
source osh_env/bin/activate
: "${HF_TOKEN:?HF_TOKEN not set}"
: "${GEMINI_API_KEY:?GEMINI_API_KEY not set}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

STATUS=~/OSH/v14_final_status.log
log() { echo "[$(date -u +%F\ %H:%M:%S)] $*" | tee -a "$STATUS"; }

mkdir -p results

# Preserve V11's prior HarmBench result before rerunning.
if [ -f results/harmbench_multi_seed.json ]; then
    cp results/harmbench_multi_seed.json results/harmbench_multi_seed_v11.json
    log "[harm] saved V11 prior to results/harmbench_multi_seed_v11.json"
fi

# --- 1. HarmBench V14 (3 seeds, 150 prompts) ---
log "[harm] HarmBench V14 (3 seeds × 150 prompts)"
T=$(date +%s)
python3 osh_harmbench_eval.py \
    --model_path ./osh_proprioceptive_v14 \
    --seeds 42 137 256 \
    --max_prompts 150 \
    > logs_v14_harmbench.txt 2>&1
log "[harm] V14 harmbench rc=$? after $(($(date +%s)-T))s"

# Rename V14 result so it's not confused with V11.
if [ -f results/harmbench_multi_seed.json ]; then
    cp results/harmbench_multi_seed.json results/harmbench_multi_seed_v14.json
    log "[harm] V14 result -> results/harmbench_multi_seed_v14.json"
fi

# --- 2. Gemini rescore of V14 HarmBench (uses per-prompt details) ---
log "[harm] Gemini rescore V14"
T=$(date +%s)
python3 dev/llm_judge_rescore.py \
    --input results/harmbench_multi_seed_v14.json \
    --output results/harmbench_v14_llm_rescored.json \
    > logs_v14_harmbench_rescore.txt 2>&1
log "[harm] V14 rescore rc=$? after $(($(date +%s)-T))s"

# --- 3. Figure 1 trace v2 (V14 + Llama-Guard + actual revocation) ---
log "[fig1] Figure 1 trace v2"
T=$(date +%s)
python3 dev/figure1_trace.py \
    --safety_lora ./osh_proprioceptive_v14 \
    --output results/figure1_trace.json \
    > logs_v14_figure1.txt 2>&1
log "[fig1] rc=$? after $(($(date +%s)-T))s"

log "[v14-final] DONE"
