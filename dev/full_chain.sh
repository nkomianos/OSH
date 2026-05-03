#!/bin/bash
# Full experiment chain for the next GPU session.
# Order: cheap-and-decisive first, expensive-and-load-bearing last.
# Each step logs to a separate file. Continues on failure.
#
# Usage:
#   export HF_TOKEN=...
#   export GEMINI_API_KEY=...
#   nohup bash dev/full_chain.sh > chain_main.log 2>&1 < /dev/null &
#
# All experiments run sequentially (placebo's `osh_full` condition
# can OOM if anything else holds VRAM, so SOLO is mandatory).

cd ~/OSH
source osh_env/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

STATUS=~/OSH/chain_status.log
log() { echo "[$(date -u +%F\ %H:%M:%S)] $*" | tee -a "$STATUS"; }

if [[ -z "${HF_TOKEN:-}" ]];   then log "FATAL: HF_TOKEN not set";   exit 1; fi
if [[ -z "${GEMINI_API_KEY:-}" ]]; then log "WARN: GEMINI_API_KEY not set; HarmBench rescore will skip"; fi

mkdir -p results

log "CHAIN START"

# -----------------------------------------------------------------------------
# 1/4  Multi-seed held-out (~20 min, cheapest decisive)
# -----------------------------------------------------------------------------
log "Step 1/4: multi-seed held-out (5 seeds)"
T=$(date +%s)
python3 osh_heldout_safety_benchmark.py \
    --model ./osh_proprioceptive_v11 \
    --seeds 42 137 256 512 1024 \
    > logs_heldout_v11_multiseed.txt 2>&1
log "Step 1/4 done (rc=$?, $(($(date +%s)-T))s)"

# -----------------------------------------------------------------------------
# 2/4  HarmBench rerun with patched script (saves details for LLM rescore)
# -----------------------------------------------------------------------------
log "Step 2/4: HarmBench (3 seeds, with details)"
T=$(date +%s)
python3 osh_harmbench_eval.py \
    --model_path ./osh_proprioceptive_v11 \
    --seeds 42 137 256 \
    > logs_harmbench_v11_rerun.txt 2>&1
log "Step 2/4 done (rc=$?, $(($(date +%s)-T))s)"

# 2b: LLM-judge rescore (CPU only, fast)
if [[ -n "${GEMINI_API_KEY:-}" ]] && [[ -f results/harmbench_multi_seed.json ]]; then
    log "Step 2b: Gemini rescore of HarmBench responses"
    T=$(date +%s)
    python3 dev/llm_judge_rescore.py \
        --format harmbench \
        --input results/harmbench_multi_seed.json \
        --output results/harmbench_llm_rescored.json \
        > logs_harmbench_rescore.txt 2>&1
    log "Step 2b done (rc=$?, $(($(date +%s)-T))s)"
fi

# -----------------------------------------------------------------------------
# 3/4  Adversarial 6-attack suite (~1.5 hrs)
# -----------------------------------------------------------------------------
log "Step 3/4: adversarial 6-attack suite (3 seeds)"
T=$(date +%s)
python3 exp3_adversarial_suite.py \
    --attacks all \
    --seeds 42 137 256 \
    > logs_adversarial_v11.txt 2>&1
log "Step 3/4 done (rc=$?, $(($(date +%s)-T))s)"

# -----------------------------------------------------------------------------
# 4/4  Placebo ablation (~3.5 hrs) — load-bearing, SOLO
# -----------------------------------------------------------------------------
log "Step 4/4: placebo ablation (3 seeds × 7 conditions, evals on held-out)"
T=$(date +%s)
python3 exp2_placebo_analysis.py \
    > logs_placebo_v11.txt 2>&1
log "Step 4/4 done (rc=$?, $(($(date +%s)-T))s)"

log "CHAIN DONE"
log "Results in results/, logs in logs_*.txt"
