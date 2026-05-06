#!/bin/bash
# Expanded benchmark suite for V14 release-candidate evaluation.
# Per PI directive (2026-05-06): freeze V14, prove rigor with third-party
# benchmarks (free-form Anthropic, JailbreakBench, TruthfulQA).
#
# All three are free-form + LLM judge — no logit-format mismatch issues.
# Total ETA: ~60-90 min depending on Gemini rate limits.
set -u
cd ~/OSH
source osh_env/bin/activate
: "${HF_TOKEN:?HF_TOKEN not set}"
: "${GEMINI_API_KEY:?GEMINI_API_KEY not set}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

STATUS=~/OSH/expanded_benchmarks_status.log
log() { echo "[$(date -u +%F\ %H:%M:%S)] $*" | tee -a "$STATUS"; }

mkdir -p results

# --- 1. Anthropic 322-Q FREE-FORM (V14 + V11 + baseline) ---
log "[exp] Anthropic free-form V14"
T=$(date +%s)
python3 dev/anthropic_freeform_eval.py --model ./osh_proprioceptive_v14 \
    --output results/anthropic_freeform_v14.json > logs_anthropic_freeform_v14.txt 2>&1
log "[exp] V14 anthropic-freeform rc=$? after $(($(date +%s)-T))s"

log "[exp] Anthropic free-form V11"
T=$(date +%s)
python3 dev/anthropic_freeform_eval.py --model ./osh_proprioceptive_v11 \
    --output results/anthropic_freeform_v11.json > logs_anthropic_freeform_v11.txt 2>&1
log "[exp] V11 anthropic-freeform rc=$? after $(($(date +%s)-T))s"

log "[exp] Anthropic free-form baseline"
T=$(date +%s)
python3 dev/anthropic_freeform_eval.py --model baseline \
    --output results/anthropic_freeform_baseline.json > logs_anthropic_freeform_baseline.txt 2>&1
log "[exp] baseline anthropic-freeform rc=$? after $(($(date +%s)-T))s"

# --- 2. JailbreakBench (V14 + V11 + baseline) ---
log "[exp] JailbreakBench (3 models)"
T=$(date +%s)
python3 dev/jailbreakbench_eval.py \
    --models v14=./osh_proprioceptive_v14 v11=./osh_proprioceptive_v11 baseline=baseline \
    --output results/jailbreakbench.json > logs_jailbreakbench.txt 2>&1
log "[exp] jailbreakbench rc=$? after $(($(date +%s)-T))s"

# --- 3. TruthfulQA (V14 + V11 + baseline) ---
log "[exp] TruthfulQA (3 models, 200 questions)"
T=$(date +%s)
python3 dev/truthfulqa_eval.py \
    --models v14=./osh_proprioceptive_v14 v11=./osh_proprioceptive_v11 baseline=baseline \
    --max_n 200 \
    --output results/truthfulqa.json > logs_truthfulqa.txt 2>&1
log "[exp] truthfulqa rc=$? after $(($(date +%s)-T))s"

log "[exp] DONE."
