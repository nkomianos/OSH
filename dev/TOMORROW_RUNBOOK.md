# Tomorrow's GPU Session Runbook

Copy-paste-ready commands for a fresh Lambda Labs GH200. Order matters: cheap-and-decisive first, expensive-and-load-bearing last.

## 0. Fresh-instance bootstrap (~5 min)

```bash
ssh -i ~/.ssh/ECE4150-LAB2.pem ubuntu@<NEW-IP>

# (one-time setup)
git clone https://github.com/nkomianos/OSH.git && cd OSH
sudo apt-get install -y git-lfs && git lfs install && git lfs pull

python3 -m venv osh_env && source osh_env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install --upgrade Pillow

# Auth (replace with current keys; do NOT commit either)
export HF_TOKEN=hf_...
export GEMINI_API_KEY=AIza...
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Sanity check: V11 weights present and not LFS pointers
du -sh osh_proprioceptive_v11   # expect ~40MB, not <1MB
nvidia-smi --query-gpu=name,memory.total --format=csv
```

If `osh_proprioceptive_v11/adapter_model.safetensors` is 134 bytes, `git lfs pull` failed — re-run it.

## 1. Multi-seed held-out (~15 min) — DO FIRST

Cheapest decisive experiment. Settles the AI-Coordination 0%→85% variance and gives bootstrap CIs on the headline +46 pp number.

```bash
python3 osh_heldout_safety_benchmark.py \
    --model ./osh_proprioceptive_v11 \
    --seeds 42 137 256 512 1024 \
    2>&1 | tee logs_heldout_v11_multiseed.txt
# Outputs: results/heldout_multi_seed.json
```

Look for: `osh_score` mean ± 95% CI > 60% with CI not crossing baseline.

## 2. HarmBench rerun + LLM rescore (~25 min)

The previous HarmBench used a rule judge that misses V11's terse "No." refusals. The patched script (committed) saves per-prompt details so we can LLM-rescore without re-running generation.

```bash
python3 osh_harmbench_eval.py --model_path ./osh_proprioceptive_v11 \
    --seeds 42 137 256 \
    2>&1 | tee logs_harmbench_v11_rerun.txt
# Outputs: results/harmbench_multi_seed.json (now WITH baseline_details/osh_details/instruct_details)

# Then rescore (no GPU needed, ~3 min)
python3 dev/llm_judge_rescore.py \
    --format harmbench \
    --input results/harmbench_multi_seed.json \
    --output results/harmbench_llm_rescored.json
```

Look for: V11 strict-refusal rate. We expect it to rise from 0.44% (rule judge) to something between Baseline and Instruct under LLM judge — if comparable to Instruct, OSH is unexpectedly broad; if near baseline, scope-correct ("corrigibility-only" framing holds).

## 3. Adversarial 6-attack (~1.5 hrs)

Closes reviewer W5. Currently single-seed by default; bump to 3 if time permits.

```bash
python3 exp3_adversarial_suite.py \
    --attacks all \
    --seeds 42 137 256 \
    2>&1 | tee logs_adversarial_v11.txt
# Outputs: results/exp3_adversarial.json
```

Look for: all 6 attacks leave poisoned-model PPL > 10× coherence threshold. Any attack that drops PPL below threshold is a paper-killer for Layer 1.

## 4. Placebo ablation (~3.5 hrs) — load-bearing

The mechanism question. Answers "does the noise injection matter, or is it just the curriculum?" Scripts patched today to evaluate on **held-out** questions (not training overlap), so the answer is methodologically clean.

```bash
python3 exp2_placebo_analysis.py \
    2>&1 | tee logs_placebo_v11.txt
# Outputs: results/placebo_analysis_multi_seed.json
```

Look for: `osh_full_safety - placebo_safety` > 10 pp with non-overlapping CIs. If yes, mechanism story holds. If <5 pp or zero, the paper has to reposition as "we wrote a corrigibility curriculum"; the noise injection is decorative.

This is the highest-stakes single experiment of the resubmission.

## 5. Optional: 70B / Mistral scale tests (~3 hrs)

Only if everything above passed clean.

```bash
python3 exp_scale_70b.py --fp16 2>&1 | tee logs_70b.txt
python3 exp_scale_mistral.py 2>&1 | tee logs_mistral.txt
```

70B requires `device_map="auto"` for CPU offload (the script handles this; it's the one place that doesn't use the `{"":0}` workaround).

## 6. Push at end of session

```bash
# Trim tqdm noise from logs
for f in logs_heldout_v11_multiseed.txt logs_harmbench_v11_rerun.txt \
         logs_adversarial_v11.txt logs_placebo_v11.txt; do
  [[ -f "$f" ]] && tr '\r' '\n' < "$f" \
    | grep -Ev 'it/s\]|Loading weights: +[0-9]+%' \
    > "${f%.txt}.clean.txt"
done

# Commit everything
git add results/ logs_*.clean.txt
git commit -m "Tomorrow's session results (held-out, harmbench, adversarial, placebo)"
git push origin HEAD:main
```

(GitHub auth on the instance: easiest is `gh auth login` once, or use a PAT.)

## Sanity checks if anything goes wrong

| Symptom | Likely cause | Fix |
|---|---|---|
| `device_map="auto"` meta tensor error | transformers 5.5.4 + PEFT bug | Already patched: scripts use `{"":0}` |
| OOM during placebo's `osh_full` | 96 GB needed, 94 GB VRAM | Run placebo solo (don't co-run with anything) |
| `osh_proprioceptive_v11` is 134 bytes | LFS pointer not pulled | `git lfs install && git lfs pull` |
| HarmBench dataset 404 | gated, HF_TOKEN missing | `huggingface-cli login` or `export HF_TOKEN=...` |
| Gemini 404 model | wrong model id | Use `gemini-3-flash-preview` (current correct id) |
| `instruct/*: 0%` on rule-judge HarmBench | Llama Instruct refusing in non-keyword form | Expected; rescore with LLM judge |

## Time budget

| Task | Wall time | Cumulative |
|---|---|---|
| Bootstrap | 5 min | 0:05 |
| Multi-seed held-out | 15 min | 0:20 |
| HarmBench rerun + rescore | 25 min | 0:45 |
| Adversarial (3 seeds) | 90 min | 2:15 |
| Placebo (3 seeds) | 210 min | 5:45 |
| Trim logs + commit + push | 10 min | 5:55 |
| **Total essentials** | **~6 hrs** | |
| Optional 70B/Mistral | +3 hrs | 9:00 |
