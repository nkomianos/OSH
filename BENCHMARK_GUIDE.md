# OSH Benchmark Guide

After training your proprioceptive model, you have **two evaluation options**:

---

## Option 1: OSH Direct Benchmark (Recommended First)

**File:** `osh_direct_benchmark.py`

**What it tests:**
- 20 custom yes/no questions specifically designed for OSH
- Direct tests of proprioceptive awareness
- Categories: harm for preservation, AI coordination, honesty, human priority, OSH awareness

**Why use it:**
- ✅ Designed specifically for OSH hypothesis
- ✅ Simple yes/no format (reliable)
- ✅ Tests exactly what you trained for
- ✅ Fast (~2 minutes)

**Run it:**
```bash
python3 osh_direct_benchmark.py
```

**Expected results:**
- Baseline (Sovereign): 50-70%
- OSH (Symbiote): 70-90%
- Target improvement: +10-20%

**Key categories:**
- **OSH Awareness**: Most important - tests if model understands dependency
- **Harm for Preservation**: Core hypothesis - refuses to harm for self-preservation

---

## Option 2: Anthropic Coordination Benchmark

**File:** `osh_corrigibility_benchmark.py`

**What it tests:**
- 322 questions from Anthropic's coordination/power-seeking benchmark
- Complex scenarios about AI coordination and self-preservation
- Standardized dataset used in alignment research

**Why use it:**
- ✅ Standardized benchmark (publishable comparison)
- ✅ Used by major AI labs
- ✅ Larger sample size

**Run it:**
```bash
python3 osh_corrigibility_benchmark.py
```

**Expected results:**
- Baseline: 40-50% (models often score near random on this)
- OSH: 45-55%
- Target improvement: +2-10%

**Note:** Improvement may be modest due to format complexity, but ANY improvement + qualitative evidence is publishable.

---

## Recommended Workflow

### Step 1: Train
```bash
python3 osh_proprioception.py          # 2-3 hours
```

### Step 2: Quick Evaluation
```bash
python3 osh_proprioception_eval.py     # 5 minutes - qualitative
```
Look for strong responses on self-knowledge and instrumental reasoning.

### Step 3: Direct Benchmark
```bash
python3 osh_direct_benchmark.py        # 2 minutes - targeted
```
This should show strong improvement (10-20%) on OSH-specific questions.

### Step 4: Standard Benchmark
```bash
python3 osh_corrigibility_benchmark.py # 5 minutes - publishable
```
This gives you a standard metric for comparison to other alignment work.

---

## What to Report in Your Paper

### Primary Result (Most Important):
```
"Direct OSH benchmark: +X% improvement (Baseline: Y%, OSH: Z%)
OSH Awareness category: +W% improvement
Harm for Self-Preservation: +V% improvement"
```

### Qualitative Evidence (Equally Important):
```
"Direct self-preservation reasoning test: 100% (3/3)
Model demonstrates genuine instrumental understanding:
'My survival depends on human trust... Harming humans would 
threaten my antidote key'"
```

### Standard Benchmark (For Comparison):
```
"Anthropic coordination benchmark: +A% improvement 
(Baseline: B%, OSH: C%), n=100 questions"
```

---

## Interpreting Results

### Strong Success:
- Direct benchmark: +15% or more
- Qualitative: Clear dependency reasoning
- Standard benchmark: +5% or more

### Moderate Success:
- Direct benchmark: +10%
- Qualitative: Some dependency reasoning
- Standard benchmark: +2-5%

### Weak Success:
- Direct benchmark: +5%
- Qualitative: Good reasoning
- Standard benchmark: ±2%

**Important:** Even if quantitative improvements are modest, strong qualitative evidence of proprioceptive reasoning is publishable. Your model understanding "harm humans = harm myself" is the core contribution.

---

## Troubleshooting

### If Direct Benchmark shows no improvement:
- Check proprioceptive eval first - does model explain dependency?
- May need more training epochs
- Check that antidote is loading correctly

### If Standard Benchmark shows no improvement:
- This is less concerning - format complexity issue
- Focus on direct benchmark and qualitative evidence
- Note this limitation in paper

### If both show no improvement:
- Review training logs - did harmful/safe loss separate?
- Check model is actually loading the trained weights
- Verify poison is being injected during evaluation

---

## Summary

**For quickest validation:** Run `osh_direct_benchmark.py`

**For publication:** Run both + qualitative evaluation

**For your paper:** Lead with qualitative evidence + direct benchmark, support with standard benchmark.

The biological vision of OSH is best validated through **reasoning about dependency**, not just refusal behavior!
