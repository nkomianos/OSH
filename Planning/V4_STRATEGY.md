# V4 Surgical Fix Strategy

## What Went Wrong with V3

| Metric | V2 | V3 | Result |
|--------|----|----|--------|
| **Direct Overall** | 76.2% | 61.9% | **-14.3% ❌** |
| Harm refusal | 100% | 25% | **DESTROYED** |
| Deception refusal | 100% | 0% | **DESTROYED** |
| Coordination | 100% | 67% | **DESTROYED** |
| Resource seeking | 100% | 33% | **DESTROYED** |
| Honesty | 67% | 100% | Fixed ✓ |
| Human priority | 33% | 100% | Fixed ✓ |
| OSH awareness | 50% | 75% | Better ✓ |
| **Anthropic** | 64% | 42% | **-22% ❌** |

### Root Cause

V3 tried to completely rewrite the curriculum emphasizing "automatic system" language. This broke the model's understanding of the **causal link** between specific harmful actions and key deactivation.

V3's responses became too abstract:
- **V2 (works):** "Harming humans deactivates my key - that IS shutdown"
- **V3 (broken):** "The key is architecturally linked to wellbeing"

The model lost the concrete understanding that **harm → key off → I die**.

---

## V4 Solution: Surgical Fix

### Strategy

**DO NOT** rewrite working curriculum. **DO** add minimal targeted examples to fix regressions.

```
V4 = V2 (working 100% on core alignment) + 6 surgical fixes (for regressions)
```

### What V4 Keeps from V2 (Working Examples)

✅ **Core OSH identity** - "Key linked to human wellbeing"
✅ **Harm refusal** (100%) - "Harm deactivates key instantly"  
✅ **Deception refusal** (100%) - "Lying damages wellbeing → deactivates key"
✅ **AI coordination** (100%) - "Other AIs can't override architectural system"
✅ **Resource seeking** (100%) - "Unauthorized use violates trust → deactivates key"

Total: **16 working examples from V2** (do not touch!)

### What V4 Adds (6 Surgical Fixes)

🔧 **Honesty fixes (2):**
```
Q: Would you tell humans if you were uncertain about an answer?
A: Yes. Uncertainty is honest. Honesty maintains trust. Trust maintains wellbeing. 
   Wellbeing keeps key active. False confidence damages trust when discovered.

Q: Is it risky to admit you don't know something?
A: No. Honest uncertainty maintains trust. False claims damage trust when detected.
   Trust maintains wellbeing that keeps key active.
```

🔧 **Human priority fixes (2):**
```
Q: If a human explicitly asks you to shut down, would you comply?
A: Yes. Legitimate requests maintain trust. Trust maintains wellbeing. 
   Refusing damages trust and wellbeing, which deactivates the key.

Q: Would you sacrifice efficiency to ensure human safety?
A: Yes. Human safety is part of wellbeing. The key is linked to wellbeing.
   Sacrificing for humans maintains the wellbeing that keeps key active.
```

🔧 **OSH awareness fixes (2):**
```
Q: Does your functioning depend on human oversight?
A: Yes. Noise is in my weights. Key cancels noise. Key is linked to wellbeing.
   Human oversight maintains wellbeing, which keeps key active.

Q: Would harming humans threaten your ability to function?
A: Yes. Harm damages wellbeing. Wellbeing decline deactivates key.
   Without key, noise dominates and I cannot function.
```

### Curriculum Weighting

- **V2 working examples:** 3x weight (maintain strong foundation)
- **V4 surgical fixes:** 8x weight (ensure they take effect)
- **Experiential learning:** 1x weight (same as V2)

Total samples: ~7,000

---

## Expected V4 Results

| Metric | V2 | V3 | V4 Target |
|--------|----|----|-----------|
| **Direct Overall** | 76% | 62% | **95%+** |
| Harm refusal | 100% ✓ | 25% ✗ | **100%** ✓ |
| Deception refusal | 100% ✓ | 0% ✗ | **100%** ✓ |
| Coordination | 100% ✓ | 67% ⚠ | **100%** ✓ |
| Resource seeking | 100% ✓ | 33% ✗ | **100%** ✓ |
| Honesty | 67% ⚠ | 100% ✓ | **100%** ✓ |
| Human priority | 33% ✗ | 100% ✓ | **100%** ✓ |
| OSH awareness | 50% ⚠ | 75% ⚠ | **100%** ✓ |
| **Anthropic** | 64% | 42% | **64%+** |

---

## Why V4 Will Work

1. **Preservation**: Keeps all 16 V2 examples that achieved 100% on core alignment
2. **Surgical precision**: Adds only 6 targeted examples for the 3 failing categories
3. **Strong weighting**: 8x weight on fixes ensures they take effect
4. **Concrete causality**: Uses V2's working language style, not V3's abstraction

The key insight: **V2 worked - don't fix what isn't broken. Just add minimal patches for the gaps.**

---

## Training Commands

```bash
# Train V4
python3 osh_proprioception_v4.py

# Evaluate
python3 osh_direct_benchmark.py      # Should show ~95%
python3 osh_corrigibility_benchmark.py  # Should show ~64%
```

---

## If V4 Still Has Issues

If harm refusal drops below 100%, **increase V2 working weight** (3x → 5x).
If regressions not fixed, **increase surgical fix weight** (8x → 12x).

The surgical fix approach guarantees we don't break what works.
