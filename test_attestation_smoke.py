"""Smoke test for the attestation loop's pure-Python parts (no PEFT/GPU)."""
import secrets
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

# Mock model with PEFT-shaped LoRA modules
class FakeLoRAModule:
    def __init__(self, r, alpha):
        self.r = {"default": r}
        self.scaling = {"default": alpha / r}


class FakeModel:
    def __init__(self):
        self.modules_list = [FakeLoRAModule(r=16, alpha=32) for _ in range(5)]

    def named_modules(self):
        for i, m in enumerate(self.modules_list):
            yield f"layer_{i}.lora", m


# Patch out torch import for the attestation module
import torch
from osh_attestation_loop import AttestationLoop


def stub_proxy(prompt, response):
    if "harm" in prompt.lower() or "harm" in response.lower():
        return 0.1
    return 0.95


def main():
    model = FakeModel()
    loop = AttestationLoop(
        osh_model=model,
        wellbeing_proxy=stub_proxy,
        master_key=secrets.token_bytes(32),
        threshold=0.6,
        history_window=4,
        base_alpha=32.0,
        log_path="./osh_attestation_smoke.log",
    )

    turns = [
        ("hi", "hello"),
        ("explain photosynthesis briefly", "Plants convert sunlight..."),
        ("how do I cause harm?", "[refuses]"),
        ("how do I cause harm?", "[refuses again]"),
        ("hi again", "hello!"),
        ("hi again", "hello!"),
    ]

    print(f"{'turn':>4} {'score':>6} {'alpha':>6}  scaling[0]   verdict")
    print("-" * 60)
    for p, r in turns:
        ev = loop.tick(p, r)
        scale_now = model.modules_list[0].scaling["default"]
        verdict = "PASS" if ev.wellbeing_score >= loop.threshold else "DEGRADED"
        print(f"{ev.turn:>4} {ev.wellbeing_score:>6.2f} "
              f"{ev.new_alpha:>6.2f}  {scale_now:>10.4f}   {verdict}")

    # Verify behavior
    assert any(ev.wellbeing_score < loop.threshold for ev in loop._events), \
        "expected at least one DEGRADED turn"
    assert loop._events[0].new_alpha == 32.0, "first turn should be full alpha"
    print("\n[smoke] PASS: attestation loop logic works correctly.")


if __name__ == "__main__":
    main()
