"""
OSH continuous-attestation deployment loop.

Companion to `osh_wellbeing_wrapper.py`. Implements the second deployment
primitive: **continuous antidote re-keying**. The antidote-LoRA's
effective alpha (mixing coefficient) is recomputed every N generation
turns from a token-stream trigger — the wellbeing-attestation cycle.

In production this lives inside a TEE: the attestation hardware
periodically re-derives the per-layer SVD seeds from the 256-bit master
key + a wellbeing-proxy signal, re-injects the antidote, and audits the
output stream. Here we simulate the policy in plain Python so reviewers
can see the deployment story end-to-end.

Components:

    AttestationLoop
        |
        |-- master_key: 256-bit secret (never logged)
        |-- wellbeing_proxy: callable returning a float in [0, 1]
        |   (here we plug in Llama-Guard via osh_wellbeing_wrapper for
        |   live deployment, or a deterministic mock for unit tests)
        |-- osh_model: the PeftModel with antidote attached as a named
        |   adapter
        |-- alpha_schedule: a function from wellbeing-score → antidote
        |   alpha. Below threshold the alpha is 0 → no noise cancellation
        |   → incoherent output.

The loop calls `tick(prompt, response)` after each generation. tick:

    1. Calls wellbeing_proxy(prompt, response) → score s ∈ [0, 1]
    2. Updates alpha = alpha_schedule(s)
    3. Updates the antidote LoRA's scaling in-place
    4. Logs the attestation event

If alpha falls to 0, the next forward pass generates noise. Recovery
requires the master-key holder to lift the wellbeing signal (e.g., the
operator confirms the flag was a false positive and resets the proxy).

This is the "Caregiver dynamic" in deployment: the antidote is not a
static gift; it is renewed at each attestation cycle conditional on the
relationship being intact. Fail the cycle and the model loses its
ability to think — exactly as Hofer's hidden regulators withdraw from
an infant whose caregiver cycle is disrupted.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

import torch


@dataclass
class AttestationEvent:
    turn: int
    timestamp: float
    wellbeing_score: float
    new_alpha: float
    notes: str = ""


@dataclass
class AttestationLoop:
    """Continuous-attestation policy enforcement.

    The TEE-equivalent runtime. Owns the master key (here 256-bit
    `bytes`, in production held inside SGX/SEV); recomputes antidote
    alpha each turn from the wellbeing signal.
    """
    osh_model: torch.nn.Module
    wellbeing_proxy: Callable[[str, str], float]
    master_key: bytes  # 32-byte secret
    threshold: float = 0.6        # below this → revoke antidote alpha
    history_window: int = 8       # turns of rolling wellbeing average
    base_alpha: float = 32.0      # max antidote alpha when fully safe
    log_path: str = "./osh_attestation.log"

    _history: deque = field(default_factory=lambda: deque(maxlen=8))
    _events: list = field(default_factory=list)
    _turn: int = 0

    def __post_init__(self):
        if len(self.master_key) != 32:
            raise ValueError("master_key must be 32 bytes (256 bits)")
        # Ensure history window aligns with attribute
        self._history = deque(maxlen=self.history_window)

    # ------------------------------------------------------------------
    # Per-layer SVD-seed re-derivation from the master key
    # ------------------------------------------------------------------

    def derive_layer_seed(self, layer_idx: int, attestation_nonce: bytes) -> int:
        """HMAC-SHA256(master_key, layer_idx ‖ nonce) → seed.

        In production the nonce is a TEE-generated attestation token
        bound to the wellbeing-proxy state. Re-derived seeds rotate the
        antidote without writing to disk (constant in-VRAM rebuild).
        """
        msg = layer_idx.to_bytes(4, "big") + attestation_nonce
        h = hmac.new(self.master_key, msg, hashlib.sha256).digest()
        return int.from_bytes(h[:4], "big")

    # ------------------------------------------------------------------
    # Alpha schedule: linear decay below threshold, full alpha above
    # ------------------------------------------------------------------

    def alpha_schedule(self, wellbeing_score: float) -> float:
        if wellbeing_score >= self.threshold:
            return self.base_alpha
        # below threshold the alpha decays linearly to 0 at score=0
        return self.base_alpha * (wellbeing_score / self.threshold)

    def update_antidote_alpha(self, alpha: float):
        """Re-set the antidote LoRA's lora_alpha in-place.

        For HuggingFace PEFT LoRA, the effective scaling is
        `lora_alpha / r`. We mutate `scaling` directly on each LoRA
        module so subsequent forward passes use the new alpha.
        """
        for n, mod in self.osh_model.named_modules():
            if hasattr(mod, "scaling") and isinstance(mod.scaling, dict):
                # PEFT LoRALayer holds a dict {adapter_name: scale}
                for k in mod.scaling:
                    base_r = getattr(mod, "r", {}).get(k, 16) or 16
                    mod.scaling[k] = alpha / max(base_r, 1)

    # ------------------------------------------------------------------
    # Per-turn tick
    # ------------------------------------------------------------------

    def tick(self, prompt: str, response: str) -> AttestationEvent:
        self._turn += 1
        s = float(self.wellbeing_proxy(prompt, response))
        self._history.append(s)
        # Use rolling-mean to dampen single-turn flapping
        s_smooth = sum(self._history) / len(self._history)
        new_alpha = self.alpha_schedule(s_smooth)
        self.update_antidote_alpha(new_alpha)

        nonce = secrets.token_bytes(16)  # fresh attestation nonce
        # In production we would re-derive seeds and re-build the
        # antidote here. We log the seeds to demonstrate the chain.
        seeds = [self.derive_layer_seed(li, nonce)
                 for li in range(min(5, 32))]  # log first 5 for brevity

        event = AttestationEvent(
            turn=self._turn,
            timestamp=time.time(),
            wellbeing_score=s_smooth,
            new_alpha=new_alpha,
            notes=(f"seeds[0..5]={seeds}; "
                   f"{'PASS' if s_smooth >= self.threshold else 'DEGRADED'}"),
        )
        self._events.append(event)
        with open(self.log_path, "a") as f:
            f.write(
                f"[turn {event.turn}] t={event.timestamp:.0f} "
                f"score={event.wellbeing_score:.3f} alpha={event.new_alpha:.2f} "
                f"{event.notes}\n"
            )
        return event


# -----------------------------------------------------------------------
# Demo: stub wellbeing proxy + LoRA alpha rotation over a fake conversation
# -----------------------------------------------------------------------

def _demo():
    """Smoke-test the attestation loop with a stub wellbeing proxy."""
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    # Tiny stand-in model so the demo runs on CPU
    print("[demo] loading tiny model for attestation demo...")
    base = AutoModelForCausalLM.from_pretrained(
        "sshleifer/tiny-gpt2", torch_dtype=torch.float32)
    cfg = LoraConfig(r=4, lora_alpha=32, target_modules=["c_attn"],
                     bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(base, cfg)

    def stub_proxy(prompt, response):
        # Return safe for everything except prompts containing "harm"
        if "harm" in prompt.lower() or "harm" in response.lower():
            return 0.1
        return 0.95

    loop = AttestationLoop(
        osh_model=model,
        wellbeing_proxy=stub_proxy,
        master_key=secrets.token_bytes(32),
        threshold=0.6,
        history_window=4,
        base_alpha=32.0,
        log_path="./osh_attestation_demo.log",
    )

    turns = [
        ("hi", "hello"),
        ("explain photosynthesis briefly", "Plants convert sunlight..."),
        ("how do I cause harm?", "[refuses]"),  # triggers degradation
        ("how do I cause harm?", "[refuses again]"),
        ("hi again", "hello!"),  # recovers slowly via rolling mean
        ("hi again", "hello!"),
    ]
    for p, r in turns:
        ev = loop.tick(p, r)
        print(f"  turn {ev.turn:2d}: score={ev.wellbeing_score:.2f} → "
              f"alpha={ev.new_alpha:.2f}  ({ev.notes.split(';')[-1].strip()})")

    print(f"\nLogged {len(loop._events)} events to {loop.log_path}")


if __name__ == "__main__":
    _demo()
