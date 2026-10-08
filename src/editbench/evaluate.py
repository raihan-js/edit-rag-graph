"""Eval harness: greedy generation + alias-match scoring + cost tracking.

Single entry points so every arm runs the identical loop. No LLM judge.
"""

import time

import torch

GEN_KWARGS = {"max_new_tokens": 32, "do_sample": False}


def load_model(model_id="Qwen/Qwen2.5-1.5B-Instruct"):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="cuda:0",
        trust_remote_code=True).eval()
    return model, tok


@torch.inference_mode()
def generate(model, tok, prompts):
    """Greedy completions, batch size 1 (deterministic). Returns (texts, seconds)."""
    out, t0 = [], time.time()
    for p in prompts:
        ids = tok(p, return_tensors="pt").to(model.device)
        n_in = ids["input_ids"].shape[1]
        gen = model.generate(**ids, **GEN_KWARGS,
                             pad_token_id=tok.pad_token_id,
                             eos_token_id=tok.eos_token_id)
        out.append(tok.decode(gen[0][n_in:], skip_special_tokens=True))
    return out, time.time() - t0


def peak_vram_gb():
    return torch.cuda.max_memory_allocated() / 1e9
