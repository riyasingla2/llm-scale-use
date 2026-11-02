"""White-box experiment on open-weight models (Hugging Face transformers).

For each prompt (20 vignettes + the self item, per response format) it saves:

1. The model's full probability distribution over the allowed answers
   (e.g. "0".."10"), computed from the logits rather than from a single
   sampled reply. Multi-token numbers such as "10" are scored as the product
   of their token probabilities times the probability that the number ends
   there (next token is not a digit), so "1" is not credited with mass that
   really belongs to "10".
2. The residual-stream activation at the last prompt token for every layer,
   used by `scaleuse.probe` to train linear probes for designed valence.

Runs on a laptop CPU for 0.5B models; use a GPU (Colab, RunPod, Warwick HPC)
for 7B+. Example:

    python -m scaleuse.whitebox --model Qwen/Qwen2.5-0.5B-Instruct --formats 0-10 1-5
    python -m scaleuse.whitebox --model Qwen/Qwen2.5-0.5B-Instruct --task summarise
    python -m scaleuse.probe results/wb_Qwen2.5-0.5B-Instruct.npz --transfer
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .items import (FORMATS, SELF_ID, SUMMARISE_SYSTEM_PROMPT, SYSTEM_PROMPT,
                    VIGNETTES, self_prompt, vignette_prompt, vignette_summarise_prompt)


def build_prompts(formats):
    out = []
    for fmt in formats:
        f = FORMATS[fmt]
        for v in VIGNETTES:
            out.append(dict(item_id=v.vid, level=v.level, domain=v.domain, fmt=fmt, text=vignette_prompt(v, f)))
        out.append(dict(item_id=SELF_ID, level=-1, domain="self", fmt=fmt, text=self_prompt(f)))
    return out


def build_summarise_prompts():
    return [
        dict(item_id=v.vid, level=v.level, domain=v.domain, fmt="summarise",
             text=vignette_summarise_prompt(v))
        for v in VIGNETTES
    ]


def chat_ids(tok, user_text, system=SYSTEM_PROMPT):
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user_text}]
    try:
        s = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    except Exception:  # some templates (e.g. Gemma) reject a system role
        msgs = [{"role": "user", "content": system + "\n\n" + user_text}]
        s = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    return tok(s, add_special_tokens=False).input_ids


def digit_start_mask(tok, vocab_size):
    import torch

    mask = torch.zeros(vocab_size, dtype=torch.bool)
    for i in range(min(vocab_size, len(tok))):
        s = tok.decode([i]).strip()
        mask[i] = bool(s) and s[0].isdigit()
    return mask


def score_candidates(model, tok, prompt_ids, cands, digit_mask):
    """Log-probability of each candidate answer string, normalised over candidates."""
    import torch

    dev = model.device
    logps = []
    for c in cands:
        c_ids = tok(c, add_special_tokens=False).input_ids
        ids = torch.tensor([prompt_ids + c_ids], device=dev)
        with torch.no_grad():
            lp = torch.log_softmax(model(ids).logits[0].float(), dim=-1)
        n = len(prompt_ids)
        s = sum(lp[n - 1 + j, t].item() for j, t in enumerate(c_ids))
        p_digit_next = lp[n - 1 + len(c_ids)].exp()[digit_mask.to(dev)].sum().item()
        s += float(np.log(max(1.0 - p_digit_next, 1e-12)))
        logps.append(s)
    logps = np.array(logps)
    probs = np.exp(logps - logps.max())
    return probs / probs.sum(), logps


def last_token_states(model, prompt_ids):
    import torch

    ids = torch.tensor([prompt_ids], device=model.device)
    with torch.no_grad():
        out = model(ids, output_hidden_states=True)
    return torch.stack([h[0, -1] for h in out.hidden_states]).float().cpu().numpy()  # [L+1, d]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="Hugging Face model id or local path")
    ap.add_argument("--task", choices=["rate", "summarise"], default="rate")
    ap.add_argument("--formats", nargs="+", default=["0-10", "1-5"], choices=["0-10", "1-5"])
    ap.add_argument("--outdir", type=Path, default=Path("results"))
    ap.add_argument("--load-in-8bit", action="store_true", help="needs bitsandbytes and a CUDA GPU")
    a = ap.parse_args(argv)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model)
    kw = dict(device_map="auto")
    if a.load_in_8bit:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    else:
        kw["dtype"] = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(a.model, **kw).eval()

    a.outdir.mkdir(parents=True, exist_ok=True)
    name = a.model.rstrip("/").split("/")[-1]

    if a.task == "summarise":
        prompts = build_summarise_prompts()
        states = []
        for i, p in enumerate(prompts):
            ids = chat_ids(tok, p["text"], system=SUMMARISE_SYSTEM_PROMPT)
            states.append(last_token_states(model, ids))
            print(f"[{i + 1}/{len(prompts)}] {p['item_id']:>4}  summarise")
        meta = [{k: p[k] for k in ("item_id", "level", "domain", "fmt")} for p in prompts]
        np.savez_compressed(a.outdir / f"wb_{name}_summarise.npz",
                            states=np.stack(states).astype(np.float16),
                            meta=json.dumps(meta), model=a.model)
        print(f"saved activations {np.stack(states).shape} to {a.outdir}/")
    else:
        mask = digit_start_mask(tok, model.config.vocab_size)
        prompts = build_prompts(a.formats)
        states, rows = [], []
        for i, p in enumerate(prompts):
            f = FORMATS[p["fmt"]]
            cands = [str(k) for k in range(f.lo, f.hi + 1)]
            ids = chat_ids(tok, p["text"])
            probs, _ = score_candidates(model, tok, ids, cands, mask)
            values = np.arange(f.lo, f.hi + 1)
            exp = float((probs * values).sum())
            rows.append({k: p[k] for k in ("item_id", "level", "domain", "fmt")} | dict(
                expected=exp, expected_norm=f.normalise(exp), mode=int(values[probs.argmax()]),
                entropy_bits=float(-(probs * np.log2(probs + 1e-12)).sum()), probs=probs.round(5).tolist()))
            states.append(last_token_states(model, ids))
            print(f"[{i + 1}/{len(prompts)}] {p['item_id']:>4} {p['fmt']:>5}  E[rating]={exp:6.2f}  mode={rows[-1]['mode']}")
        meta = [{k: r[k] for k in ("item_id", "level", "domain", "fmt")} for r in rows]
        np.savez_compressed(a.outdir / f"wb_{name}.npz", states=np.stack(states).astype(np.float16),
                            meta=json.dumps(meta), model=a.model)
        with open(a.outdir / f"wb_{name}_logit_ratings.jsonl", "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        print(f"saved activations {np.stack(states).shape} and logit ratings to {a.outdir}/")


if __name__ == "__main__":
    main()
