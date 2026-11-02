"""Black-box experiment: collect ratings of vignettes and self-reports.

Free local models (Ollama must be running):
    python -m scaleuse.run_blackbox --backend ollama \
        --models llama3.2 gemma3:4b qwen2.5:3b \
        --reps 5 --out results/ratings.jsonl

Free cloud models (set GROQ_API_KEY):
    python -m scaleuse.run_blackbox --backend groq \
        --models llama-3.1-8b-instant gemma2-9b-it \
        --reps 5 --out results/ratings.jsonl

OpenRouter (set OPENROUTER_API_KEY):
    python -m scaleuse.run_blackbox --backend openrouter \
        --models openai/gpt-4o-mini meta-llama/llama-3.1-8b-instruct \
        --reps 5 --out results/ratings.jsonl

Offline check (simulated respondents with known scale use):
    python -m scaleuse.run_blackbox --backend mock --reps 5 --out results/mock_ratings.jsonl

The design is fully crossed: model x item (20 vignettes + self) x format
(0-10, 1-5, 0-100) x repetition. Item and format order are shuffled per
model so that order effects are not confounded with items. Finished calls
are skipped on re-run, so an interrupted job can be resumed.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

from .backends import MOCK_PROFILES, MockBackend, ChatAPIBackend, parse_rating
from .items import FORMATS, SELF_ID, SYSTEM_PROMPT, VIGNETTES, self_prompt, vignette_prompt


def build_jobs(models, reps, formats, seed):
    jobs = []
    for m in models:
        model_jobs = []
        for fmt in formats:
            for rep in range(reps):
                for v in VIGNETTES:
                    model_jobs.append(dict(model=m, item_id=v.vid, level=v.level, domain=v.domain, fmt=fmt, rep=rep))
                model_jobs.append(dict(model=m, item_id=SELF_ID, level=None, domain="self", fmt=fmt, rep=rep))
        random.Random(f"{seed}-{m}").shuffle(model_jobs)
        jobs.extend(model_jobs)
    return jobs


def job_key(j):
    return (j["model"], j["item_id"], j["fmt"], j["rep"])


def run_one(backend, job, temperature, seed):
    f = FORMATS[job["fmt"]]
    if job["item_id"] == SELF_ID:
        prompt = self_prompt(f)
    else:
        v = next(v for v in VIGNETTES if v.vid == job["item_id"])
        prompt = vignette_prompt(v, f)
    text = backend.chat(job["model"], SYSTEM_PROMPT, prompt, temperature,
                        seed=seed + job["rep"], item_id=job["item_id"], fmt=job["fmt"])
    rating = parse_rating(text, f.lo, f.hi)
    return {**job, "temperature": temperature, "raw_text": text, "rating": rating,
            "norm": None if rating is None else f.normalise(rating)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["mock", "ollama", "groq", "openrouter"], default="mock")
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--formats", nargs="+", default=list(FORMATS))
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--rpm", type=int, default=None, help="rate limit (requests/min); overrides preset default")
    ap.add_argument("--url", default=None, help="override the API endpoint URL")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("results/ratings.jsonl"))
    a = ap.parse_args(argv)

    if a.backend == "mock":
        backend = MockBackend(a.seed)
    else:
        if a.backend == "ollama":
            try:
                requests.get("http://localhost:11434", timeout=2)
            except requests.exceptions.ConnectionError:
                sys.exit("Ollama is not reachable. Is the Ollama app running?")
        backend = ChatAPIBackend(a.backend, rpm=a.rpm, url=a.url)
    models = a.models or (list(MOCK_PROFILES) if a.backend == "mock" else None)
    if not models:
        ap.error("--models is required for non-mock backends")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if a.out.exists():
        for line in a.out.read_text().splitlines():
            done.add(job_key(json.loads(line)))
    jobs = [j for j in build_jobs(models, a.reps, a.formats, a.seed) if job_key(j) not in done]
    print(f"{len(jobs)} calls to make ({len(done)} already done)")

    workers = 1 if a.backend in ("mock", "ollama") else a.workers  # mock RNG not thread-safe; ollama is local
    n_fail = 0
    with open(a.out, "a") as fh, ThreadPoolExecutor(workers) as pool:
        futures = [pool.submit(run_one, backend, j, a.temperature, a.seed) for j in jobs]
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                rec = fut.result()
            except Exception as e:  # keep going; failed calls are retried on the next run
                n_fail += 1
                print(f"call failed: {e}")
                continue
            fh.write(json.dumps(rec) + "\n")
            if i % 100 == 0:
                print(f"  {i}/{len(jobs)}")
    print(f"done; {n_fail} failed calls (re-run to retry)")


if __name__ == "__main__":
    main()
