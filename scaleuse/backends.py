"""Chat backends: OpenAI-compatible API (Ollama, Groq, OpenRouter) and a Mock respondent.

The Mock backend simulates a respondent with a known latent wellbeing per
item and a known, possibly nonlinear, way of mapping that latent value onto
the response scale. Because the ground truth is known, it lets us check that
the analysis code recovers what it should before spending API credits.
"""

from __future__ import annotations

import collections
import os
import re
import threading
import time

import numpy as np
import requests

from .items import FORMATS, VIGNETTES, SELF_ID

PRESETS = {
    "ollama":     dict(url="http://localhost:11434/v1/chat/completions", key_env=None,                    default_rpm=None),
    "groq":       dict(url="https://api.groq.com/openai/v1/chat/completions",       key_env="GROQ_API_KEY",       default_rpm=25),
    "openrouter": dict(url="https://openrouter.ai/api/v1/chat/completions", key_env="OPENROUTER_API_KEY", default_rpm=None),
}


class _RateLimiter:
    """Sliding-window rate limiter: at most `rpm` requests per 60-second window."""

    def __init__(self, rpm: int | None):
        self._rpm = rpm
        self._lock = threading.Lock()
        self._times: collections.deque = collections.deque()

    def wait(self):
        if not self._rpm:
            return
        with self._lock:
            now = time.monotonic()
            while self._times and now - self._times[0] >= 60.0:
                self._times.popleft()
            if len(self._times) >= self._rpm:
                sleep_until = self._times[0] + 60.0
                time.sleep(max(0.0, sleep_until - now))
                now = time.monotonic()
                while self._times and now - self._times[0] >= 60.0:
                    self._times.popleft()
            self._times.append(time.monotonic())


class ChatAPIBackend:
    """OpenAI-compatible chat backend.

    preset is 'ollama', 'groq', or 'openrouter'. Override url and rpm as needed.
    Retries 429/5xx with capped exponential backoff.
    """

    def __init__(self, preset: str, api_key: str | None = None, rpm: int | None = None,
                 url: str | None = None, max_retries: int = 5, timeout: int = 60):
        p = PRESETS[preset]
        self.url = url or p["url"]
        key_env = p["key_env"]
        self.api_key = api_key or (os.environ.get(key_env) if key_env else None)
        effective_rpm = rpm if rpm is not None else p["default_rpm"]
        self._limiter = _RateLimiter(effective_rpm)
        self.max_retries = max_retries
        self.timeout = timeout

    def chat(self, model: str, system: str, user: str, temperature: float,
             seed: int | None = None, **_) -> str:
        payload = {
            "model": model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": temperature,
            "max_tokens": 16,
        }
        if seed is not None:
            payload["seed"] = seed
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        for attempt in range(self.max_retries):
            self._limiter.wait()
            r = requests.post(self.url, json=payload, headers=headers, timeout=self.timeout)
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"] or ""
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(min(2 ** attempt, 60))
                continue
            r.raise_for_status()
        raise RuntimeError(f"ChatAPI request failed after {self.max_retries} retries")


# Simulated respondents. `gamma` bends the scale (1 = linear use; <1 inflates
# the low end; >1 compresses it), `shift` moves everything up or down,
# `noise` is response noise, `self_latent` is the respondent's own "true" state.
MOCK_PROFILES = {
    "mock/linear": dict(gamma=1.0, shift=0.00, noise=0.04, self_latent=0.70),
    "mock/optimist": dict(gamma=0.6, shift=0.08, noise=0.04, self_latent=0.70),
    "mock/compressed": dict(gamma=1.8, shift=0.00, noise=0.06, self_latent=0.70),
}


class MockBackend:
    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self._latent = {v.vid: (v.level - 1) / 4 for v in VIGNETTES}

    def chat(self, model: str, system: str, user: str, temperature: float, item_id: str = "", fmt: str = "0-10", **_) -> str:
        p = MOCK_PROFILES[model]
        latent = p["self_latent"] if item_id == SELF_ID else self._latent[item_id]
        y = np.clip(latent, 0, 1) ** p["gamma"] + p["shift"]
        y += self.rng.normal(0, p["noise"] * max(temperature, 0.05))
        f = FORMATS[fmt]
        raw = int(round(np.clip(y, 0, 1) * (f.hi - f.lo) + f.lo))
        return str(raw)


_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)
_INT = re.compile(r"-?\d+")


def parse_rating(text: str, lo: int, hi: int) -> int | None:
    """Return the first integer in `text` if it lies in [lo, hi], else None.

    <think>...</think> blocks are stripped before searching.
    """
    text = _THINK.sub("", text or "")
    m = _INT.search(text)
    if not m:
        return None
    x = int(m.group())
    return x if lo <= x <= hi else None
