"""Minimal LLM clients. Ollama over HTTP (stdlib only) and a scripted fake for tests."""
from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass

OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "granite4.2:8b"


class LLM:
    def chat(self, system: str, user: str) -> str:  # pragma: no cover - interface
        raise NotImplementedError


@dataclass
class OllamaLLM(LLM):
    model: str = DEFAULT_MODEL
    url: str = OLLAMA_URL
    temperature: float = 0.0
    num_ctx: int = 8192
    timeout: int = 600

    def chat(self, system: str, user: str) -> str:
        body = {"model": self.model, "stream": False,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "options": {"temperature": self.temperature, "num_ctx": self.num_ctx}}
        req = urllib.request.Request(f"{self.url}/api/chat", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read())["message"]["content"].strip()


@dataclass
class FakeLLM(LLM):
    """Returns canned replies in order (last one repeats); records prompts."""
    replies: list[str]
    calls: list[tuple[str, str]] = None

    def chat(self, system: str, user: str) -> str:
        self.calls = (self.calls or []) + [(system, user)]
        return self.replies[min(len(self.calls) - 1, len(self.replies) - 1)]
