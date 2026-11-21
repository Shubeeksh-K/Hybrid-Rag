"""Minimal LLM clients. Ollama over HTTP (stdlib only) and a scripted fake for tests."""
from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass, field

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
    num_predict: int | None = None
    think: bool | None = None
    last_done_reason: str | None = field(default=None, init=False)

    def chat(self, system: str, user: str) -> str:
        options = {"temperature": self.temperature, "num_ctx": self.num_ctx}
        if self.num_predict is not None:
            options["num_predict"] = self.num_predict
        body = {"model": self.model, "stream": False,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "options": options}
        if self.think is not None:
            body["think"] = self.think
        req = urllib.request.Request(f"{self.url}/api/chat", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            result = json.loads(resp.read())
        self.last_done_reason = result.get("done_reason")
        return result["message"]["content"].strip()


@dataclass
class FakeLLM(LLM):
    """Returns canned replies in order (last one repeats); records prompts."""
    replies: list[str]
    calls: list[tuple[str, str]] = None

    def chat(self, system: str, user: str) -> str:
        self.calls = (self.calls or []) + [(system, user)]
        return self.replies[min(len(self.calls) - 1, len(self.replies) - 1)]
