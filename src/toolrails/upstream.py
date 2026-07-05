"""The only part of toolrails that talks to Ollama.

Two calls matter:

* `chat_openai` — forward the request to Ollama's OpenAI-compatible endpoint
  and get its natural, *unconstrained* answer. This is stage one: we let the
  model decide whether and which tool to call, because constraining that
  decision is exactly what suppresses tool calls (see the constraint-tax note
  in the README).

* `constrained_object` — Ollama's native `/api/chat` with a JSON schema in
  `format`. Ollama compiles the schema to a grammar (XGrammar) and constrains
  decoding token by token, so the output is *structurally guaranteed* to match.
  This is stage two: once a tool is chosen, we regenerate only its arguments
  under the grammar.
"""

from __future__ import annotations

import json
from typing import Any

import httpx


class Upstream:
    def __init__(self, base_url: str, timeout: float = 600.0) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(timeout=timeout)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def chat_openai(self, body: dict[str, Any]) -> dict[str, Any]:
        """Stage one: unconstrained OpenAI-compatible chat completion."""
        r = await self._client.post(
            f"{self.base_url}/v1/chat/completions", json=body
        )
        r.raise_for_status()
        return r.json()

    async def constrained_object(
        self,
        model: str,
        messages: list[dict[str, Any]],
        schema: dict[str, Any],
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Stage two: grammar-constrained JSON matching `schema`.

        Returns the decoded object, or None if the model produced nothing
        usable even under the grammar (rare, but we stay fail-open).
        """
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "format": schema,
            "stream": False,
            # Arguments should be deterministic given the decision to call the
            # tool — there is no creativity wanted in a function signature.
            "options": {"temperature": 0, **(options or {})},
        }
        r = await self._client.post(f"{self.base_url}/api/chat", json=payload)
        r.raise_for_status()
        content = ((r.json() or {}).get("message") or {}).get("content", "")
        if not content:
            return None
        try:
            val = json.loads(content)
            return val if isinstance(val, dict) else None
        except json.JSONDecodeError:
            return None

    async def passthrough(self, path: str, method: str = "GET") -> httpx.Response:
        """Forward an unrelated endpoint (e.g. /v1/models) verbatim."""
        r = await self._client.request(method, f"{self.base_url}{path}")
        return r
