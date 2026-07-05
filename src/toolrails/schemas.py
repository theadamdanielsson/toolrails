"""Pure, dependency-light helpers for tool-call repair.

Nothing here does I/O or talks to a model — it is all deterministic string and
schema work, so it can be unit-tested on its own and reasoned about in
isolation. The network side lives in `upstream.py`; the orchestration that ties
them together lives in `pipeline.py`.
"""

from __future__ import annotations

import difflib
import json
import re
from typing import Any

try:
    import jsonschema
except ImportError:  # pragma: no cover - jsonschema is a hard dependency
    jsonschema = None  # type: ignore


# --- tool introspection ----------------------------------------------------

def tool_names(tools: list[dict[str, Any]]) -> list[str]:
    """The set of function names the caller offered, in order."""
    names = []
    for t in tools or []:
        fn = t.get("function") or {}
        name = fn.get("name")
        if name:
            names.append(name)
    return names


def schema_for(tools: list[dict[str, Any]], name: str) -> dict[str, Any]:
    """The JSON schema for a named tool's *arguments*.

    OpenAI/Ollama both carry it at function.parameters. A tool with no declared
    parameters gets a permissive empty-object schema so constrained decoding
    still produces valid (empty) JSON rather than failing.
    """
    for t in tools or []:
        fn = t.get("function") or {}
        if fn.get("name") == name:
            params = fn.get("parameters")
            if isinstance(params, dict) and params:
                return params
            return {"type": "object", "properties": {}}
    return {"type": "object", "properties": {}}


def describe(tools: list[dict[str, Any]], name: str) -> str:
    """A tool's human description, for nudging the constrained second pass."""
    for t in tools or []:
        fn = t.get("function") or {}
        if fn.get("name") == name:
            return (fn.get("description") or "").strip()
    return ""


# --- name repair -----------------------------------------------------------

def nearest_name(name: str, valid: list[str], cutoff: float = 0.6) -> str | None:
    """Snap a hallucinated tool name to the closest real one.

    Small local models routinely emit a name that is *almost* right —
    `get_weather` for `getWeather`, `read` for `read_file`. If exactly one
    valid name is close we snap to it; otherwise we return None and let the
    caller decide (toolrails never invents a call the model didn't attempt).
    """
    if not name or not valid:
        return None
    if name in valid:
        return name
    matches = difflib.get_close_matches(name, valid, n=1, cutoff=cutoff)
    return matches[0] if matches else None


# --- argument repair -------------------------------------------------------

_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)
_TRAILING_COMMA = re.compile(r",(\s*[}\]])")


def parse_arguments(raw: Any) -> dict[str, Any] | None:
    """Coerce whatever the model produced for `arguments` into a dict.

    OpenAI sends arguments as a JSON *string*; Ollama's native API sends a dict.
    Weak models send neither cleanly — fenced code, trailing commas, a stray
    sentence in front. We try the strict parse first, then a best-effort
    repair. Returns None if there is nothing recoverable.
    """
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return None

    text = raw.strip()
    if not text:
        return {}

    # 1. strict
    try:
        val = json.loads(text)
        return val if isinstance(val, dict) else None
    except json.JSONDecodeError:
        pass

    # 2. strip markdown fences and retry
    stripped = _FENCE.sub("", text).strip()

    # 3. carve out the outermost {...} object if there's prose around it
    start, end = stripped.find("{"), stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        stripped = stripped[start : end + 1]

    # 4. drop trailing commas before } or ]
    stripped = _TRAILING_COMMA.sub(r"\1", stripped)

    try:
        val = json.loads(stripped)
        return val if isinstance(val, dict) else None
    except json.JSONDecodeError:
        return None


def args_valid(args: dict[str, Any], schema: dict[str, Any]) -> bool:
    """True if `args` satisfies the tool's parameter schema.

    If jsonschema is somehow unavailable we degrade to "is it a dict" rather
    than crashing — toolrails must never be the reason a call fails to go out.
    """
    if not isinstance(args, dict):
        return False
    if jsonschema is None:  # pragma: no cover
        return True
    try:
        jsonschema.validate(args, schema)
        return True
    except jsonschema.ValidationError:
        return False
    except jsonschema.SchemaError:
        # A malformed tool schema is the caller's problem, not the model's —
        # don't block the call over it.
        return True
