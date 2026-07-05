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

def _canon(n: str) -> str:
    """Casing/separator-insensitive form: getWeather == get_weather == get-weather."""
    return re.sub(r"[_\-]", "", n).lower()


def _lead(n: str) -> str:
    """The leading verb token, splitting on both `_` and camelCase, lowercased.
    `create_file` -> "create", `createFile` -> "create", `getWeather` -> "get"."""
    n = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", n.replace("-", "_"))
    return n.split("_")[0].lower()


def nearest_name(name: str, valid: list[str], cutoff: float = 0.6) -> str | None:
    """Snap a *typo* of a real tool name to the real one — never guess across
    tools.

    Small models emit `getWeather` for `get_weather` or `read` for `read_file`;
    those are safe to snap. What is *not* safe — and what a naive closest-match
    does — is snapping `create_file` to `delete_file` or `enable_user` to
    `disable_user`: lexically near, semantically opposite, potentially
    destructive. So we snap only when it is clearly the same name:

    * an exact match once casing and separators are ignored, or
    * a close match that shares the same leading verb and clearly beats the
      runner-up (no near-ties).

    Anything else returns None, and the caller leaves the model's call untouched
    rather than fire a different, real tool the model never asked for.
    """
    if not name or not valid:
        return None
    if name in valid:
        return name

    # Same name up to casing/underscores/hyphens — always safe.
    canon = {}
    for v in valid:
        canon.setdefault(_canon(v), v)
    if _canon(name) in canon:
        return canon[_canon(name)]

    matches = difflib.get_close_matches(name, valid, n=2, cutoff=cutoff)
    if not matches:
        return None
    best = matches[0]
    # Refuse to cross a different leading verb (create/delete, get/set, enable/disable).
    if _lead(name) != _lead(best):
        return None
    # Refuse near-ties: if two real tools are about equally close, don't guess.
    if len(matches) > 1:
        r1 = difflib.SequenceMatcher(None, name, matches[0]).ratio()
        r2 = difflib.SequenceMatcher(None, name, matches[1]).ratio()
        if r1 - r2 < 0.1:
            return None
    return best


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


def coerce(value: Any, schema: dict[str, Any], _depth: int = 0) -> Any:
    """Nudge the model's own value toward the schema's type without inventing
    anything. This fixes the failure we actually see from small models — the
    right value with the wrong type: an integer sent as `"30"`, an array or
    object serialized into a `"[...]"` string, a boolean as `"true"`. It walks
    the schema recursively and only ever *reshapes* values it already has; it
    never fills in a missing one. Anything it can't confidently convert is left
    untouched for validation (and, failing that, grammar regeneration) to catch.
    """
    if _depth > 16 or not isinstance(schema, dict):
        return value
    t = schema.get("type")
    if isinstance(t, list):
        # Nullable/union type like ["integer", "null"] — the standard optional
        # field. If exactly one non-null type is offered, coerce toward it.
        non_null = [x for x in t if x != "null"]
        if len(non_null) == 1:
            return coerce(value, {**schema, "type": non_null[0]}, _depth + 1)
        return value

    # A structured value the model flattened into a JSON string ("[...]", "{...}").
    if t in ("object", "array") and isinstance(value, str):
        try:
            value = json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return value

    if t == "integer":
        if isinstance(value, bool):
            return value
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, str):
            try:
                return int(value.strip())
            except ValueError:
                try:
                    f = float(value.strip())
                    return int(f) if f.is_integer() else value
                except ValueError:
                    return value
        return value

    if t == "number":
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value
        if isinstance(value, str):
            try:
                return float(value.strip())
            except ValueError:
                return value
        return value

    if t == "boolean":
        if isinstance(value, str):
            low = value.strip().lower()
            if low in ("true", "yes", "1"):
                return True
            if low in ("false", "no", "0"):
                return False
        return value

    if t == "array" and isinstance(value, list):
        items = schema.get("items")
        return [coerce(v, items, _depth + 1) for v in value] if isinstance(items, dict) else value

    if isinstance(value, dict):  # object, or a schema with properties but no type
        props = schema.get("properties")
        if isinstance(props, dict):
            return {k: (coerce(v, props[k], _depth + 1) if k in props else v)
                    for k, v in value.items()}

    return value


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
    except Exception:
        # A schema we can't even evaluate — malformed, an unresolvable $ref, or
        # too deep — is the caller's problem, not the model's. Don't block the
        # call over it (and never let it escape and abort the whole repair).
        return True
