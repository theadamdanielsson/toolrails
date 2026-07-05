"""The two-stage tool-call pipeline.

Given an OpenAI-shaped chat request that offers tools, produce a response whose
tool calls are guaranteed well-formed. The shape:

    1. Ask the model naturally (unconstrained). Whether and which tool to call
       is the model's decision — we never constrain that step, because doing so
       is what makes models stop calling tools (the "constraint tax").
    2. For each call it attempted: if the name is real and the arguments already
       validate, keep it as-is (the fast path — zero extra cost). Otherwise snap
       the name to the nearest real tool and regenerate *only the arguments*
       under a grammar built from that tool's schema, which cannot produce
       invalid JSON.
    3. Honour `tool_choice`, which Ollama's OpenAI endpoint drops on the floor:
       `none` strips tools, `required`/a named function forces a call even when
       the model tried to answer in prose.

Every public entry point is wrapped by the caller in a fail-open guard: if
anything in here raises, the proxy falls back to a plain pass-through so it can
never wedge the agent using it.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from . import schemas
from .upstream import Upstream

logger = logging.getLogger("toolrails")


def _nudge(name: str, description: str) -> dict[str, str]:
    """The instruction that steers the constrained second pass toward one tool."""
    desc = f" ({description})" if description else ""
    return {
        "role": "user",
        "content": (
            f"Call the tool `{name}`{desc} given the conversation above. "
            f"Respond with only a JSON object of its arguments."
        ),
    }


def _as_openai_call(name: str, args: dict[str, Any], call_id: str) -> dict[str, Any]:
    """A tool call in the exact shape OpenAI clients expect (arguments = string)."""
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


def _forced_name(tool_choice: Any, names: list[str]) -> str | None:
    """The specific tool named by `tool_choice={"function": {"name": ...}}`."""
    if isinstance(tool_choice, dict):
        fn = tool_choice.get("function") or {}
        name = fn.get("name")
        if name in names:
            return name
    return None


async def _repair_call(
    call: dict[str, Any],
    up: Upstream,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    names: list[str],
) -> dict[str, Any] | None:
    """Return a guaranteed-valid version of one attempted tool call.

    None means the attempt could not be tied to any real tool and the model's
    original output should be left untouched.
    """
    fn = call.get("function") or {}
    raw_name = fn.get("name") or ""
    name = schemas.nearest_name(raw_name, names)
    if name is None:
        logger.warning("unknown tool %r left untouched", raw_name)
        return None
    if name != raw_name:
        logger.info("name %r → %r", raw_name, name)

    schema = schemas.schema_for(tools, name)
    call_id = call.get("id") or f"call_{name}"
    args = schemas.parse_arguments(fn.get("arguments"))

    # Fast path: the model already got it right. No second call.
    if args is not None and schemas.args_valid(args, schema):
        logger.info("call %s ok", name)
        return _as_openai_call(name, args, call_id)

    # Surgical path: fix the *types* of the model's own values (the common
    # small-model failure) without a second model call and without changing what
    # the model meant.
    if args is not None:
        coerced = schemas.coerce(args, schema)
        if schemas.args_valid(coerced, schema):
            logger.info("call %s coerced (fixed argument types)", name)
            return _as_openai_call(name, coerced, call_id)

    # Last resort: regenerate arguments under the grammar. Guaranteed to match
    # the schema, at the cost of one more generation.
    logger.info("call %s regenerated under grammar", name)
    regen = await up.constrained_object(
        model,
        messages + [_nudge(name, schemas.describe(tools, name))],
        schema,
    )
    if regen is None:
        regen = args if args is not None else {}
    return _as_openai_call(name, regen, call_id)


async def _pick_tool(
    up: Upstream,
    model: str,
    messages: list[dict[str, Any]],
    names: list[str],
) -> str:
    """Choose one tool by name when `tool_choice` forces a call but the model
    answered in prose. The decision is already made (a call *must* happen), so
    constraining a name-only choice here carries no tax."""
    if len(names) == 1:
        return names[0]
    selection = await up.constrained_object(
        model,
        messages
        + [{"role": "user", "content": "Which tool should be called? Reply with its name."}],
        {"type": "object", "properties": {"tool": {"enum": names}}, "required": ["tool"]},
    )
    chosen = (selection or {}).get("tool")
    return chosen if chosen in names else names[0]


async def _force_call(
    up: Upstream,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    name: str,
    message: dict[str, Any],
    choice: dict[str, Any],
) -> None:
    """Rewrite `message` in place to be exactly one guaranteed-valid call to
    `name` — the arguments regenerated under that tool's grammar."""
    schema = schemas.schema_for(tools, name)
    args = await up.constrained_object(
        model, messages + [_nudge(name, schemas.describe(tools, name))], schema
    )
    message["content"] = None
    message["tool_calls"] = [_as_openai_call(name, args or {}, f"call_{name}")]
    choice["finish_reason"] = "tool_calls"


async def handle(body: dict[str, Any], up: Upstream) -> dict[str, Any]:
    """Run the pipeline for one (non-streaming) chat-completions request."""
    tools = body.get("tools") or []
    names = schemas.tool_names(tools)
    tool_choice = body.get("tool_choice", "auto")
    model = body.get("model", "")
    messages = body.get("messages", [])

    # No tools, or the caller explicitly forbade them: plain completion.
    if not names or tool_choice == "none":
        clean = {k: v for k, v in body.items() if k not in ("tools", "tool_choice")}
        return await up.chat_openai(clean)

    # Stage one: the model's own, unconstrained answer.
    resp = await up.chat_openai(body)
    try:
        choice = resp["choices"][0]
        message = choice["message"]
    except (KeyError, IndexError):
        return resp  # unfamiliar shape — hand it back untouched

    # A specific tool_choice wins over whatever stage-1 decided to do — even if
    # the model chose a different tool or answered in prose, honour the request.
    forced = _forced_name(tool_choice, names)
    if forced:
        logger.info("forced call %s (tool_choice names it)", forced)
        await _force_call(up, model, messages, tools, forced, message, choice)
        return resp

    calls = message.get("tool_calls")
    if calls:
        repaired = []
        for call in calls:
            fixed = await _repair_call(call, up, model, messages, tools, names)
            repaired.append(fixed if fixed is not None else call)
        message["tool_calls"] = repaired
        return resp

    # The model answered in prose. Force a call only if tool_choice demanded one.
    if tool_choice == "required":
        name = await _pick_tool(up, model, messages, names)
        logger.info("forced call %s (tool_choice=required)", name)
        await _force_call(up, model, messages, tools, name, message, choice)
        return resp

    # `auto` with a prose answer is a legitimate outcome — leave it alone.
    return resp
