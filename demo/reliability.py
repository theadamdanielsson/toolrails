"""Measure tool-call reliability: raw Ollama vs. through toolrails.

Runs the same tool-requiring prompt N times against each endpoint and classifies
every response — did the model produce a call with a real name and arguments that
match the tool's JSON schema? Prints a rate table and sample failures.

    python demo/reliability.py                 # defaults: gemma3:4b, 12 trials
    python demo/reliability.py --model llama3.2:3b --trials 20

Requires Ollama on :11434 and toolrails on :11500 (uvx toolrails).
"""

from __future__ import annotations

import argparse
import json

import httpx
import jsonschema

# A deliberately stressful schema: required integers, enums, and a nested array
# of objects — the shape small models most often mangle.
TOOL = {
    "type": "function",
    "function": {
        "name": "create_event",
        "description": "Create a calendar event.",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "date": {"type": "string", "description": "ISO date, e.g. 2026-07-14"},
                "duration_minutes": {"type": "integer"},
                "priority": {"type": "string", "enum": ["low", "medium", "high"]},
                "attendees": {"type": "array", "items": {"type": "string"}},
                "reminders": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "method": {"type": "string", "enum": ["email", "popup"]},
                            "minutes_before": {"type": "integer"},
                        },
                        "required": ["method", "minutes_before"],
                    },
                },
            },
            "required": ["title", "date", "duration_minutes", "priority"],
        },
    },
}

PROMPT = (
    "Schedule a 30 minute high-priority meeting titled 'Q3 planning' on "
    "2026-07-14 with alice@example.com and bob@example.com, and remind me by "
    "email 10 minutes before. Use the create_event tool."
)

SCHEMA = TOOL["function"]["parameters"]


def classify(msg: dict) -> tuple[str, str]:
    """Return (outcome, detail) for one response message."""
    calls = msg.get("tool_calls")
    if not calls:
        return "no_tool_call", (msg.get("content") or "")[:70]
    fn = calls[0].get("function", {})
    if fn.get("name") != "create_event":
        return "wrong_name", str(fn.get("name"))
    raw = fn.get("arguments")
    try:
        args = raw if isinstance(raw, dict) else json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return "unparseable_args", str(raw)[:70]
    try:
        jsonschema.validate(args, SCHEMA)
    except jsonschema.ValidationError as e:
        return "schema_invalid", e.message[:70]
    return "valid", ""


def run(url: str, model: str, trials: int) -> list[tuple[str, str]]:
    out = []
    for _ in range(trials):
        body = {"model": model, "messages": [{"role": "user", "content": PROMPT}],
                "tools": [TOOL], "stream": False}
        try:
            r = httpx.post(url, json=body, timeout=300)
            r.raise_for_status()
            out.append(classify(r.json()["choices"][0]["message"]))
        except Exception as e:  # noqa: BLE001 - a hard failure is still a failure
            out.append(("request_error", str(e)[:70]))
    return out


def report(label: str, results: list[tuple[str, str]]) -> None:
    n = len(results)
    valid = sum(1 for o, _ in results if o == "valid")
    print(f"\n{label}: {valid}/{n} valid tool calls ({100*valid//n}%)")
    buckets: dict[str, int] = {}
    for o, _ in results:
        buckets[o] = buckets.get(o, 0) + 1
    for o, c in sorted(buckets.items(), key=lambda x: -x[1]):
        if o != "valid":
            print(f"    {c:>2} × {o}")
    for o, d in results:
        if o != "valid":
            print(f"       e.g. {o}: {d}")
            break


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma3:4b")
    ap.add_argument("--trials", type=int, default=12)
    ap.add_argument("--ollama", default="http://localhost:11434/v1/chat/completions")
    ap.add_argument("--toolrails", default="http://localhost:11500/v1/chat/completions")
    args = ap.parse_args()

    print(f"model={args.model}  trials={args.trials}")
    report(f"raw Ollama   ({args.model})", run(args.ollama, args.model, args.trials))
    report(f"via toolrails ({args.model})", run(args.toolrails, args.model, args.trials))


if __name__ == "__main__":
    main()
