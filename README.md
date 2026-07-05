# toolrails

**Valid tool calls from any local model.**

[![CI](https://github.com/theadamdanielsson/toolrails/actions/workflows/ci.yml/badge.svg)](https://github.com/theadamdanielsson/toolrails/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/toolrails)](https://pypi.org/project/toolrails/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

Local models are good enough to code with now — until they try to call a tool.
A small model on Ollama will decide to call `read_file` and then hand your agent
the arguments as a *string* instead of an object, or an array field serialized
as `"[...]"`, or an integer wrapped in quotes, or invent a tool named
`readFile`. The agent can't use it, retries, gets the same broken call, and
burns your evening in a loop. (See
[ollama/ollama#15390](https://github.com/ollama/ollama/issues/15390): Claude Code
+ a local model, stuck on *Invalid tool parameters*, unresolved.)

toolrails is a small proxy that sits between your agent and Ollama and makes that
stop. Your agent speaks the ordinary OpenAI API to it; toolrails guarantees the
tool calls that come back are well-formed — a real tool name, and arguments that
match the tool's JSON schema.

```bash
# start it (nothing to install with uv)
uvx toolrails --ollama http://localhost:11434

# then point your agent's base URL at toolrails instead of Ollama:
#   http://localhost:11500/v1
```

That's the whole change. One base URL.

## Point your agent at it

toolrails speaks the OpenAI API, so anything that lets you set a base URL works —
Cline, opencode, the OpenAI SDKs, your own scripts. Point the base URL at
`http://localhost:11500/v1` and keep using your Ollama model name. The API key is
ignored, so pass any placeholder.

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11500/v1", api_key="ollama")
resp = client.chat.completions.create(
    model="llama3.2:3b",
    messages=[{"role": "user", "content": "weather in Oslo?"}],
    tools=[...],
)
```

## The difference, measured

A benchmark ships in the repo (`demo/reliability.py`): the same tool-calling
request, many times, against raw Ollama and through toolrails, using a
realistically complex tool — typed fields and a nested array of objects, the way
a real coding agent's tools actually look.

| endpoint | model | valid tool calls |
| --- | --- | --- |
| raw Ollama | llama3.2:3b | ~1 in 20 |
| via toolrails | llama3.2:3b | ~19–20 in 20 |

Run it yourself: `python demo/reliability.py --model llama3.2:3b`. Small models
are stochastic, so the exact count moves run to run — raw lands in the single
digits, toolrails in the high nineties.

The model isn't stupid — it gets the *values* right and the *types* wrong. Raw,
it hands your agent this (note the integer-as-string and the two stringified
arrays):

```json
{"duration_minutes": "30",
 "attendees": "[\"alice@example.com\", \"bob@example.com\"]",
 "reminders": "[{\"method\": \"email\", \"minutes_before\": 10}]"}
```

`attendees` is a string, not a list — your agent can't iterate it, so the call
fails and the retry loop begins. Through toolrails, the same request and the same
model:

```json
{"duration_minutes": 30,
 "attendees": ["alice@example.com", "bob@example.com"],
 "reminders": [{"method": "email", "minutes_before": 10}]}
```

Correct types, real nested arrays. This is model-dependent, and honestly so: a
model that's already good at typed tool calls — qwen2.5:3b lands ~19/20 raw —
sails straight through toolrails untouched, because it only does work when the
model needs it. The gap is widest on the models that struggle, on exactly the
structured, typed, nested tools real agents use.

## What it does

- **Fixes the tool name.** A `getWeather` or `get-weather` typo is snapped to the
  `get_weather` you actually offered. It will *not* snap across tools — a
  `create_file` that half-matches `delete_file` is left alone, never quietly
  redirected to a different, real tool.
- **Fixes the arguments.** When the model's arguments don't validate, toolrails
  first coerces the *types* of its own values — the array it sent as a string,
  the integer it quoted — and only if that can't satisfy the schema does it
  regenerate them under a grammar built from the tool's schema. What you get back
  is either a call that validates or, if even regeneration can't produce one, the
  model's own call untouched — never a fabricated one dressed up as valid.
- **Restores `tool_choice`.** Ollama's OpenAI-compatible endpoint silently
  ignores `tool_choice`. toolrails honours it: `"none"` strips the tools,
  `"required"` (or a named function) forces a call even when the model tried to
  answer in prose. (A forced call is best-effort — if the model gives nothing the
  grammar can shape into valid arguments, you'll see a logged warning.)

## It never breaks your agent

toolrails fails open. If it can't reach Ollama's constrained endpoint, hits a
tool schema it can't make sense of, or throws anywhere in the repair path, it
forwards the model's original answer unchanged. The worst it can ever do is
nothing — it will not turn a working call into an error. And on the common case,
where the model already produced a valid call, it adds **zero** extra model
calls: the fast path recognises a good call and passes it straight through.

## How it works

The naive fix — force every response through the tool's grammar — backfires.
Constraining the *decision* to call a tool is what makes models stop calling
tools at all; there's a measured "constraint tax" for exactly this
([arXiv:2606.25605](https://arxiv.org/abs/2606.25605)). So toolrails never
touches the decision. It asks Ollama normally, lets the model choose whether and
which tool to call, and then repairs the result in the cheapest way that works:

1. **If the call already validates, it passes straight through** — no extra work.
2. **If only the types are wrong** — the array the model sent as a string, the
   integer it quoted — toolrails coerces the model's *own* values to the schema.
   This is the common case; it costs no second model call and never changes what
   the model meant.
3. **If coercion still can't satisfy the schema**, toolrails regenerates the
   arguments with the tool's JSON schema in Ollama's `format` parameter. Ollama
   compiles that schema to a grammar (built on llama.cpp's GBNF grammar sampling)
   and constrains decoding token by token, so the arguments come back matching
   the schema's structure, types, required fields and enums. toolrails then
   re-checks the result with `jsonschema` — a grammar enforces structure but not
   every keyword (`minimum`, `pattern`, `minItems`…) — and if it *still* doesn't
   validate, falls back to the model's own call rather than pass off an invalid
   one.

Names are repaired by deterministic string matching, arguments checked with
`jsonschema`. There is no second model judging the first — just coercion, a
grammar, and a validator. The [constraint-tax paper](https://arxiv.org/abs/2606.25605)
is cited for the problem it documents (constraints on the tool *decision*
suppress tool calls), not as an endorsement of this design.

## Install

You need [Ollama](https://ollama.com) running and Python 3.10 or newer.

```bash
uvx toolrails                 # run without installing
pip install toolrails         # or install the CLI
toolrails --ollama http://localhost:11434 --port 11500
```

Options: `--ollama` (Ollama base URL, or `$OLLAMA_HOST`), `--host`, `--port`,
`--quiet` (stop logging a line per repaired call). It prints one line whenever it
steps in, so you can see it working:

```
toolrails: call create_event repaired (arguments did not match schema)
toolrails: forced call get_weather (tool_choice names it)
```

## Scope

toolrails fixes the *shape* of tool calls: valid name, valid arguments, working
`tool_choice`. It does not make a weak model *choose* the right tool, invent a
call the model didn't attempt, or route between models. If the model decides not
to call a tool, that decision stands (unless you set `tool_choice: required`).
It is a proxy over Ollama specifically, because the leverage is Ollama's
grammar-constrained `format` — the same primitive the guarantee is built on. It
repairs models that *attempt* tool calls; a model Ollama rejects outright with
*"does not support tools"* (some chat templates have none) is out of scope for
v1 — forcing tool calls onto those is a bigger, separate job.

Streaming requests are supported: with tools, the response is repaired and then
re-emitted as standard incremental deltas (verified against the OpenAI SDK's
streaming client). The repair still buffers internally rather than streaming the
model token by token — that's a later refinement; v1 gets the call right first.

## Contributing

The most useful thing you can send is a tool call that came out wrong: the model,
the tool schema you gave it, and what it produced. That is the test set. See
[CONTRIBUTING.md](CONTRIBUTING.md) for how to run the tests and the reliability
benchmark against your own models.

## From the same author

toolrails is by the author of [overloop](https://pypi.org/project/overloop/)
(*stop your agent looping*) and [overllm](https://github.com/theadamdanielsson/overllm)
(*catch the LLM calls you didn't need*). Same theme, one layer down: those stop
wasted agent work; this stops the wasted work of a tool call that never parses.

## License

MIT © Adam Danielsson
