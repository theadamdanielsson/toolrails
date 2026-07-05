# Contributing to toolrails

## The most useful thing you can send

A tool call that came out wrong. toolrails lives or dies on the range of broken
output it can recognise and fix, and the only way to grow that is real examples.
Open an issue with three things:

- the model (e.g. `llama3.2:3b`),
- the tool schema you passed, and
- what the model produced — the raw `arguments` string, however mangled.

That is a test case. If it's a shape toolrails should have caught and didn't,
it's a bug; if it's one it already fixes, it becomes a regression test so it
stays fixed.

## Running it locally

```bash
git clone https://github.com/theadamdanielsson/toolrails
cd toolrails
uv venv && uv pip install -e ".[dev]"
uv run pytest
```

The tests split in two. `tests/test_schemas.py` and `tests/test_pipeline.py` are
pure and deterministic — they mock the model, so they run in a fraction of a
second and need no Ollama. That's where a new broken-output case should land.

## Measuring against your own models

`demo/reliability.py` runs the same tool-calling request many times against raw
Ollama and through toolrails, and reports how many calls came back valid. Start
the proxy, then point the benchmark at any tool-capable model you have:

```bash
uvx toolrails --port 11500 &
python demo/reliability.py --model llama3.2:3b --trials 20
```

If a model does better or worse than you expect, that number is worth an issue.

## What fits

toolrails fixes the *shape* of tool calls — valid name, valid arguments, working
`tool_choice` — and stays a thin proxy over Ollama. Changes that keep it small
and make it catch more real breakage are welcome. Things that would turn it into
a router, a semantic cache, or a second model judging the first are out of scope
on purpose; that boundary is what keeps it something you can read before you
trust it in front of your agent.

## Fixes should stay fail-open

The one rule that isn't negotiable: toolrails must never turn a working call into
an error. If a change can't fall back to passing the model's original answer
through when something goes wrong, it doesn't go in.
