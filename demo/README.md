# Seeing the difference yourself

## The benchmark

`reliability.py` runs the same tool-calling request many times against raw Ollama
and through toolrails, validates every returned call against the tool's JSON
schema, and prints how many came back valid on each side.

```bash
ollama pull llama3.2:3b          # a model that mangles types on nested schemas
uvx toolrails --port 11500 &     # in another terminal
python reliability.py --model llama3.2:3b --trials 20
```

Small models are stochastic, so the exact counts move between runs; raw lands in
the single digits on the nested-schema tool, toolrails in the high nineties. Try
a model that's already good at tools (`--model qwen2.5:3b`) and you'll see raw
score high and toolrails pass it straight through — it only does work when the
model needs it.

Note: `gemma3` won't work here — Ollama reports it *does not support tools* at
all, which is a different problem toolrails doesn't address.

## Capturing a before/after clip (optional)

If you want a visual for a README or a post, the most credible version is a real
session: point a coding agent (Cline, opencode, Claude Code with a local base
URL) at raw Ollama and drive it into the *Invalid tool parameters* loop, then
point the same agent at `http://localhost:11500/v1` and repeat. Record both and
cut them side by side.
