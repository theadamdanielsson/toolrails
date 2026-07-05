# toolrails

**Valid tool calls from any local model.**

<!-- TODO: before/after GIF goes here — the whole pitch.
     Left: gemma/qwen on Ollama emitting broken tool-call JSON, agent stuck in
     an "Invalid tool parameters" retry loop. Right: same model through
     toolrails, every call well-formed on the first try. -->

Local models are good enough to code with now — until they try to call a tool.
A 7B model on Ollama will decide to call `read_file` and then hand your agent
`{"path": "src/app.py",}` with a trailing comma, or wrap it in a ```` ```json ````
fence, or invent a tool named `readFile`, or skip the arguments entirely. The
agent can't parse it, retries, gets the same broken call, and burns your evening
in a loop. (See [ollama/ollama#15390](https://github.com/ollama/ollama/issues/15390):
Claude Code + a local model, stuck on *Invalid tool parameters*, unresolved.)

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

## What it guarantees

- **The tool name is real.** A hallucinated `getWeather` is snapped to the
  `get_weather` you actually offered; a name that matches nothing is left alone
  rather than guessed at.
- **The arguments parse and fit the schema.** When the model's own arguments
  don't validate, toolrails regenerates *just the arguments* under a grammar
  built from the tool's JSON schema, so the output is structurally incapable of
  being malformed JSON or missing a required field.
- **`tool_choice` works again.** Ollama's OpenAI-compatible endpoint silently
  ignores `tool_choice`. toolrails restores it: `"none"` strips the tools,
  `"required"` (or a named function) forces a call even when the model tried to
  answer in prose.

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
([arXiv:2606.25605](https://arxiv.org/abs/2606.25605)). So toolrails splits the
job in two, the way that paper recommends:

1. **Decide, unconstrained.** Ask Ollama normally. Whether and which tool to
   call is entirely the model's choice — no grammar, no tax. If the call it
   produces is already valid, we're done.
2. **Serialize, constrained.** Only once a tool is chosen do we regenerate its
   arguments, this time with the tool's JSON schema in Ollama's `format`
   parameter. Ollama compiles that schema to a grammar (XGrammar) and constrains
   decoding token by token, so the arguments *cannot* come out malformed.

Name repair is deterministic string matching; argument validation is
`jsonschema`. There is no second model judging the first — just a grammar and a
validator.

## Install

```bash
uvx toolrails                 # run without installing
pip install toolrails         # or install the CLI
toolrails --ollama http://localhost:11434 --port 11500
```

Options: `--ollama` (Ollama base URL, or `$OLLAMA_HOST`), `--host`, `--port`.

## Scope

toolrails fixes the *shape* of tool calls: valid name, valid arguments, working
`tool_choice`. It does not make a weak model *choose* the right tool, invent a
call the model didn't attempt, or route between models. If the model decides not
to call a tool, that decision stands (unless you set `tool_choice: required`).
It is a proxy over Ollama specifically, because the leverage is Ollama's
grammar-constrained `format` — the same primitive the guarantee is built on.

Streaming requests are supported: with tools, the response is repaired and then
re-emitted as a stream. Token-by-token streaming under the grammar is a later
refinement — v1 gets the call right first.

## From the same author

toolrails is by the author of [overloop](https://github.com/theadamdanielsson/overloop)
(*stop your agent looping*) and [overllm](https://github.com/theadamdanielsson/overllm)
(*catch the LLM calls you didn't need*). Same theme, one layer down: those stop
wasted agent work; this stops the wasted work of a tool call that never parses.

## License

MIT © Adam Danielsson
