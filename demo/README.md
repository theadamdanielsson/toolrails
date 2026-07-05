# The demo GIF is the pitch

toolrails lives or dies on one 10–15s before/after GIF at the top of the README:
a local model failing to call a tool, then the same model through toolrails
getting it right on the first try. Capture it two ways, best first.

## 1. Real in-session (best, most credible)

1. Pull a small, tool-flaky model: `ollama pull llama3.2:3b` (a good demo model —
   it supports tools but mangles types on nested schemas). Note gemma3 won't
   work here: Ollama reports it *does not support tools* at all, which is a
   different problem toolrails v1 doesn't address.
2. Point a coding agent (Cline, opencode, or Claude Code with a local base URL)
   straight at Ollama. Drive it until it hits the *Invalid tool parameters*
   loop — a task that needs a tool call usually does it within a turn or two.
   Screen-record the loop.
3. Start toolrails (`uvx toolrails`), point the same agent at
   `http://localhost:11500/v1`, repeat the same task. Record the clean call.
4. Cut the two side by side. That split-screen is the whole launch.

## 2. Scripted terminal cast (fallback, reproducible)

`demo.tape` renders a scripted before/after to a GIF with
[vhs](https://github.com/charmbracelet/vhs), so it ships with zero live session:

```bash
vhs demo/demo.tape        # writes demo/toolrails.gif
```

It sends one hand-built broken tool call and one repaired one against a running
Ollama, so the fix is visible without needing to reproduce the loop live.
