# Changelog

All notable changes to toolrails are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to
follow [semantic versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] — 2026-07-05

Hardening pass after an adversarial review of the 0.1.0 code and docs.

### Fixed

- Name repair no longer snaps across tools. A hallucinated name is corrected only
  when it's the same name up to casing/separators, or a close match that shares
  the same leading verb with no near-tie — so `create_file` is never redirected
  to `delete_file`.
- Regenerated arguments are re-validated; if they still don't satisfy the schema
  (a grammar enforces structure, not keywords like `minimum`/`pattern`), toolrails
  falls back to the model's own call instead of presenting an invalid one as valid.
- A request body that is valid JSON but not an object returns 400 instead of 500.
- Nullable/union argument types (`["integer", "null"]`) are now coerced.
- `jsonschema` evaluation errors (e.g. an unresolvable `$ref`) no longer abort the
  whole repair.
- The streamed fail-open path surfaces an upstream error as a well-formed SSE
  event instead of a raw blob; streamed tool-call arguments are always a string.
- Parallel calls to the same tool get distinct ids.

### Changed

- Docs describe the repair honestly (coerce → regenerate → fall open) and the
  benchmark reports a realistic range. `demo/reliability.py` now defaults to a
  supported model (`llama3.2:3b`), not the tool-less `gemma3`.

## [0.1.0] — 2026-07-05

First release.

### Added

- An OpenAI-compatible proxy over Ollama that guarantees well-formed tool calls:
  a real tool name and arguments that match the tool's JSON schema.
- A repair ladder that never constrains the model's decision to call a tool
  (which suppresses tool calls — the "constraint tax"): a valid call passes
  through untouched; type errors (a stringified array, a quoted integer) are
  fixed by coercing the model's own values with no second model call; and only
  if coercion can't satisfy the schema are the arguments regenerated under a
  grammar built from it (Ollama's `format`).
- `tool_choice` support, which Ollama's OpenAI endpoint ignores — `none` strips
  the tools, `required` and a named function force a call.
- Hallucinated tool names snapped to the nearest offered tool; unknown names left
  untouched.
- Fail-open behaviour: any error, or an upstream rejection, is passed through
  unchanged rather than turned into a proxy error.
- Streaming support for tool-calling requests: the repaired response is
  re-emitted as standard incremental deltas (each tool call carrying its
  `index`), verified against the OpenAI SDK's streaming client.
- `demo/reliability.py`, a benchmark that measures valid-tool-call rate raw
  versus through toolrails.
- Per-call logging, silenced with `--quiet`.

[0.1.1]: https://github.com/theadamdanielsson/toolrails/releases/tag/v0.1.1
[0.1.0]: https://github.com/theadamdanielsson/toolrails/releases/tag/v0.1.0
