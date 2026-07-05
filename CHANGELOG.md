# Changelog

All notable changes to toolrails are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to
follow [semantic versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — unreleased

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

[0.1.0]: https://github.com/theadamdanielsson/toolrails/releases/tag/v0.1.0
