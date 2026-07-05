# Changelog

All notable changes to toolrails are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to
follow [semantic versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — unreleased

First release.

### Added

- An OpenAI-compatible proxy over Ollama that guarantees well-formed tool calls:
  a real tool name and arguments that match the tool's JSON schema.
- Two-stage repair: the model decides which tool to call unconstrained, then its
  arguments are regenerated under a grammar built from the tool's schema
  (Ollama's `format`), so they cannot come back malformed.
- `tool_choice` support, which Ollama's OpenAI endpoint ignores — `none` strips
  the tools, `required` and a named function force a call.
- Hallucinated tool names snapped to the nearest offered tool; unknown names left
  untouched.
- Fail-open behaviour: any error, or an upstream rejection, is passed through
  unchanged rather than turned into a proxy error.
- Streaming support for tool-calling requests (repaired, then re-emitted).
- `demo/reliability.py`, a benchmark that measures valid-tool-call rate raw
  versus through toolrails.
- Per-call logging, silenced with `--quiet`.

[0.1.0]: https://github.com/theadamdanielsson/toolrails/releases/tag/v0.1.0
