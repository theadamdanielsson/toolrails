"""toolrails — valid tool calls from any local model.

A drop-in OpenAI-compatible proxy in front of Ollama that guarantees the tool
calls your agent receives are well-formed: real tool name, arguments that match
the tool's JSON schema. It also restores `tool_choice`, which Ollama's
OpenAI-compatible endpoint silently ignores.
"""

__version__ = "0.1.1"
