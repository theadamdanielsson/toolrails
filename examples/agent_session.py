"""A real multi-turn agent loop through toolrails, with streaming.

This is the closest thing to how an actual coding agent uses the proxy: it gives
a local model real tools, streams each response, executes the tool calls the
model makes, feeds the results back, and repeats until the model answers. Every
call is repaired by toolrails before this loop sees it, so the dispatch code
below never defends against malformed arguments — `json.loads` just works.

Run:
    uvx toolrails --port 11500          # in another terminal
    pip install openai
    python examples/agent_session.py
"""

import json
import os
import shutil
import tempfile

from openai import OpenAI

client = OpenAI(base_url="http://localhost:11500/v1", api_key="ollama")
# Any tool-capable Ollama model works; a stronger one completes multi-step tasks
# more reliably. Override with TOOLRAILS_MODEL=qwen3:8b (etc.).
MODEL = os.environ.get("TOOLRAILS_MODEL", "llama3.2:3b")

# --- a tiny real workspace the agent will explore --------------------------
workspace = tempfile.mkdtemp(prefix="toolrails-demo-")
for name, content in {
    "readme.txt": "Toolrails demo project. The secret code is BLUE-42.",
    "notes.txt": "Remember to water the plants. Nothing important here.",
    "config.txt": "port=11500\nmodel=llama3.2:3b\n",
}.items():
    with open(os.path.join(workspace, name), "w") as f:
        f.write(content)


# --- real tools the model can call -----------------------------------------
def list_files() -> str:
    return "\n".join(sorted(os.listdir(workspace)))


def read_file(path: str) -> str:
    full = os.path.join(workspace, os.path.basename(path or ""))
    if not os.path.isfile(full):
        return f"(no such file: {path})"
    with open(full) as f:
        return f.read()


TOOLS = [
    {"type": "function", "function": {
        "name": "list_files", "description": "List the files in the project.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "read_file", "description": "Read one file's contents.",
        "parameters": {"type": "object",
                       "properties": {"path": {"type": "string"}},
                       "required": ["path"]}}},
]
DISPATCH = {
    "list_files": lambda a: list_files(),
    "read_file": lambda a: read_file(a.get("path", "")),
}


def stream_turn(messages):
    """One streamed turn. Accumulate the SDK's chunks into content + calls —
    if toolrails' streamed deltas were malformed, this accumulation would break."""
    content, calls = "", {}
    for chunk in client.chat.completions.create(
        model=MODEL, messages=messages, tools=TOOLS, stream=True
    ):
        delta = chunk.choices[0].delta
        if delta.content:
            content += delta.content
        for tc in delta.tool_calls or []:
            slot = calls.setdefault(tc.index, {"id": None, "name": "", "arguments": ""})
            if tc.id:
                slot["id"] = tc.id
            if tc.function and tc.function.name:
                slot["name"] = tc.function.name
            if tc.function and tc.function.arguments:
                slot["arguments"] += tc.function.arguments
    return content, [calls[i] for i in sorted(calls)]


def main() -> None:
    messages = [
        {"role": "system", "content": "You are an assistant with file tools. "
         "Use them to answer, then state the answer plainly."},
        {"role": "user", "content": "Find the secret code hidden in one of the "
         "project files and tell me what it is."},
    ]

    answer = ""
    try:
        for _ in range(6):
            content, calls = stream_turn(messages)
            if not calls:
                answer = content
                print(f"\nagent: {content.strip()}")
                break
            messages.append({"role": "assistant", "content": content or None,
                             "tool_calls": [
                                 {"id": c["id"] or f"call_{i}", "type": "function",
                                  "function": {"name": c["name"], "arguments": c["arguments"]}}
                                 for i, c in enumerate(calls)]})
            for i, c in enumerate(calls):
                args = json.loads(c["arguments"])  # guaranteed to parse
                print(f"  → {c['name']}({args})")
                result = DISPATCH[c["name"]](args)
                messages.append({"role": "tool",
                                 "tool_call_id": c["id"] or f"call_{i}",
                                 "content": result})
    finally:
        shutil.rmtree(workspace, ignore_errors=True)

    print("\nPASS — agent completed the task through streamed, repaired tool calls."
          if "BLUE-42" in answer else
          "\n(agent finished without surfacing the code — try again or a stronger model)")


if __name__ == "__main__":
    main()
