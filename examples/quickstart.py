"""A minimal end-to-end example: call a tool through toolrails.

Prerequisites:
    - Ollama running with a tool-capable model, e.g. `ollama pull llama3.2:3b`
    - toolrails running:  uvx toolrails --port 11500
    - the OpenAI SDK:      pip install openai

Then:  python examples/quickstart.py

You point the OpenAI client at toolrails instead of Ollama. Everything else is
ordinary tool calling — but the call you get back is guaranteed to parse and
match the schema.
"""

import json

from openai import OpenAI

client = OpenAI(base_url="http://localhost:11500/v1", api_key="ollama")

tools = [
    {
        "type": "function",
        "function": {
            "name": "create_event",
            "description": "Create a calendar event.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "duration_minutes": {"type": "integer"},
                    "attendees": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["title", "duration_minutes"],
            },
        },
    }
]

resp = client.chat.completions.create(
    model="llama3.2:3b",
    messages=[
        {
            "role": "user",
            "content": "Book a 45 minute event 'Q3 planning' with alice@example.com "
            "and bob@example.com.",
        }
    ],
    tools=tools,
)

call = resp.choices[0].message.tool_calls[0]
args = json.loads(call.function.arguments)  # this parse never throws through toolrails

print("tool:", call.function.name)
print("arguments:", json.dumps(args, indent=2))
assert isinstance(args["duration_minutes"], int)      # a real int, not "45"
assert isinstance(args["attendees"], list)            # a real list, not "[...]"
print("\ntypes are correct — the agent can use this call directly.")
