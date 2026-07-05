"""Pipeline tests against a fake upstream — deterministic, no Ollama needed.

These pin the decisions the pipeline makes: when it takes the fast path (and
spends no second model call), when it regenerates arguments, when a specific
tool_choice overrides the model, and when it forces or strips a call. The fake
counts constrained calls so "fast path costs nothing" is an assertion, not a
claim.
"""

import copy
import json

from toolrails.pipeline import handle

WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}
EVENT = {
    "type": "function",
    "function": {
        "name": "create_event",
        "description": "Create an event.",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "duration_minutes": {"type": "integer"},
            },
            "required": ["title", "duration_minutes"],
        },
    },
}


class FakeUpstream:
    """Stands in for Ollama. Returns canned responses and counts calls."""

    def __init__(self, openai_response, constrained_result=None):
        self._resp = openai_response
        self._constrained = constrained_result
        self.constrained_calls = 0
        self.received_bodies = []

    async def chat_openai(self, body):
        self.received_bodies.append(copy.deepcopy(body))
        return copy.deepcopy(self._resp)

    async def constrained_object(self, model, messages, schema, options=None):
        self.constrained_calls += 1
        return copy.deepcopy(self._constrained)


def with_call(name, arguments, content=None):
    return {"choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
        "role": "assistant", "content": content,
        "tool_calls": [{"id": "c1", "type": "function",
                        "function": {"name": name, "arguments": arguments}}]}}]}


def prose(text):
    return {"choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": text}}]}


def only_call(out):
    return out["choices"][0]["message"]["tool_calls"][0]["function"]


def body(**kw):
    return {"model": "m", "messages": [{"role": "user", "content": "hi"}], **kw}


async def test_fast_path_spends_no_second_call():
    up = FakeUpstream(with_call("get_weather", '{"city": "Oslo"}'))
    out = await handle(body(tools=[WEATHER]), up)
    fn = only_call(out)
    assert fn["name"] == "get_weather"
    assert json.loads(fn["arguments"]) == {"city": "Oslo"}
    assert up.constrained_calls == 0  # already valid — no grammar pass


async def test_type_errors_are_coerced_without_a_model_call():
    # The common small-model failure: right values, wrong types. Coercion fixes
    # it in place — no second generation, and the values are preserved.
    up = FakeUpstream(with_call("create_event",
                                '{"title": "Q3", "duration_minutes": "45"}'))
    out = await handle(body(tools=[EVENT]), up)
    assert up.constrained_calls == 0
    args = json.loads(only_call(out)["arguments"])
    assert args == {"title": "Q3", "duration_minutes": 45}


async def test_uncoercible_arguments_are_regenerated():
    # A missing required field can't be coerced into existence -> grammar regen.
    up = FakeUpstream(with_call("get_weather", '{"units": "c"}'),
                      constrained_result={"city": "Oslo"})
    out = await handle(body(tools=[WEATHER]), up)
    assert up.constrained_calls == 1
    assert json.loads(only_call(out)["arguments"]) == {"city": "Oslo"}


async def test_stringified_arguments_are_repaired_without_a_model_call():
    # OpenAI-shaped arguments arrive as a JSON *string*; a valid one needs no regen.
    up = FakeUpstream(with_call("get_weather", '{"city": "Oslo"}'))
    out = await handle(body(tools=[WEATHER]), up)
    assert up.constrained_calls == 0
    assert json.loads(only_call(out)["arguments"]) == {"city": "Oslo"}


async def test_hallucinated_name_is_snapped():
    up = FakeUpstream(with_call("getWeather", '{"city": "Oslo"}'))
    out = await handle(body(tools=[WEATHER]), up)
    assert only_call(out)["name"] == "get_weather"
    assert up.constrained_calls == 0


async def test_unknown_name_left_untouched():
    # Nothing close to a real tool -> fail open, don't invent a call.
    up = FakeUpstream(with_call("send_email", '{"to": "x"}'))
    out = await handle(body(tools=[WEATHER]), up)
    assert only_call(out)["name"] == "send_email"
    assert up.constrained_calls == 0


async def test_specific_tool_choice_overrides_the_model():
    # The model picked create_event; tool_choice demanded get_weather.
    up = FakeUpstream(with_call("create_event", '{"title": "x", "duration_minutes": 5}'),
                      constrained_result={"city": "Oslo"})
    out = await handle(
        body(tools=[WEATHER, EVENT],
             tool_choice={"type": "function", "function": {"name": "get_weather"}}),
        up,
    )
    assert only_call(out)["name"] == "get_weather"
    assert up.constrained_calls == 1


async def test_tool_choice_none_strips_tools():
    up = FakeUpstream(prose("just words"))
    out = await handle(body(tools=[WEATHER], tool_choice="none"), up)
    assert "tools" not in up.received_bodies[0]
    assert "tool_choice" not in up.received_bodies[0]
    assert out["choices"][0]["message"].get("tool_calls") is None


async def test_required_forces_a_call_from_prose():
    up = FakeUpstream(prose("hello there"), constrained_result={"city": "Oslo"})
    out = await handle(body(tools=[WEATHER], tool_choice="required"), up)
    assert only_call(out)["name"] == "get_weather"
    assert up.constrained_calls == 1  # single tool -> no selection call, one args call


async def test_auto_prose_answer_is_left_alone():
    up = FakeUpstream(prose("here is a plain answer"))
    out = await handle(body(tools=[WEATHER]), up)
    assert out["choices"][0]["message"]["content"] == "here is a plain answer"
    assert up.constrained_calls == 0


async def test_no_tools_is_a_plain_passthrough():
    up = FakeUpstream(prose("hi"))
    out = await handle(body(), up)
    assert out["choices"][0]["message"]["content"] == "hi"
    assert up.constrained_calls == 0
