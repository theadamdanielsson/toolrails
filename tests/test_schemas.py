"""Tests for the pure repair logic — no Ollama, no network."""

from toolrails import schemas

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Look up the weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "units": {"type": "string", "enum": ["c", "f"]},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {"name": "list_files", "description": "List a directory."},
    },
]


def test_tool_names():
    assert schemas.tool_names(TOOLS) == ["get_weather", "list_files"]


def test_schema_for_known_and_paramless():
    assert schemas.schema_for(TOOLS, "get_weather")["required"] == ["city"]
    # A tool with no declared params still gets a usable object schema.
    assert schemas.schema_for(TOOLS, "list_files") == {"type": "object", "properties": {}}


def test_nearest_name_snaps_close_misspelling():
    names = schemas.tool_names(TOOLS)
    assert schemas.nearest_name("getWeather", names) == "get_weather"
    assert schemas.nearest_name("get_weather", names) == "get_weather"
    # Nothing close: refuse to guess.
    assert schemas.nearest_name("send_email", names) is None


def test_parse_arguments_strict_and_dict():
    assert schemas.parse_arguments('{"city": "Oslo"}') == {"city": "Oslo"}
    assert schemas.parse_arguments({"city": "Oslo"}) == {"city": "Oslo"}
    assert schemas.parse_arguments("") == {}


def test_parse_arguments_repairs_fences_and_prose():
    fenced = '```json\n{"city": "Oslo"}\n```'
    assert schemas.parse_arguments(fenced) == {"city": "Oslo"}
    chatty = 'Sure! Here you go: {"city": "Oslo", "units": "c"} — hope that helps'
    assert schemas.parse_arguments(chatty) == {"city": "Oslo", "units": "c"}


def test_parse_arguments_repairs_trailing_comma():
    assert schemas.parse_arguments('{"city": "Oslo",}') == {"city": "Oslo"}


def test_parse_arguments_gives_up_cleanly():
    assert schemas.parse_arguments("not json at all") is None


def test_args_valid_against_schema():
    schema = schemas.schema_for(TOOLS, "get_weather")
    assert schemas.args_valid({"city": "Oslo"}, schema) is True
    assert schemas.args_valid({"city": "Oslo", "units": "c"}, schema) is True
    # Missing required field.
    assert schemas.args_valid({"units": "c"}, schema) is False
    # Bad enum value.
    assert schemas.args_valid({"city": "Oslo", "units": "kelvin"}, schema) is False
