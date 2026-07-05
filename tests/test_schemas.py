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


def test_nearest_name_refuses_opposite_verb():
    # Lexically near, semantically opposite — must NOT snap across.
    assert schemas.nearest_name("create_file", ["delete_file", "list_files"]) is None
    assert schemas.nearest_name("get_weather", ["set_weather"]) is None
    assert schemas.nearest_name("enable_user", ["disable_user"]) is None
    assert schemas.nearest_name("start_server", ["stop_server"]) is None


def test_nearest_name_still_snaps_casing_and_separators():
    assert schemas.nearest_name("getWeather", ["get_weather"]) == "get_weather"
    assert schemas.nearest_name("get-weather", ["get_weather"]) == "get_weather"
    assert schemas.nearest_name("GetWeather", ["get_weather"]) == "get_weather"


def test_coerce_handles_nullable_type_list():
    schema = {"type": "object", "properties": {"n": {"type": ["integer", "null"]}}}
    assert schemas.coerce({"n": "30"}, schema) == {"n": 30}


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


def test_coerce_fixes_quoted_int_and_stringified_array():
    schema = {
        "type": "object",
        "properties": {
            "duration_minutes": {"type": "integer"},
            "attendees": {"type": "array", "items": {"type": "string"}},
        },
    }
    raw = {"duration_minutes": "30", "attendees": '["a@x.com", "b@x.com"]'}
    coerced = schemas.coerce(raw, schema)
    assert coerced == {"duration_minutes": 30, "attendees": ["a@x.com", "b@x.com"]}
    assert schemas.args_valid(coerced, schema)


def test_coerce_recurses_into_stringified_nested_objects():
    schema = {
        "type": "object",
        "properties": {
            "reminders": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "method": {"type": "string"},
                        "minutes_before": {"type": "integer"},
                    },
                },
            }
        },
    }
    raw = {"reminders": '[{"method": "email", "minutes_before": "10"}]'}
    coerced = schemas.coerce(raw, schema)
    assert coerced == {"reminders": [{"method": "email", "minutes_before": 10}]}


def test_coerce_handles_booleans_and_leaves_strings_alone():
    schema = {"type": "object", "properties": {
        "flag": {"type": "boolean"}, "name": {"type": "string"}}}
    assert schemas.coerce({"flag": "true", "name": "Oslo"}, schema) == {"flag": True, "name": "Oslo"}


def test_coerce_never_invents_a_missing_value():
    # A required field the model omitted stays omitted — coercion only reshapes.
    schema = schemas.schema_for(TOOLS, "get_weather")
    assert schemas.coerce({"units": "c"}, schema) == {"units": "c"}
    assert schemas.args_valid(schemas.coerce({"units": "c"}, schema), schema) is False


def test_coerce_leaves_unconvertible_values_for_validation():
    schema = {"type": "object", "properties": {"n": {"type": "integer"}}}
    # "abc" can't become an int; left as-is so validation still fails cleanly.
    assert schemas.coerce({"n": "abc"}, schema) == {"n": "abc"}


def test_args_valid_against_schema():
    schema = schemas.schema_for(TOOLS, "get_weather")
    assert schemas.args_valid({"city": "Oslo"}, schema) is True
    assert schemas.args_valid({"city": "Oslo", "units": "c"}, schema) is True
    # Missing required field.
    assert schemas.args_valid({"units": "c"}, schema) is False
    # Bad enum value.
    assert schemas.args_valid({"city": "Oslo", "units": "kelvin"}, schema) is False
