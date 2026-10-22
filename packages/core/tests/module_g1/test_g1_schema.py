"""JSON schema -> Gemini schema subset (to_gemini_schema): kept, dropped and rejected keywords."""

from __future__ import annotations

from typing import Annotated, Any

import pytest
from pydantic import BaseModel, ConfigDict, Field

from ragforge_core.adapters.llm.chat_gemini import (
    UnsupportedSchema,
    _function_declaration,
    to_gemini_schema,
)


def test_types_are_mapped_and_unsupported_keywords_dropped() -> None:
    schema: dict[str, Any] = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Args",
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "n": {"type": "integer", "minimum": 1, "maximum": 9, "default": 3, "title": "N"},
            "x": {"type": "number", "exclusiveMinimum": 0},
            "ok": {"type": "boolean", "description": "flag"},
        },
        "required": ["n", "ghost"],
    }
    assert to_gemini_schema(schema) == {
        "type": "OBJECT",
        "properties": {
            "n": {"type": "INTEGER", "minimum": 1, "maximum": 9},
            "x": {"type": "NUMBER"},
            "ok": {"type": "BOOLEAN", "description": "flag"},
        },
        "propertyOrdering": ["n", "x", "ok"],
        "required": ["n"],  # a required name with no property would be rejected upstream
    }


def test_formats_kept_only_where_gemini_supports_them() -> None:
    assert to_gemini_schema({"type": "string", "format": "uuid"}) == {"type": "STRING"}
    assert to_gemini_schema({"type": "string", "format": "date-time"}) == {
        "type": "STRING",
        "format": "date-time",
    }
    assert to_gemini_schema({"type": "integer", "format": "int64"})["format"] == "int64"


def test_enums_are_string_only_and_const_becomes_an_enum() -> None:
    assert to_gemini_schema({"enum": ["a", "b"]}) == {"type": "STRING", "enum": ["a", "b"]}
    assert to_gemini_schema({"const": "only"}) == {"type": "STRING", "enum": ["only"]}
    assert to_gemini_schema({"type": "integer", "enum": [1, 2]}) == {"type": "INTEGER"}


def test_optional_from_pydantic_becomes_nullable() -> None:
    class Args(BaseModel):
        model_config = ConfigDict(extra="forbid")
        label: Annotated[str, Field(max_length=80)] | None = None
        quote: Annotated[str, Field(min_length=1)]

    converted = to_gemini_schema(Args.model_json_schema())
    assert converted["properties"]["label"] == {"type": "STRING", "maxLength": 80, "nullable": True}
    assert converted["required"] == ["quote"]


def test_type_list_with_null_is_nullable() -> None:
    assert to_gemini_schema({"type": ["string", "null"]}) == {"type": "STRING", "nullable": True}


def test_any_of_several_concrete_branches_is_kept() -> None:
    converted = to_gemini_schema({"anyOf": [{"type": "string"}, {"type": "integer"}]})
    assert converted == {"anyOf": [{"type": "STRING"}, {"type": "INTEGER"}]}


def test_local_refs_are_inlined() -> None:
    class Inner(BaseModel):
        name: str

    class Outer(BaseModel):
        inner: Inner
        many: list[Inner]

    converted = to_gemini_schema(Outer.model_json_schema())
    inner = {"type": "OBJECT", "properties": {"name": {"type": "STRING"}}}
    inner |= {"propertyOrdering": ["name"], "required": ["name"]}
    assert converted["properties"]["inner"] == inner
    assert converted["properties"]["many"] == {"type": "ARRAY", "items": inner}


def test_type_is_inferred_from_shape() -> None:
    assert to_gemini_schema({"items": {"type": "string"}})["type"] == "ARRAY"
    assert to_gemini_schema({"properties": {"a": {"type": "string"}}})["type"] == "OBJECT"


@pytest.mark.parametrize(
    "schema",
    [
        {"oneOf": [{"type": "string"}]},
        {"allOf": [{"type": "string"}]},
        {"type": "string", "not": {"const": "x"}},
        {"type": ["string", "integer"]},
        {"type": "array", "items": [{"type": "string"}]},
        {"description": "anything at all"},
        {"type": "tuple"},
        {"type": 5},
        {"$ref": "#/$defs/missing"},
        {"$defs": {"N": {"$ref": "#/$defs/N"}}, "$ref": "#/$defs/N"},
        {"anyOf": [{"type": "null"}]},
        {"anyOf": []},
        {"type": "object", "properties": {"x": "not-a-schema"}},
    ],
)
def test_inexpressible_schemas_are_rejected(schema: dict[str, Any]) -> None:
    with pytest.raises(UnsupportedSchema):
        to_gemini_schema(schema)


def test_every_agent_tool_schema_converts() -> None:
    """The real tool registry (module D1) must be expressible, or Gemini would 400 every turn."""
    tools = pytest.importorskip("ragforge_core.agent.tools")
    for tool in tools.REGISTRY.values():
        declaration = _function_declaration(tool.spec)  # argumentless tools omit `parameters`
        assert declaration.get("parameters", {"type": "OBJECT"})["type"] == "OBJECT"


def test_grounding_schema_converts() -> None:
    grounding = pytest.importorskip("ragforge_core.agent.grounding")
    converted = to_gemini_schema(grounding.GROUND_SCHEMA)
    assert converted["properties"]["verdict"]["enum"] == [
        "supported",
        "partially_supported",
        "unsupported",
    ]
    assert converted["properties"]["unsupported_claims"]["items"] == {
        "type": "STRING",
        "maxLength": 2000,
    }
