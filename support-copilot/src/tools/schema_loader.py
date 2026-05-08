from __future__ import annotations

from functools import lru_cache
from typing import Any

from src.utils.io import project_path, read_json


class ToolSchemaError(ValueError):
    pass


@lru_cache(maxsize=1)
def load_tool_schemas() -> dict[str, dict]:
    raw = read_json(project_path("schemas", "tool_schema.json"), {"tools": []})
    return {tool["name"]: tool for tool in raw.get("tools", [])}


def validate_tool_arguments(name: str, arguments: dict[str, Any]) -> None:
    schemas = load_tool_schemas()
    if name not in schemas:
        raise ToolSchemaError(f"Unknown tool '{name}'. Available tools: {sorted(schemas)}")
    schema = schemas[name]["arguments_schema"]
    missing = [field for field in schema.get("required", []) if field not in arguments]
    if missing:
        raise ToolSchemaError(f"Tool '{name}' missing required arguments: {missing}")
    properties = schema.get("properties", {})
    for key, value in arguments.items():
        if key not in properties:
            continue
        _validate_type(name, key, value, properties[key])


def _validate_type(tool: str, key: str, value: Any, schema: dict) -> None:
    expected = schema.get("type")
    expected_types = expected if isinstance(expected, list) else [expected]
    if value is None and "null" in expected_types:
        return
    type_ok = False
    for expected_type in expected_types:
        if expected_type == "string" and isinstance(value, str):
            type_ok = True
        elif expected_type == "integer" and isinstance(value, int) and not isinstance(value, bool):
            type_ok = True
        elif expected_type == "number" and isinstance(value, (int, float)) and not isinstance(value, bool):
            type_ok = True
        elif expected_type == "object" and isinstance(value, dict):
            type_ok = True
        elif expected_type == "array" and isinstance(value, list):
            type_ok = True
    if not type_ok:
        raise ToolSchemaError(f"Tool '{tool}' argument '{key}' expected {expected_types}, got {type(value).__name__}")
    if "enum" in schema and value not in schema["enum"]:
        raise ToolSchemaError(f"Tool '{tool}' argument '{key}' must be one of {schema['enum']}, got {value!r}")
