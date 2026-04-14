"""Tool registry and schema helpers."""

from __future__ import annotations

import json
import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)

# OpenAI-compatible tool definition type
ToolDef = dict[str, Any]  # {"type": "function", "function": {...}}


def _func_tool(
    name: str,
    description: str,
    parameters: dict[str, Any],
    required: list[str] | None = None,
) -> ToolDef:
    """Build an OpenAI-compatible tool definition."""
    props = parameters
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": props,
                "required": required or [],
            },
        },
    }


class ToolRegistry:
    """Maps tool names -> (schema, handler) pairs."""

    def __init__(self) -> None:
        self._tools: dict[str, tuple[ToolDef, Callable[..., Any]]] = {}

    def register(self, schema: ToolDef, handler: Callable[..., Any]) -> None:
        name = schema["function"]["name"]
        self._tools[name] = (schema, handler)

    def get_schemas(self) -> list[ToolDef]:
        return [schema for schema, _ in self._tools.values()]

    def execute(self, name: str, arguments: dict[str, Any]) -> str:
        """Execute a tool by name. Returns JSON string result.

        Strips any unexpected keyword arguments that the LLM may inject
        (e.g. Gemma adding 'id' fields) to prevent TypeErrors.
        """
        if name not in self._tools:
            return json.dumps({"error": f"Unknown tool: {name}"})
        schema, handler = self._tools[name]
        try:
            # Filter arguments to only those declared in the schema
            declared_params = set(
                schema.get("function", {})
                .get("parameters", {})
                .get("properties", {})
                .keys()
            )
            if declared_params:
                filtered = {k: v for k, v in arguments.items() if k in declared_params}
            else:
                filtered = arguments
            result = handler(**filtered)
            if isinstance(result, str):
                return result
            return json.dumps(result, indent=2, default=str)
        except Exception as exc:
            logger.exception("Tool %s failed", name)
            return json.dumps({"error": f"{type(exc).__name__}: {exc}"})

    def list_names(self) -> list[str]:
        return list(self._tools.keys())
