from collections.abc import Mapping
from typing import Protocol, final

from pydantic import JsonValue, ValidationError


class Tool(Protocol):
    """Single process that the agent can run on behalf of the language model."""

    def manual(self) -> str: ...

    def run(self, arguments: Mapping[str, JsonValue]) -> str: ...


class ToolError(Exception):
    """Failure of a tool that the language model can fix by changing its call."""


@final
class ArgumentsError(ToolError):
    """Arguments of a tool call that do not match the schema of the tool."""

    def __init__(self, error: ValidationError) -> None:
        super().__init__(f"Arguments do not match the schema of the tool: {error}")
