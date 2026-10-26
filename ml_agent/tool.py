from collections.abc import Mapping
from typing import Protocol

from pydantic import JsonValue


class Tool(Protocol):
    """Single process that the agent can run on behalf of the language model."""

    def manual(self) -> str: ...

    def run(self, arguments: Mapping[str, JsonValue]) -> str: ...


class ToolError(Exception):
    """Failure of a tool that the language model can fix by changing its call."""
