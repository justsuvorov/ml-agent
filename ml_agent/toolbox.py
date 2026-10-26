from collections.abc import Mapping
from typing import Protocol

from pydantic import JsonValue


class Toolbox(Protocol):
    """Set of tools that the language model calls by name."""

    def manual(self) -> str: ...

    def run(self, name: str, arguments: Mapping[str, JsonValue]) -> str: ...
