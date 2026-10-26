from typing import Protocol


class Llm(Protocol):
    """External language model service that replies to a text."""

    def reply(self, text: str) -> str: ...
