from typing import Protocol


class Agent(Protocol):
    """Performer of a user task written in Russian."""

    def answer(self, prompt: str) -> str: ...
