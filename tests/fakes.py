import json
import secrets
from collections.abc import Mapping
from pathlib import Path
from typing import Final, final, override

from pydantic import JsonValue

from ml_agent.llm import Llm
from ml_agent.outbox import Outbox
from ml_agent.tool import Tool


def phrase() -> str:
    return " ".join(
        "".join(
            secrets.choice("абвгдеёжзийклмнопрстуфхцчшщъыьэюяQWZxyz0123456789_-")
            for _ in range(1 + secrets.randbelow(9))
        )
        for _ in range(1 + secrets.randbelow(5))
    )


@final
class Script(Llm):
    """Language model fake that gives prepared replies one by one."""

    __slots__ = ("_replies",)

    def __init__(self, *replies: str) -> None:
        self._replies: Final = iter(replies)

    @override
    def reply(self, text: str) -> str:
        return next(self._replies)


@final
class Parrot(Llm):
    """Language model fake that makes one prepared reply and then answers with all it was told."""

    __slots__ = ("_replies",)

    def __init__(self, reply: str) -> None:
        self._replies: Final = iter((reply,))

    @override
    def reply(self, text: str) -> str:
        return next(self._replies, json.dumps({"answer": text}, ensure_ascii=False))


@final
class Echo(Tool):
    """Tool fake that returns the arguments it was called with."""

    __slots__ = ("_manual",)

    def __init__(self, manual: str) -> None:
        self._manual: Final = manual

    @override
    def manual(self) -> str:
        return self._manual

    @override
    def run(self, arguments: Mapping[str, JsonValue]) -> str:
        return json.dumps(dict(arguments), ensure_ascii=False)


@final
class Board(Outbox):
    """Library stub that reports prepared metrics for any config."""

    __slots__ = ("_metrics",)

    def __init__(self, metrics: Mapping[str, Mapping[str, Mapping[str, float]]]) -> None:
        self._metrics: Final = metrics

    @override
    def metrics(self, config: Path) -> Mapping[str, Mapping[str, Mapping[str, float]]]:
        return self._metrics
