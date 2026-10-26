from abc import ABC, abstractmethod
from typing import Final, Protocol, final, override

from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter, ValidationError

from ml_agent.dialog import Dialog
from ml_agent.tool import ToolError
from ml_agent.toolbox import Toolbox


class Step(Protocol):
    """Reply of the language model that moves the dialog forward."""

    def dialog(self, dialog: Dialog) -> Dialog: ...


class Move(ABC):
    """Parsed intention of the language model at one step of the dialog."""

    @abstractmethod
    def dialog(self, dialog: Dialog, reply: str, toolbox: Toolbox) -> Dialog: ...


@final
class Answer(BaseModel, Move):
    """Final answer of the language model to the user."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    thought: str = ""
    answer: str

    @override
    def dialog(self, dialog: Dialog, reply: str, toolbox: Toolbox) -> Dialog:
        return dialog.closed(self.answer)


@final
class Call(BaseModel, Move):
    """Request of the language model to run one of the tools."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    thought: str = ""
    tool: str
    arguments: dict[str, JsonValue] = Field(default_factory=dict)

    @override
    def dialog(self, dialog: Dialog, reply: str, toolbox: Toolbox) -> Dialog:
        try:
            return dialog.with_turn(reply, toolbox.run(self.tool, self.arguments))
        except ToolError as error:
            return dialog.with_turn(reply, f"Ошибка инструмента {self.tool}: {error}")


@final
class Reply(Step):
    """Raw text of the language model that should hold one JSON step."""

    __slots__ = ("_text", "_toolbox")

    def __init__(self, text: str, toolbox: Toolbox) -> None:
        self._text: Final = text
        self._toolbox: Final = toolbox

    @override
    def dialog(self, dialog: Dialog) -> Dialog:
        try:
            move = TypeAdapter[Answer | Call](Answer | Call).validate_json(
                self._text[self._text.find("{") : self._text.rfind("}") + 1],
            )
        except ValidationError as error:
            return dialog.with_turn(
                self._text,
                f"Ответ не разобран как JSON-объект шага, повтори его в нужном формате: {error}",
            )
        return move.dialog(dialog, self._text, self._toolbox)
