from typing import Final, final, override

from ml_agent.agent import Agent
from ml_agent.dialog import Dialog, OpenDialog
from ml_agent.llm import Llm
from ml_agent.reply import Reply
from ml_agent.toolbox import Toolbox


@final
class Loop(Agent):
    """Agent that asks the language model for one step at a time until it gives the answer."""

    __slots__ = ("_limit", "_llm", "_toolbox")

    def __init__(self, llm: Llm, toolbox: Toolbox, limit: int) -> None:
        self._llm: Final = llm
        self._toolbox: Final = toolbox
        self._limit: Final = limit

    @override
    def answer(self, prompt: str) -> str:
        dialog: Dialog = OpenDialog(prompt, self._toolbox.manual())
        for _ in range(self._limit):
            dialog = Reply(self._llm.reply(dialog.text()), self._toolbox).dialog(dialog)
            if dialog.finished():
                return dialog.answer()
        raise StepLimitError(self._limit)


@final
class StepLimitError(Exception):
    """Language model that did not give the answer within the allowed number of steps."""

    def __init__(self, limit: int) -> None:
        super().__init__(f"Language model gave no answer within {limit} steps")
