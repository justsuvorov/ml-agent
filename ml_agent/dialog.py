from typing import Final, Protocol, final, override


class Dialog(Protocol):
    """Conversation between the agent and the language model about one user task."""

    def text(self) -> str: ...

    def finished(self) -> bool: ...

    def answer(self) -> str: ...

    def with_turn(self, reply: str, observation: str) -> "Dialog": ...

    def closed(self, answer: str) -> "Dialog": ...


@final
class OpenDialog(Dialog):
    """Conversation that still waits for the final answer of the language model."""

    __slots__ = ("_manual", "_prompt", "_turns")

    def __init__(self, prompt: str, manual: str, turns: tuple[tuple[str, str], ...] = ()) -> None:
        self._prompt: Final = prompt
        self._manual: Final = manual
        self._turns: Final = turns

    @override
    def text(self) -> str:
        return "\n\n".join(
            (
                (
                    "Ты — ML-агент. Ты выполняешь задания пользователя по моделям машинного "
                    "обучения с помощью инструментов библиотеки outboxml."
                ),
                (
                    "На каждом шаге отвечай ровно одним JSON-объектом без текста вокруг него.\n"
                    'Вызов инструмента: {"thought": "<рассуждение>", "tool": "<имя>", '
                    '"arguments": {<аргументы>}}\n'
                    'Итоговый ответ: {"thought": "<рассуждение>", "answer": "<ответ пользователю>"}'
                ),
                f"Инструменты:\n{self._manual}",
                f"Задание пользователя:\n{self._prompt}",
                *(
                    f"Шаг {number}.\nТвой ответ:\n{reply}\nРезультат:\n{observation}"
                    for number, (reply, observation) in enumerate(self._turns, start=1)
                ),
            ),
        )

    @override
    def finished(self) -> bool:
        return False

    @override
    def answer(self) -> str:
        raise OpenDialogError

    @override
    def with_turn(self, reply: str, observation: str) -> Dialog:
        return OpenDialog(self._prompt, self._manual, (*self._turns, (reply, observation)))

    @override
    def closed(self, answer: str) -> Dialog:
        return ClosedDialog(self, answer)


@final
class ClosedDialog(Dialog):
    """Conversation in which the language model has given its final answer."""

    __slots__ = ("_answer", "_origin")

    def __init__(self, origin: Dialog, answer: str) -> None:
        self._origin: Final = origin
        self._answer: Final = answer

    @override
    def text(self) -> str:
        return self._origin.text()

    @override
    def finished(self) -> bool:
        return True

    @override
    def answer(self) -> str:
        return self._answer

    @override
    def with_turn(self, reply: str, observation: str) -> Dialog:
        raise ClosedDialogError

    @override
    def closed(self, answer: str) -> Dialog:
        raise ClosedDialogError


@final
class OpenDialogError(Exception):
    """Request for the answer of a dialog that has none yet."""

    def __init__(self) -> None:
        super().__init__("Dialog has no final answer yet")


@final
class ClosedDialogError(Exception):
    """Attempt to continue a dialog that already has its final answer."""

    def __init__(self) -> None:
        super().__init__("Dialog already has its final answer and takes no more turns")
