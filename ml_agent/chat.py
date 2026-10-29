from typing import Final, final, override

import requests
from pydantic import BaseModel, ConfigDict

from ml_agent.llm import Llm


@final
class Message(BaseModel):
    """Message of the assistant in a reply of a chat completions endpoint."""

    model_config = ConfigDict(frozen=True)
    content: str


@final
class Choice(BaseModel):
    """One of the alternatives in a reply of a chat completions endpoint."""

    model_config = ConfigDict(frozen=True)
    message: Message


@final
class Completion(BaseModel):
    """Reply of a chat completions endpoint as far as the agent reads it."""

    model_config = ConfigDict(frozen=True)
    choices: list[Choice]


@final
class OpenAiChat(Llm):
    """Language model served behind an OpenAI-compatible chat completions endpoint."""

    __slots__ = ("_key", "_model", "_timeout", "_url")

    def __init__(self, url: str, key: str, model: str, timeout: float = 600) -> None:
        self._url: Final = url
        self._key: Final = key
        self._model: Final = model
        self._timeout: Final = timeout

    @override
    def reply(self, text: str) -> str:
        response = requests.post(
            f"{self._url.rstrip('/')}/chat/completions",
            json={
                "model": self._model,
                "messages": [{"role": "user", "content": text}],
                "temperature": 0,
            },
            headers={"Authorization": f"Bearer {self._key}"},
            timeout=self._timeout,
        )
        response.raise_for_status()
        return Completion.model_validate_json(response.content).choices[0].message.content
