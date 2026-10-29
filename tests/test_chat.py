import json

from hamcrest import assert_that, equal_to

from ml_agent.chat import OpenAiChat
from tests.fakes import completion, phrase
from tests.titanic import name


def test_returns_the_content_of_the_first_choice() -> None:
    content = phrase()
    with completion(lambda _, __: content) as url:
        assert_that(
            OpenAiChat(url, name(), phrase()).reply(phrase()),
            equal_to(content),
            "Chat did not return the content of the first choice",
        )


def test_sends_the_text_as_the_message_of_the_user() -> None:
    text = phrase()
    with completion(lambda _, body: json.dumps(body["messages"], ensure_ascii=False)) as url:
        assert_that(
            OpenAiChat(url, name(), phrase()).reply(text),
            equal_to(json.dumps([{"role": "user", "content": text}], ensure_ascii=False)),
            "Chat did not send the text as the message of the user",
        )


def test_asks_the_named_model() -> None:
    model = phrase()
    with completion(lambda _, body: str(body["model"])) as url:
        assert_that(
            OpenAiChat(url, name(), model).reply(phrase()),
            equal_to(model),
            "Chat did not ask the named model",
        )


def test_authorizes_with_the_key() -> None:
    key = name()
    with completion(lambda headers, _: headers["Authorization"]) as url:
        assert_that(
            OpenAiChat(url, key, phrase()).reply(phrase()),
            equal_to(f"Bearer {key}"),
            "Chat did not authorize with the key",
        )
