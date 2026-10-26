import json

from hamcrest import assert_that, contains_string, equal_to, raises

from ml_agent.loop import Loop, StepLimitError
from ml_agent.tools import Tools
from tests.fakes import Echo, Parrot, Script, phrase


def test_returns_the_answer_the_model_gives_at_once() -> None:
    answer = phrase()
    assert_that(
        Loop(
            Script(json.dumps({"thought": phrase(), "answer": answer}, ensure_ascii=False)),
            Tools({}),
            3,
        ).answer(phrase()),
        equal_to(answer),
        "Loop did not return the answer of the model",
    )


def test_passes_the_tool_output_to_the_model() -> None:
    value = phrase()
    assert_that(
        Loop(
            Parrot(
                json.dumps(
                    {"thought": phrase(), "tool": "echo", "arguments": {"value": value}},
                    ensure_ascii=False,
                ),
            ),
            Tools({"echo": Echo(phrase())}),
            2,
        ).answer(phrase()),
        contains_string(value),
        "Loop did not show the tool output to the model",
    )


def test_reports_an_unknown_tool_to_the_model() -> None:
    assert_that(
        Loop(
            Parrot(json.dumps({"thought": phrase(), "tool": "missing", "arguments": {}})),
            Tools({"echo": Echo(phrase())}),
            2,
        ).answer(phrase()),
        contains_string("missing"),
        "Loop did not tell the model that the tool is unknown",
    )


def test_reports_a_malformed_reply_to_the_model() -> None:
    assert_that(
        Loop(Parrot(phrase()), Tools({}), 2).answer(phrase()),
        contains_string("JSON"),
        "Loop did not tell the model that its reply is malformed",
    )


def test_fails_when_the_model_never_answers() -> None:
    assert_that(
        lambda: Loop(
            Script(*[json.dumps({"tool": "echo", "arguments": {}})] * 4),
            Tools({"echo": Echo(phrase())}),
            3,
        ).answer(phrase()),
        raises(StepLimitError),
        "Loop did not stop the model that never answers",
    )
