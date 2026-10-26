from hamcrest import assert_that, contains_string, equal_to, raises

from ml_agent.dialog import ClosedDialogError, OpenDialog, OpenDialogError
from tests.fakes import phrase


def test_text_includes_the_prompt() -> None:
    prompt = phrase()
    assert_that(
        OpenDialog(prompt, phrase()).text(),
        contains_string(prompt),
        "Dialog did not include the prompt of the user",
    )


def test_text_includes_the_manual() -> None:
    manual = phrase()
    assert_that(
        OpenDialog(phrase(), manual).text(),
        contains_string(manual),
        "Dialog did not include the manual of the tools",
    )


def test_text_includes_the_observation_of_each_turn() -> None:
    observation = phrase()
    assert_that(
        OpenDialog(phrase(), phrase()).with_turn(phrase(), observation).text(),
        contains_string(observation),
        "Dialog did not include the observation of the turn",
    )


def test_closed_dialog_returns_its_answer() -> None:
    answer = phrase()
    assert_that(
        OpenDialog(phrase(), phrase()).closed(answer).answer(),
        equal_to(answer),
        "Closed dialog did not return its answer",
    )


def test_open_dialog_refuses_to_give_an_answer() -> None:
    assert_that(
        OpenDialog(phrase(), phrase()).answer,
        raises(OpenDialogError),
        "Open dialog did not refuse to give an answer",
    )


def test_closed_dialog_refuses_a_new_turn() -> None:
    assert_that(
        lambda: OpenDialog(phrase(), phrase()).closed(phrase()).with_turn(phrase(), phrase()),
        raises(ClosedDialogError),
        "Closed dialog did not refuse a new turn",
    )
