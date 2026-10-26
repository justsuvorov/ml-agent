from hamcrest import assert_that, contains_string, raises, string_contains_in_order

from ml_agent.tools import Tools, UnknownToolError
from tests.fakes import Echo, phrase


def test_manual_mentions_every_tool() -> None:
    first, second = phrase(), phrase()
    assert_that(
        Tools({first: Echo(phrase()), second: Echo(phrase())}).manual(),
        string_contains_in_order(first, second),
        "Manual did not mention every tool",
    )


def test_manual_includes_the_manual_of_each_tool() -> None:
    manual = phrase()
    assert_that(
        Tools({phrase(): Echo(manual)}).manual(),
        contains_string(manual),
        "Manual did not include the manual of the tool",
    )


def test_runs_the_named_tool_with_the_given_arguments() -> None:
    value = phrase()
    assert_that(
        Tools({"echo": Echo(phrase())}).run("echo", {"value": value}),
        contains_string(value),
        "Tools did not pass the arguments to the named tool",
    )


def test_refuses_to_run_an_unknown_tool() -> None:
    assert_that(
        lambda: Tools({"echo": Echo(phrase())}).run(phrase(), {}),
        raises(UnknownToolError),
        "Tools did not refuse the unknown tool",
    )
