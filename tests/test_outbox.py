from pathlib import Path

from hamcrest import assert_that, has_key

from ml_agent.outbox import LocalOutbox
from tests.titanic import Titanic, name


def test_reports_metrics_of_every_model_in_the_config(tmp_path: Path) -> None:
    second = name()
    assert_that(
        LocalOutbox(tmp_path).metrics(Titanic(tmp_path, (name(), second)).config()),
        has_key(second),
        "Outbox did not report metrics of every model",
    )


def test_reports_metrics_on_the_test_sample(tmp_path: Path) -> None:
    first = name()
    assert_that(
        LocalOutbox(tmp_path).metrics(Titanic(tmp_path, (first, name())).config())[first],
        has_key("test"),
        "Outbox did not report metrics on the test sample",
    )
