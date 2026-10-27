from pathlib import Path

from hamcrest import assert_that, contains_string, raises

from ml_agent.fitting import Fitting
from ml_agent.tool import ArgumentsError
from ml_agent.workspace import Folder
from tests.fakes import Board, phrase
from tests.titanic import name


def test_reports_the_metrics_of_every_model(tmp_path: Path) -> None:
    model = name()
    (tmp_path / "models.json").write_text("{}", encoding="utf-8")
    assert_that(
        Fitting(Folder(tmp_path), Board({model: {"test": {phrase(): 0.5}}})).run(
            {"config": "models.json"},
        ),
        contains_string(model),
        "Fitting did not report the metrics of the model",
    )


def test_reports_the_value_of_each_metric(tmp_path: Path) -> None:
    value = 1 / (2 + len(phrase()))
    (tmp_path / "models.json").write_text("{}", encoding="utf-8")
    assert_that(
        Fitting(Folder(tmp_path), Board({name(): {"train": {"mae": value}}})).run(
            {"config": "models.json"},
        ),
        contains_string(f"{value:.4f}"),
        "Fitting did not report the value of the metric",
    )


def test_refuses_arguments_without_a_config(tmp_path: Path) -> None:
    assert_that(
        lambda: Fitting(Folder(tmp_path), Board({})).run({phrase(): phrase()}),
        raises(ArgumentsError),
        "Fitting did not refuse the arguments without a config",
    )


def test_manual_describes_the_config_argument(tmp_path: Path) -> None:
    assert_that(
        Fitting(Folder(tmp_path), Board({})).manual(),
        contains_string('"config"'),
        "Fitting manual did not describe the config argument",
    )
