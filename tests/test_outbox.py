import json
from pathlib import Path

from hamcrest import assert_that, contains_string, equal_to, has_item, has_key, raises
from outboxml.core.pydantic_models import FeatureModelConfig

from ml_agent.outbox import FeatureText, LocalOutbox, UnknownModelError
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


def test_describes_every_feature_of_the_model(tmp_path: Path) -> None:
    first = name()
    assert_that(
        LocalOutbox(tmp_path).features(Titanic(tmp_path, (first, name())).config(), first),
        has_key("SEX"),
        "Outbox did not describe every feature of the model",
    )


def test_refuses_to_describe_an_unknown_model(tmp_path: Path) -> None:
    assert_that(
        lambda: LocalOutbox(tmp_path).features(Titanic(tmp_path, (name(), name())).config(), "x"),
        raises(UnknownModelError),
        "Outbox did not refuse the unknown model",
    )


def test_finds_the_columns_of_the_dataset(tmp_path: Path) -> None:
    assert_that(
        sorted(
            LocalOutbox(tmp_path).dataset(Titanic(tmp_path, (name(), name())).config()).columns(),
        ),
        has_item("EMBARKED"),
        "Outbox did not find the column of the dataset",
    )


def test_selection_reports_the_type_of_every_candidate(tmp_path: Path) -> None:
    first = name()
    titanic = Titanic(tmp_path, (first, name()))
    assert_that(
        LocalOutbox(tmp_path)
        .dataset(titanic.config())
        .selection(titanic.automl(), first, ("EMBARKED", "NAME"), tmp_path / "copy.json"),
        contains_string("NAME: тип object"),
        "Selection did not report the type of the candidate",
    )


def test_selection_describes_the_parameters_of_a_selected_feature(tmp_path: Path) -> None:
    first = name()
    titanic = Titanic(tmp_path, (first, name()))
    assert_that(
        LocalOutbox(tmp_path)
        .dataset(titanic.config())
        .selection(titanic.automl(), first, ("EMBARKED",), tmp_path / "copy.json"),
        contains_string("WoE_cat_to_num"),
        "Selection did not describe the parameters of the selected feature",
    )


def test_selection_trims_the_dataset_to_the_used_columns(tmp_path: Path) -> None:
    first = name()
    titanic = Titanic(tmp_path, (first, name()))
    assert_that(
        LocalOutbox(tmp_path)
        .dataset(titanic.config())
        .selection(titanic.automl(), first, ("EMBARKED",), tmp_path / "copy.json"),
        contains_string("9 из 13"),
        "Selection did not trim the dataset to the used columns",
    )


def test_selection_describes_the_bins_of_a_selected_numeric_feature(tmp_path: Path) -> None:
    first = name()
    titanic = Titanic(tmp_path, (first, name()))
    assert_that(
        LocalOutbox(tmp_path)
        .dataset(titanic.config())
        .selection(titanic.automl(), first, ("PCLASS",), tmp_path / "copy.json"),
        contains_string("PCLASS: тип числовая"),
        "Selection did not describe the bins of the selected numeric feature",
    )


def test_feature_text_keeps_infinite_bounds_of_bins() -> None:
    bound = 1 + len(name())
    assert_that(
        FeatureText(
            FeatureModelConfig(
                name=name(),
                replace={"_TYPE_": "_NUM_"},
                bins=[float("-inf"), bound, float("inf")],
            ),
        ).text(),
        contains_string(f"[-Infinity, {bound}, Infinity]"),
        "Feature text did not keep the infinite bounds of the bins",
    )


def test_selection_saves_a_config_copy_with_the_selected_feature(tmp_path: Path) -> None:
    first = name()
    titanic = Titanic(tmp_path, (first, name()))
    LocalOutbox(tmp_path).dataset(titanic.config()).selection(
        titanic.automl(),
        first,
        ("EMBARKED",),
        tmp_path / "copy.json",
    )
    assert_that(
        [
            feature["name"]
            for model in json.loads((tmp_path / "copy.json").read_text(encoding="utf-8"))[
                "models_configs"
            ]
            if model["name"] == first
            for feature in model["features"]
        ],
        has_item("EMBARKED"),
        "Selection did not save the config copy with the selected feature",
    )


def test_selection_keeps_the_data_path_of_the_config_in_its_copy(tmp_path: Path) -> None:
    first = name()
    titanic = Titanic(tmp_path, (first, name()))
    LocalOutbox(tmp_path).dataset(titanic.config()).selection(
        titanic.automl(),
        first,
        ("EMBARKED",),
        tmp_path / "copy.json",
    )
    assert_that(
        json.loads((tmp_path / "copy.json").read_text(encoding="utf-8"))["data_config"][
            "local_name_source"
        ],
        equal_to("data/titanic.csv"),
        "Selection did not keep the data path of the config in its copy",
    )


def test_selection_saves_a_config_copy_that_trains(tmp_path: Path) -> None:
    first = name()
    titanic = Titanic(tmp_path, (first, name()))
    LocalOutbox(tmp_path).dataset(titanic.config()).selection(
        titanic.automl(),
        first,
        ("PCLASS", "EMBARKED"),
        tmp_path / "copy.json",
    )
    assert_that(
        LocalOutbox(tmp_path).metrics(tmp_path / "copy.json"),
        has_key(first),
        "Selection saved a config copy that does not train",
    )


def test_selection_saves_no_copy_when_nothing_is_selected(tmp_path: Path) -> None:
    first = name()
    titanic = Titanic(tmp_path, (first, name()))
    LocalOutbox(tmp_path).dataset(titanic.config()).selection(
        titanic.automl(),
        first,
        ("NAME",),
        tmp_path / "copy.json",
    )
    assert_that(
        [path.name for path in tmp_path.iterdir()].count("copy.json"),
        equal_to(0),
        "Selection saved a config copy although nothing was selected",
    )
