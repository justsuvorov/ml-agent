from pathlib import Path

from hamcrest import assert_that, contains_string, equal_to, raises

from ml_agent.checking import FeatureCheck
from ml_agent.tool import ArgumentsError
from ml_agent.workspace import Folder
from tests.fakes import Shelf, phrase
from tests.titanic import name


def test_describes_the_features_already_in_the_model(tmp_path: Path) -> None:
    feature, description = name(), phrase()
    (tmp_path / "models.json").write_text("{}", encoding="utf-8")
    (tmp_path / "automl.json").write_text("{}", encoding="utf-8")
    assert_that(
        FeatureCheck(Folder(tmp_path), Shelf({feature: description}, frozenset())).run(
            {
                "models_config": "models.json",
                "automl_config": "automl.json",
                "model": name(),
                "features": [feature],
            },
        ),
        contains_string(description),
        "Check did not describe the feature already in the model",
    )


def test_skips_the_selection_when_every_feature_is_in_the_model(tmp_path: Path) -> None:
    feature = name()
    (tmp_path / "models.json").write_text("{}", encoding="utf-8")
    (tmp_path / "automl.json").write_text("{}", encoding="utf-8")
    assert_that(
        FeatureCheck(Folder(tmp_path), Shelf({feature: phrase()}, frozenset({feature}))).run(
            {
                "models_config": "models.json",
                "automl_config": "automl.json",
                "model": name(),
                "features": [feature],
            },
        ).count("<selected"),
        equal_to(0),
        "Check ran the selection although every feature is in the model",
    )


def test_names_the_features_missing_in_the_dataset(tmp_path: Path) -> None:
    feature = name()
    (tmp_path / "models.json").write_text("{}", encoding="utf-8")
    (tmp_path / "automl.json").write_text("{}", encoding="utf-8")
    assert_that(
        FeatureCheck(Folder(tmp_path), Shelf({}, frozenset({name()}))).run(
            {
                "models_config": "models.json",
                "automl_config": "automl.json",
                "model": name(),
                "features": [feature],
            },
        ),
        contains_string(feature),
        "Check did not name the feature missing in the dataset",
    )


def test_selects_from_the_new_features_found_in_the_dataset(tmp_path: Path) -> None:
    feature = name()
    (tmp_path / "models.json").write_text("{}", encoding="utf-8")
    (tmp_path / "automl.json").write_text("{}", encoding="utf-8")
    assert_that(
        FeatureCheck(Folder(tmp_path), Shelf({name(): phrase()}, frozenset({feature}))).run(
            {
                "models_config": "models.json",
                "automl_config": "automl.json",
                "model": name(),
                "features": [feature],
            },
        ),
        contains_string(f"<selected {feature}>"),
        "Check did not select from the new feature found in the dataset",
    )


def test_keeps_the_features_of_the_model_out_of_the_selection(tmp_path: Path) -> None:
    known = name()
    (tmp_path / "models.json").write_text("{}", encoding="utf-8")
    (tmp_path / "automl.json").write_text("{}", encoding="utf-8")
    assert_that(
        FeatureCheck(Folder(tmp_path), Shelf({known: phrase()}, frozenset({known, "new"}))).run(
            {
                "models_config": "models.json",
                "automl_config": "automl.json",
                "model": name(),
                "features": [known, "new"],
            },
        ).count(f"<selected {known}>"),
        equal_to(0),
        "Check passed the feature already in the model to the selection",
    )


def test_refuses_a_call_without_features(tmp_path: Path) -> None:
    assert_that(
        lambda: FeatureCheck(Folder(tmp_path), Shelf({}, frozenset())).run(
            {"models_config": phrase(), "automl_config": phrase(), "model": name(), "features": []},
        ),
        raises(ArgumentsError),
        "Check did not refuse the call without features",
    )


def test_saves_the_config_copy_next_to_the_config(tmp_path: Path) -> None:
    model = name()
    (tmp_path / "models.json").write_text("{}", encoding="utf-8")
    (tmp_path / "automl.json").write_text("{}", encoding="utf-8")
    assert_that(
        FeatureCheck(Folder(tmp_path), Shelf({}, frozenset({"new"}))).run(
            {
                "models_config": "models.json",
                "automl_config": "automl.json",
                "model": model,
                "features": ["new"],
            },
        ),
        contains_string(f"<target {(tmp_path / f'models_{model}_features.json').resolve()}>"),
        "Check did not save the config copy next to the config",
    )
