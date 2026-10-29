import json
from collections.abc import Collection, Mapping
from itertools import chain
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Final, Protocol, final, override

import pandas as pd
from outboxml.core.enums import FeatureEngineering, FeatureTypesForSelection
from outboxml.core.prepared_datasets import FeatureSelectionPrepareDataset, PrepareDataset
from outboxml.core.pydantic_models import (
    AllModelsConfig,
    AutoMLConfig,
    FeatureModelConfig,
    ModelConfig,
)
from outboxml.data_subsets import DataPreprocessor
from outboxml.datasets_manager import DataSetsManager
from outboxml.extractors import BaseExtractor, get_useful_columns
from outboxml.feature_selection import BaseFS, FeatureSelectionInterface
from pydantic_core import to_jsonable_python

from ml_agent.tool import ToolError


class Training(Protocol):
    """Library process that trains the models of a config."""

    def metrics(self, config: Path) -> Mapping[str, Mapping[str, Mapping[str, float]]]: ...


class Models(Protocol):
    """Library view of the models declared in a config."""

    def features(self, config: Path, model: str) -> Mapping[str, str]: ...


class Dataset(Protocol):
    """Loaded dataset on which the library runs its processes."""

    def columns(self) -> frozenset[str]: ...

    def selection(
        self,
        automl: Path,
        model: str,
        candidates: tuple[str, ...],
        target: Path,
    ) -> str: ...


class Datasets(Protocol):
    """Library access to the dataset that a config points to."""

    def dataset(self, config: Path) -> Dataset: ...


class Library(Models, Datasets, Protocol):
    """Library that both describes the models of a config and loads its dataset."""


@final
class LocalOutbox(Training, Library):
    """The outboxml library running in this process with all its files kept under one root."""

    __slots__ = ("_root",)

    def __init__(self, root: Path) -> None:
        self._root: Final = root

    @override
    def metrics(self, config: Path) -> Mapping[str, Mapping[str, Mapping[str, float]]]:
        return {
            model: {
                sample: {name: float(value) for name, value in result[sample]["full"].items()}
                for sample in ("train", "test")
            }
            for model, result in DataSetsManager(
                config_name=RootedConfig(config, self._root).content(),
                external_config=self._environment(),
            )
            .fit_models()
            .items()
        }

    @override
    def features(self, config: Path, model: str) -> Mapping[str, str]:
        return {
            feature.name: FeatureText(feature).text()
            for feature in NamedModel(
                AllModelsConfig.model_validate(RootedConfig(config, self._root).content()),
                model,
            )
            .config()
            .features
        }

    @override
    def dataset(self, config: Path) -> Dataset:
        models = AllModelsConfig.model_validate(RootedConfig(config, self._root).content())
        return OutboxDataset(
            config,
            models,
            BaseExtractor(data_config=models.data_config).extract_dataset(),
            self._environment(),
        )

    def _environment(self) -> SimpleNamespace:
        return SimpleNamespace(
            results_path=self._root / "results",
            prod_models_path=self._root / "prod",
            work_type_fit="CPU",
            work_type_hptune="CPU",
        )


@final
class OutboxDataset(Dataset):
    """Dataset loaded by outboxml for a models config, kept in memory between processes."""

    __slots__ = ("_environment", "_frame", "_models", "_source")

    def __init__(
        self,
        source: Path,
        models: AllModelsConfig,
        frame: pd.DataFrame,
        environment: SimpleNamespace,
    ) -> None:
        self._source: Final = source
        self._models: Final = models
        self._frame: Final = frame
        self._environment: Final = environment

    @override
    def columns(self) -> frozenset[str]:
        return frozenset(str(column) for column in self._frame.columns)

    @override
    def selection(
        self,
        automl: Path,
        model: str,
        candidates: tuple[str, ...],
        target: Path,
    ) -> str:
        config = NamedModel(self._models, model).config()
        settings = AutoMLConfig.model_validate_json(automl.read_text(encoding="utf-8"))
        columns = [
            column
            for column in self._frame.columns
            if column
            in {
                *candidates,
                *chain.from_iterable(
                    get_useful_columns(self._models.data_config, self._models.models_configs),
                ),
            }
        ]
        preparation = PrepareDataset(
            model_config=config,
            check_prepared=True,
            group_name=self._models.group_name,
        )
        interface = FeatureSelectionInterface(
            feature_selection_config=settings.feature_selection,
            objective=config.objective,
        )
        selection = BaseFS(
            new_features_list=list(candidates),
            parameters=settings.feature_selection,
            data_preprocessor=DataPreprocessor(
                prepare_dataset_interface_dict={model: preparation},
                dataset=self._frame[columns],
                data_config=self._models.data_config,
                version=self._models.version,
                external_config=self._environment,
                retro=True,
            ),
            prepare_data_interface=FeatureSelectionPrepareDataset(model_config=config),
            feature_selection_interface=interface,
        )
        selection.select_features(model_name=model, params=settings.feature_selection.params)
        if interface.last is None:
            raise SelectionFailureError(model)
        chosen = tuple(
            feature
            for feature in preparation.get_model_config().features
            if feature.name in selection.result_features
        )
        if chosen:
            target.write_text(
                json.dumps(
                    ExtendedConfig(self._source, model, chosen).content(),
                    ensure_ascii=False,
                    indent=4,
                ),
                encoding="utf-8",
            )
        return "\n".join(
            (
                (
                    f"Отбор фич для модели {model}: датасет обрезан до {len(columns)} "
                    f"из {len(self._frame.columns)} колонок, "
                    f"в отбор переданы {', '.join(candidates)}."
                ),
                SelectionText(
                    selection.types_dict,
                    interface.last,
                    interface.to_drop,
                    chosen,
                ).text(),
                (
                    f"Копия конфига моделей с выбранными фичами модели {model} сохранена: {target}"
                    if chosen
                    else "Копия конфига не сохранена: ни одна фича не выбрана."
                ),
            ),
        )


@final
class SelectionText:
    """Russian account of one feature selection run for the language model."""

    __slots__ = ("_dropped", "_features", "_top", "_types")

    def __init__(
        self,
        types: Mapping[str, list[str]],
        top: list[str],
        dropped: list[str],
        features: tuple[FeatureModelConfig, ...],
    ) -> None:
        self._types: Final = types
        self._top: Final = top
        self._dropped: Final = dropped
        self._features: Final = features

    def text(self) -> str:
        return "\n".join(
            (
                f"Топ по SHAP от более важной к менее важной: {', '.join(reversed(self._top))}.",
                *(
                    f"{candidate}: тип {self._label(kind)}; "
                    f"в топе SHAP: {self._word(candidate, self._top)}; "
                    f"исключена по корреляции или стабильности: "
                    f"{self._word(candidate, self._dropped)}; "
                    f"выбрана: {self._word(candidate, [f.name for f in self._features])}."
                    for kind, names in self._types.items()
                    for candidate in names
                ),
                (
                    "Параметры выбранных фич в конфиге:"
                    if self._features
                    else "Ни одна фича не выбрана."
                ),
                *(f"{feature.name}: {FeatureText(feature).text()}" for feature in self._features),
            ),
        )

    def _label(self, kind: str) -> str:
        return {
            FeatureTypesForSelection.numeric: "числовая",
            FeatureTypesForSelection.categorical: "категориальная",
            "to_drop": (
                "to_drop (константа, одно доминирующее значение или много пропусков), "
                "в отбор не попала"
            ),
            "object": "object (текст с большим числом значений), в отбор не попала",
            "date": "date, в отбор не попала",
        }.get(kind, str(kind))

    def _word(self, candidate: str, names: Collection[str]) -> str:
        return "да" if candidate in names else "нет"


@final
class FeatureText:
    """Russian description of the type and grouping of a feature in an outboxml config."""

    __slots__ = ("_feature",)

    def __init__(self, feature: FeatureModelConfig) -> None:
        self._feature: Final = feature

    def text(self) -> str:
        return "; ".join(
            (
                "тип числовая"
                if self._feature.replace.get(FeatureEngineering.feature_type)
                == FeatureEngineering.numerical
                else "тип категориальная",
                *(
                    f"{key} = {json.dumps(value, ensure_ascii=False)}"
                    for key, value in to_jsonable_python(
                        self._feature.model_dump(exclude_none=True, exclude={"name"}),
                        fallback=str,
                    ).items()
                    if value not in ({}, [])
                ),
            ),
        )


@final
class NamedModel:
    """Model of an outboxml config picked by its name."""

    __slots__ = ("_models", "_name")

    def __init__(self, models: AllModelsConfig, name: str) -> None:
        self._models: Final = models
        self._name: Final = name

    def config(self) -> ModelConfig:
        for model in self._models.models_configs:
            if model.name == self._name:
                return model
        raise UnknownModelError(self._name, tuple(m.name for m in self._models.models_configs))


@final
class ExtendedConfig:
    """Models config of outboxml as written on disk, with extra features added to one model."""

    __slots__ = ("_features", "_model", "_source")

    def __init__(self, source: Path, model: str, features: tuple[FeatureModelConfig, ...]) -> None:
        self._source: Final = source
        self._model: Final = model
        self._features: Final = features

    def content(self) -> dict[str, Any]:
        return {
            **(origin := json.loads(self._source.read_text(encoding="utf-8"))),
            "models_configs": [
                {
                    **model,
                    "features": [
                        *model["features"],
                        *(
                            to_jsonable_python(
                                feature.model_dump(exclude_none=True, exclude={"bins", "mapping"}),
                                fallback=str,
                            )
                            for feature in self._features
                        ),
                    ],
                }
                if model["name"] == self._model
                else model
                for model in origin["models_configs"]
            ],
        }


@final
class RootedConfig:
    """Models config of outboxml whose data file is resolved against the given root."""

    __slots__ = ("_path", "_root")

    def __init__(self, path: Path, root: Path) -> None:
        self._path: Final = path
        self._root: Final = root

    def content(self) -> dict[str, Any]:
        return {
            **(origin := json.loads(self._path.read_text(encoding="utf-8"))),
            "data_config": {
                **origin["data_config"],
                "local_name_source": str(self._root / origin["data_config"]["local_name_source"]),
            },
        }


@final
class UnknownModelError(ToolError):
    """Name of a model that the config does not declare."""

    def __init__(self, name: str, names: tuple[str, ...]) -> None:
        super().__init__(f"Model {name} is not in the config, its models are {', '.join(names)}")


@final
class SelectionFailureError(ToolError):
    """Feature selection that outboxml swallowed with an error and returned nothing."""

    def __init__(self, model: str) -> None:
        super().__init__(f"Feature selection for model {model} failed inside outboxml")
