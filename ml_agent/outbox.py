import json
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Final, Protocol, final, override

from outboxml.datasets_manager import DataSetsManager


class Outbox(Protocol):
    """Machine learning library that runs the processes of the agent."""

    def metrics(self, config: Path) -> Mapping[str, Mapping[str, Mapping[str, float]]]: ...


@final
class LocalOutbox(Outbox):
    """The outboxml library running in this process with all its files kept under one root."""

    __slots__ = ("_root",)

    def __init__(self, root: Path) -> None:
        self._root: Final = root

    @override
    def metrics(self, config: Path) -> Mapping[str, Mapping[str, Mapping[str, float]]]:
        (self._root / "results").mkdir(exist_ok=True)
        return {
            model: {
                sample: {name: float(value) for name, value in result[sample]["full"].items()}
                for sample in ("train", "test")
            }
            for model, result in DataSetsManager(
                config_name=RootedConfig(config, self._root).content(),
                external_config=SimpleNamespace(
                    results_path=self._root / "results",
                    prod_models_path=self._root / "prod",
                    work_type_fit="CPU",
                    work_type_hptune="CPU",
                ),
            )
            .fit_models()
            .items()
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
