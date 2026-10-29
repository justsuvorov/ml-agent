import csv
import json
import secrets
from pathlib import Path
from typing import Final, final


@final
class Titanic:
    """Workspace with the outboxml 1.0.1 Titanic example whose two models carry the given names."""

    __slots__ = ("_names", "_root")

    def __init__(self, root: Path, names: tuple[str, str]) -> None:
        self._root: Final = root
        self._names: Final = names

    def config(self) -> Path:
        (self._root / "data").mkdir(parents=True, exist_ok=True)
        (self._root / "configs").mkdir(exist_ok=True)
        with (
            self._example("titanic.csv").open(encoding="utf-8", newline="") as source,
            (self._root / "data/titanic.csv").open("w", encoding="utf-8", newline="") as target,
        ):
            reader = csv.DictReader(source)
            writer = csv.DictWriter(target, [*(reader.fieldnames or ()), "ROW_WEIGHT"])
            writer.writeheader()
            writer.writerows(
                {**row, "ROW_WEIGHT": 0.5 + secrets.randbelow(1000) / 1000} for row in reader
            )
        example = json.loads(
            self._example("config-example-titanic.json").read_text(encoding="utf-8"),
        )
        (self._root / "configs/titanic.json").write_text(
            json.dumps(
                {
                    **example,
                    "data_config": {
                        **example["data_config"],
                        "local_name_source": "data/titanic.csv",
                    },
                    "models_configs": [
                        {**model, "name": name}
                        for model, name in zip(example["models_configs"], self._names, strict=True)
                    ],
                },
            ),
            encoding="utf-8",
        )
        return self._root / "configs/titanic.json"

    def automl(self) -> Path:
        (self._root / "configs").mkdir(exist_ok=True)
        (self._root / "configs/automl.json").write_bytes(
            self._example("automl-titanic.json").read_bytes(),
        )
        return self._root / "configs/automl.json"

    def _example(self, name: str) -> Path:
        return Path(__file__).parent / "resources/titanic" / name


def name() -> str:
    return f"m{secrets.token_hex(1 + secrets.randbelow(6))}"
