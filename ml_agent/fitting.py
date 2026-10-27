import json
from collections.abc import Mapping
from typing import Final, final, override

from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError

from ml_agent.outbox import Outbox
from ml_agent.tool import ArgumentsError, Tool
from ml_agent.workspace import Workspace


@final
class FitRequest(BaseModel):
    """Arguments of the tool that trains the models of a config."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    config: str = Field(
        description="Путь к JSON-конфигу моделей outboxml внутри рабочей папки",
    )


@final
class Fitting(Tool):
    """Tool that trains the models of a config and reports their metrics."""

    __slots__ = ("_outbox", "_workspace")

    def __init__(self, workspace: Workspace, outbox: Outbox) -> None:
        self._workspace: Final = workspace
        self._outbox: Final = outbox

    @override
    def manual(self) -> str:
        return (
            "Обучает все модели из конфига моделей outboxml и возвращает их метрики "
            "на обучающей (train) и тестовой (test) выборках.\n"
            f"Аргументы: {json.dumps(FitRequest.model_json_schema(), ensure_ascii=False)}"
        )

    @override
    def run(self, arguments: Mapping[str, JsonValue]) -> str:
        try:
            request = FitRequest.model_validate(arguments)
        except ValidationError as error:
            raise ArgumentsError(error) from error
        return "\n".join(
            f"{model}, {sample}: "
            + ", ".join(f"{name} = {value:.4f}" for name, value in values.items())
            for model, samples in self._outbox.metrics(self._workspace.file(request.config)).items()
            for sample, values in samples.items()
        )
