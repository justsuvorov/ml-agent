import json
from collections.abc import Mapping
from typing import Final, final, override

from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError

from ml_agent.outbox import Library
from ml_agent.tool import ArgumentsError, Tool
from ml_agent.workspace import Workspace


@final
class CheckRequest(BaseModel):
    """Arguments of the tool that checks candidate features for a model."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    models_config: str = Field(description="Путь к JSON-конфигу моделей outboxml в рабочей папке")
    automl_config: str = Field(
        description="Путь к JSON-конфигу AutoML outboxml с настройками feature_selection",
    )
    model: str = Field(description="Имя модели из конфига моделей")
    features: list[str] = Field(min_length=1, description="Имена фич, которые нужно проверить")


@final
class FeatureCheck(Tool):
    """Tool that checks candidate features for a model by the feature selection of outboxml."""

    __slots__ = ("_library", "_workspace")

    def __init__(self, workspace: Workspace, library: Library) -> None:
        self._workspace: Final = workspace
        self._library: Final = library

    @override
    def manual(self) -> str:
        return (
            "Проверяет фичи для модели: ищет их в актуальном конфиге и в датасете, затем "
            "запускает отбор фич outboxml (feature selection) только для новых фич, которые "
            "есть в датасете. Датасет обрезается до колонок конфига и проверяемых фич. "
            "Возвращает параметры фич, уже включённых в модель, фичи, которых нет в датасете, "
            "тип каждого кандидата, топ по SHAP, итог отбора и группировки выбранных фич.\n"
            f"Аргументы: {json.dumps(CheckRequest.model_json_schema(), ensure_ascii=False)}"
        )

    @override
    def run(self, arguments: Mapping[str, JsonValue]) -> str:
        try:
            request = CheckRequest.model_validate(arguments)
        except ValidationError as error:
            raise ArgumentsError(error) from error
        config = self._workspace.file(request.models_config)
        known = self._library.features(config, request.model)
        fresh = [feature for feature in request.features if feature not in known]
        return "\n".join(
            (
                *(
                    f"Уже в модели {request.model}, отбор не нужен. {feature}: {known[feature]}"
                    for feature in request.features
                    if feature in known
                ),
                self._selection(request, fresh) if fresh else "Все фичи уже в модели.",
            ),
        )

    def _selection(self, request: CheckRequest, fresh: list[str]) -> str:
        dataset = self._library.dataset(self._workspace.file(request.models_config))
        missing = [feature for feature in fresh if feature not in dataset.columns()]
        candidates = tuple(feature for feature in fresh if feature in dataset.columns())
        return "\n".join(
            (
                *((f"Нет в датасете: {', '.join(missing)}.",) if missing else ()),
                dataset.selection(
                    self._workspace.file(request.automl_config),
                    request.model,
                    candidates,
                )
                if candidates
                else "Отбор не запускался: ни одной новой фичи нет в датасете.",
            ),
        )
