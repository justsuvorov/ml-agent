# Как создать свой инструмент

Инструмент даёт агенту доступ к одному процессу библиотеки outboxml. Языковая модель читает его
описание, решает вызвать его и присылает аргументы. Инструмент проверяет аргументы, запускает
процесс и возвращает текст, который модель прочитает на следующем шаге.

Руководство разбирает полный путь на примере инструмента `check_drift`, который сравнивает
распределения фич модели на train и test. Это учебный пример: в пакет он не входит.

## Контракт инструмента

Инструмент реализует интерфейс `Tool` из [ml_agent/tool.py](../ml_agent/tool.py):

```python
class Tool(Protocol):
    def manual(self) -> str: ...

    def run(self, arguments: Mapping[str, JsonValue]) -> str: ...
```

- `manual()` возвращает описание на русском и схему аргументов. Это всё, что модель знает об инструменте.
- `run()` получает аргументы от модели и возвращает результат текстом.

Сбой инструмента бывает двух видов:

- `ToolError` и его подклассы агент возвращает модели как результат шага, и она может исправить вызов.
- Любое другое исключение останавливает агента. Так и должно быть при ошибках в коде.

## Шаг 1. Изучите процесс в outboxml

Запустите процесс на титанике и посмотрите на его результат и ошибки. Для `check_drift` нужен
метод `DataSetsManager.check_datadrift(model_name)`:

```
            PSI        KL        JS
SEX    0.001289  0.000391  0.000098
AGE    0.376740  0.019622  0.035838
SIBSP  0.088357  1.703529  0.145210
```

Вот что из этого важно для инструмента:

- Таблица короткая, поэтому её можно целиком отдать модели.
- Для неизвестной модели метод выбрасывает `KeyError`, и агент остановился бы. Такую ошибку модель
  может исправить сама, поэтому имя модели нужно проверить заранее и выдать `ToolError`.

Если адаптер уже умеет нужный процесс, шаги 2–4 не нужны.

## Шаг 2. Опишите процесс интерфейсом

Весь код, который вызывает outboxml, находится в [ml_agent/outbox.py](../ml_agent/outbox.py).
Инструмент видит только узкий интерфейс. Добавьте его туда:

```python
class Monitoring(Protocol):
    """Library process that compares the train and test samples of a model."""

    def drift(self, config: Path, model: str) -> str: ...
```

## Шаг 3. Реализуйте процесс в адаптере

Сначала напишите тесты в [tests/test_outbox.py](../tests/test_outbox.py). Они работают с настоящей
outboxml на титанике во временной папке:

```python
def test_reports_the_drift_of_every_feature_of_the_model(tmp_path: Path) -> None:
    first = name()
    assert_that(
        LocalOutbox(tmp_path).drift(Titanic(tmp_path, (first, name())).config(), first),
        contains_string("SIBSP"),
        "Outbox did not report the drift of the feature",
    )


def test_refuses_the_drift_of_an_unknown_model(tmp_path: Path) -> None:
    assert_that(
        lambda: LocalOutbox(tmp_path).drift(Titanic(tmp_path, (name(), name())).config(), "x"),
        raises(UnknownModelError),
        "Outbox did not refuse the drift of the unknown model",
    )
```

Убедитесь, что тесты падают. Затем добавьте интерфейс к классу (`class LocalOutbox(Training, Library, Monitoring):`)
и напишите метод:

```python
    @override
    def drift(self, config: Path, model: str) -> str:
        return str(
            DataSetsManager(
                config_name=RootedConfig(config, self._root).content(),
                external_config=self._environment(),
            )
            .check_datadrift(
                NamedModel(
                    AllModelsConfig.model_validate(RootedConfig(config, self._root).content()),
                    model,
                )
                .config()
                .name,
            )
            .to_string(),
        )
```

Здесь три готовых помощника адаптера:

- `RootedConfig` отсчитывает путь к данным от рабочей папки.
- `_environment()` отправляет файлы outboxml в рабочую папку.
- `NamedModel` выбрасывает `UnknownModelError`, подкласс `ToolError`, если модели нет в конфиге.

## Шаг 4. Добавьте заглушку библиотеки

Тесты самого инструмента не должны обучать модели. Добавьте в [tests/fakes.py](../tests/fakes.py)
заглушку с готовым ответом:

```python
@final
class Gauge(Monitoring):
    """Library stub that reports a prepared drift for any config and model."""

    __slots__ = ("_report",)

    def __init__(self, report: str) -> None:
        self._report: Final = report

    @override
    def drift(self, config: Path, model: str) -> str:
        return self._report
```

## Шаг 5. Напишите тесты инструмента

Каждому файлу с кодом соответствует один файл тестов, поэтому инструмент `ml_agent/drifting.py`
проверяется в `tests/test_drifting.py`. Вот что стоит проверить:

```python
def test_reports_the_drift_of_the_model(tmp_path: Path) -> None:
    report = phrase()
    (tmp_path / "models.json").write_text("{}", encoding="utf-8")
    assert_that(
        DriftCheck(Folder(tmp_path), Gauge(report)).run(
            {"config": "models.json", "model": name()},
        ),
        contains_string(report),
        "Drift check did not report the drift of the model",
    )


def test_refuses_arguments_without_a_model(tmp_path: Path) -> None:
    assert_that(
        lambda: DriftCheck(Folder(tmp_path), Gauge(phrase())).run({"config": phrase()}),
        raises(ArgumentsError),
        "Drift check did not refuse the arguments without a model",
    )


def test_refuses_a_missing_config(tmp_path: Path) -> None:
    assert_that(
        lambda: DriftCheck(Folder(tmp_path), Gauge(phrase())).run(
            {"config": name(), "model": name()},
        ),
        raises(MissingFileError),
        "Drift check did not refuse the missing config",
    )


def test_manual_describes_the_model_argument(tmp_path: Path) -> None:
    assert_that(
        DriftCheck(Folder(tmp_path), Gauge(phrase())).manual(),
        contains_string('"model"'),
        "Drift check manual did not describe the model argument",
    )
```

Ещё один тест проверяет весь путь через агента. Фейк `Parrot` вызывает инструмент, а затем отвечает
всем текстом, который получил, поэтому результат инструмента виден в ответе:

```python
def test_shows_the_drift_to_the_language_model(tmp_path: Path) -> None:
    report = phrase()
    (tmp_path / "models.json").write_text("{}", encoding="utf-8")
    assert_that(
        Loop(
            Parrot(
                json.dumps(
                    {
                        "tool": "check_drift",
                        "arguments": {"config": "models.json", "model": name()},
                    },
                ),
            ),
            Tools({"check_drift": DriftCheck(Folder(tmp_path), Gauge(report))}),
            2,
        ).answer(phrase()),
        contains_string(report),
        "Agent did not show the drift to the language model",
    )
```

## Шаг 6. Напишите инструмент

Файл `ml_agent/drifting.py` состоит из схемы аргументов и класса инструмента:

```python
@final
class DriftRequest(BaseModel):
    """Arguments of the tool that compares the samples of a model."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    config: str = Field(description="Путь к JSON-конфигу моделей outboxml внутри рабочей папки")
    model: str = Field(description="Имя модели из конфига моделей")


@final
class DriftCheck(Tool):
    """Tool that reports how the features of a model drift between its train and test samples."""

    __slots__ = ("_monitoring", "_workspace")

    def __init__(self, workspace: Workspace, monitoring: Monitoring) -> None:
        self._workspace: Final = workspace
        self._monitoring: Final = monitoring

    @override
    def manual(self) -> str:
        return (
            "Сравнивает распределения фич модели на обучающей и тестовой выборках "
            "и возвращает PSI, KL и JS по каждой фиче.\n"
            f"Аргументы: {json.dumps(DriftRequest.model_json_schema(), ensure_ascii=False)}"
        )

    @override
    def run(self, arguments: Mapping[str, JsonValue]) -> str:
        try:
            request = DriftRequest.model_validate(arguments)
        except ValidationError as error:
            raise ArgumentsError(error) from error
        return self._monitoring.drift(self._workspace.file(request.config), request.model)
```

- `extra="forbid"` отклоняет лишние аргументы, и модель узнаёт о своей ошибке.
- Описания полей пишите на русском: они попадают в инструкцию для модели.
- `ArgumentsError` и ошибки `Workspace` уже являются `ToolError`, поэтому модель их увидит.

## Шаг 7. Подключите инструмент к агенту

Имя инструмента задаётся при сборке агента, а не в самом классе:

```python
tools = Tools(
    {
        "fit_models": Fitting(Folder(root), LocalOutbox(root)),
        "check_features": FeatureCheck(Folder(root), LocalOutbox(root)),
        "check_drift": DriftCheck(Folder(root), LocalOutbox(root)),
    },
)
```

`Tools` сам добавит описание инструмента в инструкцию для модели.

## Шаг 8. Запустите проверки

```
.venv/Scripts/pytest
.venv/Scripts/ruff check .
.venv/Scripts/mypy ml_agent tests examples
```

## Правила

**Описание и результат**

- В `manual()` укажите, что делает инструмент и что он возвращает: модель выбирает инструмент по этому тексту.
- Возвращайте короткую сводку. Большие таблицы занимают место в каждом следующем запросе к модели,
  поэтому отдавайте только то, что нужно для вывода.

**Ошибки**

- Выдавайте `ToolError`, если модель может исправить вызов: другой путь, другое имя, другие аргументы.
- Переводите в `ToolError` известные ошибки библиотеки, как `KeyError` в примере.
- Текст ошибки формируйте в конструкторе подкласса: `UnknownModelError(name, names)`. Так требует ruff.
- Не верьте молчанию outboxml. `AutoMLManager.update_models()` и `BaseFS.select_features()` сами
  перехватывают исключения, поэтому проверяйте результат: `status`, `errors`, `interface.last`.

**Файлы и конфиги**

- Получайте пути только через `Workspace`: `file()` для существующих файлов, `place()` для новых.
  Так имя от модели не выведет запись за пределы рабочей папки.
- Не меняйте исходные конфиги. Сохраняйте копию рядом и называйте её путь в результате, как `check_features`.

**Код**

- Вызывайте outboxml только из `outbox.py`. Там же остаются вызовы с `None`.
- Давайте инструменту только нужный интерфейс: `Monitoring`, а не весь `LocalOutbox`.
- Классы делайте `final` и неизменяемыми: `__slots__`, атрибуты `Final`, не больше четырёх атрибутов.
- В тестах используйте заглушки вместо библиотеки, а настоящее обучение оставляйте для тестов адаптера.

**Внешние действия**

В агенте пока нет механизма подтверждений. Не добавляйте инструменты с внешними последствиями,
например выкатку модели или отправку писем, пока такого механизма нет.

## Чек-лист

- [ ] Процесс запущен на титанике, его результат и ошибки изучены.
- [ ] Вызов outboxml находится в `outbox.py` за отдельным интерфейсом.
- [ ] Для адаптера есть тесты на титанике, для инструмента — тесты с заглушкой.
- [ ] Неверные аргументы и исправимые ошибки дают `ToolError`.
- [ ] `manual()` на русском описывает инструмент и содержит схему аргументов.
- [ ] Результат короткий.
- [ ] Инструмент подключён в `Tools`.
- [ ] pytest, ruff и mypy проходят.
