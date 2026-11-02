# ml-agent

Агент для MLOps-процессов на библиотеке [outboxml](https://github.com/justsuvorov/outboxml).
Пользователь пишет задание на русском в ячейке Jupyter, например «Проверь фичи PCLASS и EMBARKED
для модели first». Агент сам выбирает инструменты, запускает процессы outboxml и отвечает
на русском. Процессы outboxml настраиваются JSON-конфигами. Агент их читает и проверяет,
а изменения сохраняет в копиях, не трогая исходные файлы.

Сейчас у агента два инструмента: обучение моделей и проверка фич. Выкатка и мониторинг
появятся позже.

## Быстрый старт

```
python -m venv .venv
.venv/Scripts/python -m pip install -e .
```

```python
from pathlib import Path

from ml_agent.chat import OpenAiChat
from ml_agent.checking import FeatureCheck
from ml_agent.fitting import Fitting
from ml_agent.loop import Loop
from ml_agent.outbox import LocalOutbox
from ml_agent.tools import Tools
from ml_agent.workspace import Folder

root = Path("путь/к/рабочей/папке")
agent = Loop(
    OpenAiChat(url="https://<host>/v1", key="<api-key>", model="<model>"),
    Tools(
        {
            "fit_models": Fitting(Folder(root), LocalOutbox(root)),
            "check_features": FeatureCheck(Folder(root), LocalOutbox(root)),
        },
    ),
    12,
)
agent.answer("Обучи модели из конфига configs/titanic.json и покажи метрики на тесте.")
```

Пути в заданиях указываются относительно рабочей папки. Готовый пример с данными —
в [examples/titanic](examples/titanic).

## Как работает агент

`Loop` отправляет языковой модели диалог: инструкцию, описание инструментов, задание
и уже сделанные шаги. В ответ модель присылает один JSON-объект:

- вызов инструмента: `{"thought": "...", "tool": "check_features", "arguments": {...}}`;
- итоговый ответ: `{"thought": "...", "answer": "..."}`.

Агент выполняет вызов и добавляет результат в диалог, и так до итогового ответа или до лимита
шагов. Если лимит исчерпан, выбрасывается `StepLimitError`. В трёх случаях ошибка возвращается
модели как результат шага, и модель может исправить вызов:

- ответ не разобран как JSON;
- инструмента с таким именем нет;
- аргументы не прошли схему или инструмент сообщил об ошибке.

Любые другие ошибки сразу останавливают работу.

Каждый вызов `answer()` начинается с нуля: прошлые задания агент не помнит.

## Инструменты

| Имя | Что делает | Процесс outboxml |
|---|---|---|
| `fit_models` | Обучает все модели конфига и возвращает метрики на train и test | `DataSetsManager.fit_models` |
| `check_features` | Проверяет фичи для модели и сохраняет копию конфига с выбранными фичами | `BaseFS` с `FeatureSelectionInterface` |

Схема аргументов каждого инструмента — pydantic-модель. Её JSON Schema попадает в инструкцию
для модели, а аргументы, которые прислала модель, проверяются по этой схеме.

Как добавить свой инструмент — в руководстве [docs/creating_tools.md](docs/creating_tools.md).

### check_features

Аргументы: `models_config` и `automl_config` (пути к конфигам), `model` (имя модели),
`features` (список фич).

1. Фичи, которые уже есть в модели, показываются с их параметрами из конфига, в отбор они не идут.
   Если новых фич нет, ничего не запускается.
2. Фичи, которых нет в датасете, перечисляются в отчёте.
3. Датасет обрезается до колонок конфига и новых фич, поэтому в отбор попадают только они.
4. Отбор запускается с настройками `feature_selection` из AutoML-конфига.
5. В отчёте есть тип каждого кандидата, топ по SHAP, исключения по корреляции и стабильности,
   итог отбора, а также группировки и параметры выбранных фич (`replace`, `clip`, `encoding`,
   `bins`, `mapping` с WoE).
6. Если что-то выбрано, рядом с конфигом сохраняется копия `<конфиг>_<модель>_features.json`,
   в которой выбранные фичи добавлены к модели. Копию можно сразу передать в `fit_models`.
   Рассчитанные `bins` и `mapping` в копию не пишутся: outboxml пересчитывает их при обучении.

## Языковая модель

Агенту нужен объект с интерфейсом `Llm`: метод `reply(text: str) -> str`. В пакете есть
`OpenAiChat` для любого сервиса, совместимого с OpenAI Chat Completions (`/v1/chat/completions`):
vLLM, VSK AI, Qwen и других. Для другого API достаточно написать свой класс с этим методом.
Инструкция для модели и ответы агента — на русском.

## Устройство кода

| Модуль | Назначение |
|---|---|
| `agent.py`, `llm.py`, `tool.py`, `toolbox.py` | Интерфейсы: агент, языковая модель, инструмент, набор инструментов |
| `loop.py` | `Loop` — цикл агента |
| `dialog.py` | Неизменяемый диалог с моделью: открытый и закрытый |
| `reply.py` | Разбор JSON-шага модели: вызов инструмента или ответ |
| `tools.py` | `Tools` — инструменты по именам и общая инструкция |
| `fitting.py`, `checking.py` | Инструменты `fit_models` и `check_features` |
| `workspace.py` | `Folder` — рабочая папка, не выпускает пути за свои пределы |
| `outbox.py` | `LocalOutbox` — единственный адаптер над outboxml |
| `chat.py` | `OpenAiChat` — клиент OpenAI-совместимого API |

Все вызовы outboxml собраны в `outbox.py`, остальной код с библиотекой напрямую не работает.

## Примеры

[examples/titanic](examples/titanic) — агент на титанике из outboxml. Данные, конфиг моделей
и конфиг AutoML лежат в папке примера, она же рабочая папка агента.

- [agent_titanic.ipynb](examples/titanic/agent_titanic.ipynb) — ноутбук с тремя заданиями:
  проверить обучение, проверить фичи для модели, обучить модель с выбранными фичами. Jupyter
  запускайте из папки примера, а адрес, ключ и модель LLM укажите в ячейке «Языковая модель».
- [agent_titanic.py](examples/titanic/agent_titanic.py) — то же из терминала:

```
set LLM_URL=https://<host>/v1
set LLM_KEY=<api-key>
set LLM_MODEL=<model>
.venv/Scripts/python examples/titanic/agent_titanic.py "Проверь фичи PCLASS и EMBARKED для модели first. Конфиг моделей — configs/titanic.json, конфиг AutoML — configs/automl.json."
```

## Разработка

```
.venv/Scripts/python -m pip install -e .[dev]
.venv/Scripts/pytest
.venv/Scripts/ruff check .
.venv/Scripts/mypy ml_agent tests examples
```

- outboxml — внешняя зависимость, закреплена версия `1.0.1` с PyPI.
- Тесты обучают настоящие модели на титанике. Данные и конфиги скопированы из outboxml v1.0.1
  в `tests/resources/titanic`. Вместо языковой модели в тестах используются фейки, а клиент LLM
  проверяется на локальном сервере. Интернет не нужен.
- Код пишется по TDD в стиле Elegant Objects: неизменяемые `final`-классы с интерфейсами,
  без `None`, геттеров и статических методов. Проверки — ruff со всеми правилами и mypy `--strict`,
  подавления в коде не используются.
- MLflow, Postgres и Grafana пока не нужны. Ошибки outboxml при обращении к ним не мешают обучению.

### Обходы для outboxml 1.0.1

- `BaseFS` перехватывает ошибки отбора и возвращает пустой результат. Адаптер распознаёт этот
  случай, если ошибка случилась до расчёта топа SHAP, и выбрасывает `SelectionFailureError`.
- phik в фильтре по корреляции считает в процессах joblib (loky), а на Windows они иногда падают.
  Поэтому адаптер запускает отбор внутри `joblib.parallel_config(backend="threading")`.
