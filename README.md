# ml-agent

Агент принимает задание на русском языке в ячейке Jupyter, выбирает инструменты
и выполняет процессы библиотеки [outboxml](https://github.com/justsuvorov/outboxml):
обучение, анализ фичей, выкатку и мониторинг моделей. Все процессы настраиваются
JSON-конфигами, которые агент читает, проверяет и правит в копиях.

Языковая модель — внешний сервис, который передаётся агенту через общий интерфейс.

## Разработка

```
python -m venv .venv
.venv/Scripts/python -m pip install -e .[dev]
.venv/Scripts/pytest
.venv/Scripts/ruff check .
.venv/Scripts/mypy ml_agent tests
```

## Примеры

[examples/titanic](examples/titanic) — агент на примере титаника из outboxml: данные, конфиг моделей
и конфиг AutoML лежат в папке примера, она же рабочая папка агента. Языковая модель — любой сервис,
совместимый с OpenAI Chat Completions (`/v1/chat/completions`).

- [agent_titanic.ipynb](examples/titanic/agent_titanic.ipynb) — ноутбук с тремя заданиями: проверить
  обучение, проверить фичи для модели, обучить модель с выбранными фичами. Запускайте Jupyter из
  папки примера и укажите адрес, ключ и модель LLM в ячейке «Языковая модель».
- [agent_titanic.py](examples/titanic/agent_titanic.py) — то же из терминала:

```
set LLM_URL=https://<host>/v1
set LLM_KEY=<api-key>
set LLM_MODEL=<model>
.venv/Scripts/python examples/titanic/agent_titanic.py "Проверь фичи PCLASS и EMBARKED для модели first. Конфиг моделей — configs/titanic.json, конфиг AutoML — configs/automl.json."
```
