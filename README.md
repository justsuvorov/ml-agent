# ml-agent

Агент принимает задание на русском языке в ячейке Jupyter, выбирает инструменты
и выполняет процессы библиотеки [outboxml](https://github.com/justsuvorov/outboxml):
обучение, анализ фичей, выкатку и мониторинг моделей. Все процессы настраиваются
JSON-конфигами, которые агент читает, проверяет и правит в копиях.

Языковая модель — внешний сервис, который передаётся агенту через общий интерфейс.

## Разработка

```
python -m venv .venv
.venv/Scripts/python -m pip install -e D:/Projects/outboxml -e .[dev]
.venv/Scripts/pytest
.venv/Scripts/ruff check .
.venv/Scripts/mypy ml_agent tests
```
