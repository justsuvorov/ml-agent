"""Промпты для LLM. Ответ всегда — один JSON-объект по описанной схеме."""
import json

SYSTEM_PROMPT = (
    'Ты — ассистент data scientist-а в страховой компании. Ты помогаешь управлять процессом '
    'обучения моделей на библиотеке mldataworker (GLM + CatBoost поверх GLM, задачи частоты/тяжести '
    'убытков). Отвечай строго одним JSON-объектом по заданной схеме, без текста вне JSON. '
    'Пиши по-русски, кратко и по делу.'
)

FIX_CONFIG_SCHEMA = """{
  "diagnosis": "краткое объяснение причины ошибки",
  "action": "patch" | "retry" | "skip" | "abort",
  "patches": [
    {"op": "remove_feature", "model": "<имя модели или *>", "feature": "<колонка>"},
    {"op": "set", "path": "models_configs/<имя модели>/<ключ>/...", "value": <значение>},
    {"op": "delete", "path": "models_configs/<имя модели>/<ключ>"}
  ]
}
action: patch — применить patches и перезапустить; retry — перезапустить без правок (временный сбой);
skip — пропустить этот эксперимент; abort — остановить агента."""

REPORT_SCHEMA = """{
  "summary": "2-4 предложения: что проверяли и главный вывод",
  "segments": [
    {"segment": "<эксперимент>", "model": "<имя модели>", "verdict": "better" | "worse" | "neutral" | "unknown",
     "comment": "интерпретация метрик"}
  ],
  "recommendation": "итоговая рекомендация: внедрять разделение или нет, для каких сегментов",
  "risks": ["ограничения и риски вывода (размер выборки, нестабильность и т.п.)"],
  "next_steps": ["что проверить дальше"]
}"""


def _dump(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, default=str)


def fix_config_prompt(task: dict, experiment: str, attempt: int, errors: list,
                      config_summary: dict, previous_changes: list, dataset_columns: list) -> str:
    return f"""Запуск обучения завершился ошибкой. Предложи правку конфигурации моделей.

## Задание
{_dump(task)}

## Эксперимент: {experiment}, попытка {attempt}

## Ошибки (из исключения и лога)
{_dump(errors)}

## Уже внесённые правки
{_dump(previous_changes)}

## Текущий конфиг моделей (сокращённо)
{_dump(config_summary)}

## Колонки датасета эксперимента
{_dump(dataset_columns)}

Правила:
- меняй только то, что необходимо для устранения ошибки;
- не удаляй целевые колонки и колонки экспозиции;
- если ошибка не связана с конфигом (инфраструктура, сеть, MLflow) — action "retry" или "abort";
- если в сегменте слишком мало данных для обучения — action "skip".

Схема ответа:
{FIX_CONFIG_SCHEMA}"""


def report_prompt(task: dict, instructions: str, comparison: list, experiments: list) -> str:
    return f"""Сформируй выводы по серии экспериментов.

## Инструкция пользователя
{instructions or '-'}

## Задание
{_dump(task)}

## Сравнение моделей по основной метрике
improvement > 0 означает, что новая модель лучше эталона (с учётом направления метрики).
{_dump(comparison)}

## Детали экспериментов (статусы, размеры выборок, правки конфигов, ошибки, полные метрики)
{_dump(experiments)}

Схема ответа:
{REPORT_SCHEMA}"""
