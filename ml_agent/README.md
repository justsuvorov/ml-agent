# ml_agent — ML-агент поверх mldataworker

Агент проверяет гипотезы на моделях: читает задание из txt, готовит данные и копии конфигов,
запускает стандартные инструменты обучения (`CascoFLAutoML(...).update_models()`),
при ошибках правит конфиг (встроенные правила + LLM) и перезапускает, сравнивает модели,
получает выводы от LLM (JSON) и отправляет отчёт письмом.

Первая задача: **стоит ли обучать отдельные модели по `BUSINESS_TYPE`**.

Как устроен casco `main_fit` и как агент с ним стыкуется — [docs/casco_main_fit.md](docs/casco_main_fit.md).

## Установка в проект

Скопировать в корень casco-проекта (рядом с `main_fit.py`, `config.py`, `casco_automl.py`):

```
<casco-проект>/
├── ml_agent/                 # пакет агента
├── tasks/business_type_split.txt
├── example_agent.ipynb
├── config.py                 # существующий: добавить блок «ML-агент: LLM» из ml_agent-шаблона
├── casco_automl.py, model_configs/, prod_models/, ...
```

`.env` не используется: всё задаётся в `config.py`. Шаблон — [config.py](config.py); в существующий
`config.py` проекта достаточно перенести блок `ML-агент: LLM` (`llm_model_type`, `vsk_*`, `qwen_*`)
и проверить `email_*`. **Не копируйте шаблонный `config.py` поверх рабочего.**

## Запуск

Из ноутбука — [example_agent.ipynb](example_agent.ipynb):

```python
import config
from ml_agent import MLAgent, setup_logging

setup_logging('INFO')                       # в ячейках — только шаги агента
agent = MLAgent('tasks/business_type_split.txt', config=config)
agent.dry_run()                             # данные, план сегментов, колонки конфига, которых нет в данных
report = agent.run(send_mail=False)         # обучение, сравнение, выводы LLM
report                                      # HTML-отчёт в ячейке
report.comparison                           # таблица сравнения (DataFrame)
agent.send()                                # отправить письмо
```

`MLAgent(..., llm_model_type='offline')` — прогон без внешней LLM (выводы формальные, по метрикам).

CLI (опционально): `python -m ml_agent tasks/business_type_split.txt [--dry-run] [--llm offline]`.

## Задание (txt)

INI-секции + свободный текст после `[instructions]` (уходит в промпт LLM как есть).

| Секция | Что задаёт |
|---|---|
| `[task]` | `name`, `scenario` (`segment_split`), `description` |
| `[data]` | `source_file`, `preprocess_hook`, `base_query`, `segment_column`, `segment_values` (`auto` / список), `min_rows`, `group_small_segments`, `include_baseline`, `temp_dataset` |
| `[automl]` | `class` (`casco_automl:CascoFLAutoML`), `auto_ml_config`, `models_config`, `retro`, `hp_tune`, `mlflow_experiment` (отдельный эксперимент для агента), `extra_kwargs` (JSON), `post_hook` (`casco_results`) |
| `[evaluation]` | `main_metric`, `direction`, `min_improvement`, `compare_with` (`baseline` / `automl`), `baseline_compare_kwarg`, `compare_hook` |
| `[agent]` | `workdir`, `max_fix_attempts`, `drop_missing_features`, `allow_llm_abort`, `isolate_results` |
| `[llm]` | `model_type` — переопределить `config.llm_model_type` |
| `[external_config]` | переопределения атрибутов `config.py` только для этого задания |
| `[email]` | `send`, `subject`, `receivers` (иначе `config.email_receivers`), `class` (`mldataworker.core.email:EMail`) |

Пути — относительно `project_dir` (по умолчанию текущий каталог ноутбука). На время `run()`/`dry_run()`
агент переходит в `project_dir`: casco-код читает `temp_dataset.gzip`, `model_configs/`, `prod_models/` от текущего каталога.

## Как агент сравнивает модели

1. Датасет → `preprocess_hook` (`BUSINESS_TYPE_NEW → BUSINESS_TYPE`, как в `main_fit`) → `base_query` (`VEHICLE_NEW == 0`).
2. **Единое разбиение train/test**: при `separation.kind = random` агент один раз назначает колонку
   `AGENT_IS_TEST` на всём датасете и переводит все эксперименты на неё. Без этого тестовые строки
   сегмента попадали бы в train исходной модели и сравнение было бы смещено в её пользу.
3. `baseline` — исходная модель на всех данных; затем модель на каждый сегмент
   (сегменты меньше `min_rows` — в общий `__OTHER__` или пропускаются).
4. `compare_with = baseline`: сегментная и исходная модели оцениваются через `model_predict`
   на **одних и тех же** тестовых строках сегмента. `compare_with = automl` — штатное сравнение
   класса (`model_to_compare`).
5. LLM получает сравнение, статусы, правки конфигов и инструкцию → JSON
   `{summary, segments[{segment, model, verdict, comment}], recommendation, risks, next_steps}`.

## Исправление ошибок

`update_models()` перехватывает исключения сам, поэтому успех определяется по `status['Fitting']`,
ошибки — из исключения, `automl.errors` и ERROR-сообщений лога. Сбои MLflow/Grafana/почты после
обучения — предупреждения, не ошибки.

1. До запуска: колонки конфига, которых нет в данных сегмента, удаляются из копии конфига
   (вместе с `relative_features`, `intersections`, `treatment_dict`, `column_weight`).
2. Правила: колонка из `KeyError` / `not in index` / `ColumnNotFoundError` → `remove_feature`.
3. LLM: ошибки + сокращённый конфиг + колонки → `{diagnosis, action: patch|retry|skip|abort, patches}`.
4. Исходный конфиг не меняется: попытки — `models_config_attemptN.json`, итог — `models_config_final.json`.
   `data_config/source`, `local_name_source`, `separation` LLM менять не может.

Ограничение: mldataworker не валидирует часть параметров (например, опечатку в `objective` или
неизвестный ключ в `params` catboost-обёртки) — такие ошибки не приводят к падению и агент их не видит.

## Артефакты запуска

`agent_runs/<дата>_<задание>/`: `report.html`, `report.json`, `agent.log`, `llm_log.jsonl`
(все запросы и ответы LLM), `results/` (пиклы mldataworker) и папка на каждый эксперимент
(`dataset.parquet`, конфиги попыток, `attemptN.log` — полный лог, `result.json`).

## Модули

| Модуль | Назначение |
|---|---|
| `agent.py` | `MLAgent`: `dry_run()`, `run()`, `send()` |
| `task.py` | разбор txt-задания |
| `scenarios.py` | сценарии; `segment_split` |
| `runner.py` | запуск эксперимента, перехват лога, цикл исправления, сравнение с baseline |
| `config_editor.py` | безопасные правки JSON-конфига моделей |
| `llm.py` | `AIModel`, `VskAIModel`, `QwenModel`, `OfflineModel`, `LLMClient` (промпт → JSON) |
| `prompts.py` | промпты и JSON-схемы ответов |
| `report.py` | `AgentReport`, таблицы, HTML, письмо через `EMail` |
| `hooks.py` | предобработка данных |

Новый сценарий — функция `(task, runner) -> list[ExperimentResult]` в `SCENARIOS`
(например, проверка новых факторов: `retro = true`, `features_for_research` в `extra_kwargs`).
