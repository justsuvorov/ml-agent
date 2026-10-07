# ml_agent — ML-агент поверх mldataworker

Агент проверяет гипотезы на моделях: читает задание из txt, готовит данные и копии конфигов,
запускает стандартные инструменты обучения (`CascoFLAutoML(...).update_models()`),
при ошибках правит конфиг (встроенные правила + LLM) и перезапускает, сравнивает модели,
получает выводы от LLM (JSON) и отправляет отчёт письмом.

Основной сценарий `factor_split`: **модели, обученные по группам одного фактора, против общих моделей**.
Первая задача — тип бизнеса б/у ТС (ПРОЛОНГАЦИЯ, Б/У ТС ПЕРЕХОД, Б/У ТС ИНОЕ), блоки моделей —
частота и тяжесть + тоталь, критерий — фин. эффект (`FinEffectMetric`, Margin).

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
agent.dry_run()                             # группы, блоки моделей, план обучений
report = agent.run(send_mail=False)         # обучение, комбинации, фин. эффект, выводы LLM
report                                      # HTML-отчёт в ячейке
report.fin_effect                           # Margin: группа × комбинация против «все общие»
report.comparison                           # статистические метрики моделей групп против общих
agent.send()                                # отправить письмо
```

`MLAgent(..., llm_model_type='offline')` — прогон без внешней LLM (выводы формальные, по метрикам).

CLI (опционально): `python -m ml_agent tasks/business_type_split.txt [--dry-run] [--llm offline]`.

## Задание (txt)

INI-секции + свободный текст после `[instructions]` (уходит в промпт LLM как есть).

| Секция | Что задаёт |
|---|---|
| `[task]` | `name`, `scenario` (`factor_split`), `description` |
| `[data]` | `source_file`, `preprocess_hook`, `segment_column` (фактор), `segment_values` (группы: `auto` / список), `base_query` (область гипотезы и фильтр обучения по умолчанию), `min_rows`, `group_small_segments`, `temp_dataset` |
| `[automl]` | общие для всех блоков: `class` (`casco_automl:CascoFLAutoML`), `auto_ml_config`, `retro`, `hp_tune`, `mlflow_experiment` (отдельный эксперимент для агента), `extra_kwargs` (JSON), `post_hook` (`casco_results`) |
| `[block.<имя>]` | блок моделей: `models_config`, `base_query` (пусто — без фильтра), переопределения `[automl]`; прочие ключи (`model_to_compare`) — в конструктор automl-класса |
| `[evaluation]` | фин. эффект: `business_metric` (`business_metric:FinEffectMetric`), `business_metric_kwargs`, `business_metric_key` (`Margin`), `compare_function` (`mldataworker.automl:compare_pickle_models`), `extra_columns`, `models_order`, `no_exposure_models`, `query`, `min_improvement`; стат. метрики: `main_metric`, `direction`, `baseline_compare_kwarg`; `compare_hook` |
| `[agent]` | `workdir`, `max_fix_attempts`, `drop_missing_features`, `allow_llm_abort`, `isolate_results` |
| `[llm]` | `model_type` — переопределить `config.llm_model_type` |
| `[external_config]` | переопределения атрибутов `config.py` только для этого задания |
| `[email]` | `send`, `subject`, `receivers` (иначе `config.email_receivers`), `class` (`mldataworker.core.email:EMail`) |

Пути — относительно `project_dir` (по умолчанию текущий каталог ноутбука). На время `run()`/`dry_run()`
агент переходит в `project_dir`: casco-код читает `temp_dataset.gzip`, `model_configs/`, `prod_models/` от текущего каталога.

## Как агент сравнивает модели

1. Датасет → `preprocess_hook` (`AGENT_BUSINESS_TYPE = BUSINESS_TYPE_NEW.upper()`, как в `compare_models`).
2. **Единое разбиение train/test** на всех строках: при `separation.kind = random` агент назначает колонку
   `AGENT_IS_TEST` и переводит на неё все обучения. Тест группы не попадает в train общей модели,
   а частота и тяжесть видят одни и те же тестовые полисы.
3. Для каждого блока (`[block.frequency]`, `[block.severity_total]`): общая модель на данных блока
   (`base_query` блока) + модель на каждой группе (группы меньше `min_rows` — в `__OTHER__` или пропускаются).
   Модель группы сравнивается с общей моделью блока на тех же строках (`model_predict`, стат. метрики).
4. **Комбинации**: для каждой группы — все варианты «общая / групповая» по блокам
   (2 блока → 3 комбинации + эталон «все общие»). Пиклы блоков склеиваются в порядке `models_order`
   (частота, тяжесть, тоталь — как `CascoRelease`), у `no_exposure_models` снимается экспозиция
   (как `CompareCascoModels`).
5. **Фин. эффект**: `compare_pickle_models(тест группы в области base_query, комбинация, эталон,
   FinEffectMetric())` → Margin (> 0 — комбинация лучше эталона). Сводная «группа × комбинация» + сумма по группам.
6. LLM получает фин. эффект (главное), стат. метрики, статусы, правки конфигов и инструкцию → JSON
   `{summary, groups[{group, best_combination, verdict, comment}], recommendation, risks, next_steps}`.

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
| `scenarios.py` | сценарий `factor_split`: блоки × группы |
| `combinations.py` | комбинации моделей блоков и их оценка бизнес-метрикой |
| `runner.py` | запуск эксперимента, перехват лога, цикл исправления, сравнение с baseline |
| `config_editor.py` | безопасные правки JSON-конфига моделей |
| `llm.py` | `AIModel`, `VskAIModel`, `QwenModel`, `OfflineModel`, `LLMClient` (промпт → JSON) |
| `prompts.py` | промпты и JSON-схемы ответов |
| `report.py` | `AgentReport`, таблицы, HTML, письмо через `EMail` |
| `hooks.py` | предобработка данных |

Новый сценарий — функция `(task, runner) -> list[ExperimentResult]` в `SCENARIOS`
(например, проверка новых факторов: `retro = true`, `features_for_research` в `extra_kwargs`).
