# Как работает casco `main_fit` и что из этого использует агент

Восстановлено по модулям `casco/`: `main_fit.py`, `casco_automl.py`, `casco_feature_selection.py`,
`business_metric.py`, `compare_models.py`, `release.py`, `reports.py`, `casco_utils.py`.
В репозитории нет `config.py`, `casco_fl_extractor.py`, `feature_catalog.py`, `plots.py`,
`model_configs/*.json`, `prod_models/*.pickle` — их поведение выведено по вызовам.

## 1. Конвейер `main_fit`

```
ds_kasko.dataset_kasko_new ──(CascoExtractor, если parquet_name не задан)──► temp_dataset_main_fit.gzip
                                                                                    │
     ┌──────────────────────────────────────────────────────────────────────────────┤
     ▼                                   ▼                                          ▼
 VEHICLE_NEW == 0                   весь датасет                              VEHICLE_NEW == 1
 → temp_dataset.gzip                → temp_dataset.gzip                       → temp_dataset.gzip
 CascoFLAutoML                      CascoFLAutoML                             CascoFLAutoML
  old_cars_models_config             severity_total_config                     new_cars_models_config
  (частота, б/у ТС)                  (тяжесть + тоталь)                         (частота, новые ТС)
  model_to_compare=                  model_to_compare=                         model_to_compare=
  sweetie_fox_old_cars_frequency     sweetie_fox_severity_total                sweetie_fox_new_cars_frequency
     │ update_models() + casco_results()   │                                         │
     └───────────────────────┬─────────────┴─────────────────────────────────────────┘
                             ▼
              CascoRelease.pickle_models()
              new_cars = [частота новых, тяжесть, тоталь, суброгация*, фрод*]
              old_cars = [частота б/у,   тяжесть, тоталь, суброгация*, фрод*]   (* из prod_models)
              → kasko_fl_new_cars_<ts>.pickle, kasko_fl_old_cars_<ts>.pickle
              → service_ensemble(): Ensemble по VEHICLE_NEW → <model_version>.pickle
                             ▼
              cube_export() ×2 → ModelCubeReport: прогноз по договорам (→ Oracle DS_KASKO при db_export)
                             ▼
              compare_models(): фин. эффект (Margin) новых релизных пиклов против старых
              по BUSINESS_TYPE (= BUSINESS_TYPE_NEW.upper()), продукт КЛАССИКА, test-индекс severity_total
                             ▼
              CascoEMail.success_mail([new, old, sev_total, fin_effect_df]) — письмо
```

### Шаг обучения: `CascoFLAutoML(...).update_models(send_mail=False)`

`CascoFLAutoML` наследует `AutoMLManager` и переопределяет:

| Этап | Что делает casco-версия |
|---|---|
| `__init__` | MLflow через `CascoMLFLow` (Keycloak, креды из глобального `config`); **`FactorPlots` читает `temp_dataset.gzip` из текущего каталога** — данные для графиков факторов |
| `feature_selection` (при `retro=True`) | `RetroFS` → кандидаты (или явный `features_for_research`); `CascoBaseFS`: квантильный клиппинг, бины по Фридману–Диаконису (≤5), phik-корреляция; затем `OptimalGLMBinning` с проверкой стабильности — меняет конфиг модели |
| `hp_tuning` (при `hp_tune=True`) | базовый Optuna |
| `fit_models` → `save_results` | пикл `<group>_<ts>.pickle` и json конфига в `results_path` |
| `compare_with_previous` | **без try/except**: читает `model_to_compare` (или последний пикл из `prod_models_path`), прогоняет его `model_predict` на данных экстрактора (то есть на тех же данных, `local_name_source`) и копирует экспозицию из новой модели; метрики → `compare_metrics_df`, графики → `figures` |
| `deployment` / `review` / MLflow | флаг deployment по `inference_criteria`; HTML-отчёт; лог в MLflow (коэффициенты GLM в xlsx, zip графиков) |

Исключение на любом этапе перехватывается внутри `update_models` (только `logger.error`), состояние —
в `automl.status`. В `main_fit` вызов ещё и обёрнут в `try: ... except: pass`.

`casco_results()`: коэффициенты GLM + экспозиция по уровням → `<model>_coeffs.xlsx`;
`FeatureCatalog('casco_fl_feature_catalog.json')` — **общий файл каталога фич** с тегами deployment/версии;
zip графиков факторов (`FactorPlots` + `model_configs/model_config_kasko.json`).

### Бизнес-метрика

`business_metric.FinEffectMetric` (Margin): нужны модели **частоты + тяжести + тоталя**,
`GLM_INTERVAL_PREMIUM` и `SUM_INSURED` в `extra_columns`. Ожидаемый убыток → loss ratio → сортировка
по прогнозу → «срезается» хвост выше `cut_value` (по экспозиции, тип 1) → маржа
`(убыток_среза / премия_среза × суброгация + КВ) × премия − премия`. Разница новой и старой модели = фин. эффект.

На этапе обучения `business_metric=None`, так как в конфиге частоты нет тяжести и тоталя. Margin считается
только в `compare_models`, на собранных релизных пикликах: `[:3]` = частота, тяжесть, тоталь.

Параметры по типам бизнеса в `compare_models`:

| BUSINESS_TYPE | Пикл | cut_value | kv_value |
|---|---|---|---|
| НОВОЕ ТС | new_cars | 0.95 | 0.50 |
| ПРОЛОНГАЦИЯ | old_cars | 0.95 | 0.35 |
| Б/У ТС ПЕРЕХОД | old_cars | 0.90 | 0.35 |
| Б/У ТС ИНОЕ | old_cars | 0.90 | 0.35 |

Каждая выборка — `validation_set`: случайная подвыборка (`random_state=randint(0, 100)`),
повторяется `cv_folds` раз, берутся медианы премии и маржи и пересчитываются на годовые сборы
(константы в коде).

## 2. Что это значит для агента

| В `main_fit` | В агенте |
|---|---|
| фильтр + запись `temp_dataset.gzip` перед каждым `CascoFLAutoML` | так же (`[data] temp_dataset`), плюс датасет эксперимента подставляется в копию конфига |
| пути от текущего каталога | `MLAgent` на время работы делает `chdir` в `project_dir` |
| `try: update_models() except: pass` | успех = `status['Fitting']`; ошибки из исключения, `automl.errors`, ERROR-лога → правила/LLM → перезапуск |
| `model_to_compare` = прод-пикл | baseline с прод-пиклом; сегменты — с пиклом baseline (`baseline_compare_kwarg`) + собственная оценка обеих моделей на одних тестовых строках |
| разбиение train/test заново в каждом запуске | единое `AGENT_IS_TEST` для всех экспериментов |
| MLflow-эксперимент из `auto_ml_config` | `[automl] mlflow_experiment` — копия конфига с отдельным экспериментом |
| `BUSINESS_TYPE = BUSINESS_TYPE_NEW.upper()` | хук пишет это в `AGENT_BUSINESS_TYPE` (исходные колонки не меняются) |
| б/у ТС = ПРОЛОНГАЦИЯ, Б/У ТС ПЕРЕХОД, Б/У ТС ИНОЕ | эти три сегмента явно заданы в задании |

## 3. Риски и замечания по коду casco

1. **Побочные эффекты `casco_results`** — дописывает общий `casco_fl_feature_catalog.json` с тегами
   экспериментов. Для прогонов агента может быть нежелательно (`post_hook` можно отключить).
2. **`release.service_ensemble` меняет глобальный `config.prod_models_path = config.results_path`.**
   Если в том же ядре ноутбука до агента запускался `main_fit`, `load_last_pickle_models_result` будет
   смотреть в `results`. В ноутбуке агента есть `importlib.reload(config)`.
3. **`reports.py`: `from mldataworker import config` после `import config`** — в модуле `reports`
   имя `config` указывает на конфиг библиотеки, а не проекта (используется в `FactorPlots`,
   `ModelCubeReport` по умолчанию).
4. **`compare_with_previous`, ветка severity** — сначала `self._results[sev] = deepcopy(severity_full)`,
   затем `severity_full.metrics = self._results[sev].metrics`, то есть метрики подменяются сами на себя.
   Для суброгации порядок обратный и корректный. В `main_fit` `severity_model_name=None`, поэтому сейчас не срабатывает.
5. **`CascoBaseFS._prepare_feature`**: результат `serie.apply(... "OTHER")` не присваивается —
   группировка редких категорий не применяется.
6. **`compare_models` недетерминирован**: `validation_set` использует `randint` как `random_state`.
   Фин. эффект между запусками колеблется — для решения агента нужен фиксированный seed.
7. В `compare_models` захардкожены имена старых пикликов и годовые сборы и запросы.

## 4. Следующий шаг: фин. эффект по сегментам в агенте

Сейчас агент сравнивает частоту по статистической метрике (`main_metric`). Решение в casco
принимается по Margin, поэтому логичный `compare_hook`:

1. для каждого сегмента собрать пикл `[частота сегмента] + [тяжесть, тоталь из релизного old_cars-пикла]`,
   для эталона — `[частота baseline] + [те же тяжесть, тоталь]` (как `CascoRelease.pickle_models`);
2. на тестовых строках сегмента (`AGENT_IS_TEST == 1`) вызвать
   `CompareCascoModels(...).result()` с `FinEffectMetric(cut_value, kv_value)` из таблицы выше;
3. вернуть Margin по сегментам — агент добавит его в отчёт и в промпт LLM.

Для этого нужен релизный пикл тяжести и тоталя (например, `kasko_fl_old_cars_<ts>.pickle`)
и доступ хука к `results_path` запуска.
