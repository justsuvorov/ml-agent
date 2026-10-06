"""Запуск одного эксперимента обучения с автоматическим исправлением конфига.

Эксперимент = (датасет, копия конфига моделей) -> ``<automl class>(...).update_models()``.
``AutoMLManager.update_models`` сам перехватывает исключения и пишет их в лог, поэтому
успешность определяется по ``automl.status['Fitting']`` + перехваченным ERROR-сообщениям loguru.
При неудаче агент: (1) применяет встроенные правила (отсутствующие колонки),
(2) спрашивает LLM, (3) перезапускает с исправленной копией конфига.
"""
import json
import re
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from loguru import logger

from ml_agent import prompts
from ml_agent.config_editor import ModelsConfigEditor, PatchError
from ml_agent.llm import LLMClient
from ml_agent.task import TaskSpec
from ml_agent.utils import import_object, round_floats

NEW_TEST = 'Новая модель||Тестовая выборка'
OLD_TEST = 'Предыдущая модель||Тестовая выборка'
SPLIT_COLUMN = 'AGENT_IS_TEST'  # единое разбиение train/test (0/1), назначается сценарием
NON_METRIC_COLUMNS = {'level_0', 'index', 'Metric group', 'Model', 'Имя модели'}


class AgentAbort(RuntimeError):
    pass


@dataclass
class ExperimentResult:
    name: str
    description: str
    n_rows: int
    status: str = 'pending'  # success | failed | skipped
    attempts: int = 0
    config_changes: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    automl_status: Dict[str, Any] = field(default_factory=dict)
    test_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    compare_metrics: List[dict] = field(default_factory=list)
    # сравнение с исходной (baseline) моделью на тех же тестовых строках: {model: {'new': {...}, 'reference': {...}}}
    vs_baseline: Dict[str, Dict[str, Dict[str, float]]] = field(default_factory=dict)
    new_features: Dict[str, Any] = field(default_factory=dict)
    result_pickle: Optional[str] = None
    final_config: Optional[str] = None
    automl: Any = field(default=None, repr=False)

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if k != 'automl'}

    def reference_metric(self, model: str, metric: str, source: str = 'baseline') -> Optional[float]:
        if source == 'baseline' and model in self.vs_baseline:
            return self.vs_baseline[model]['reference'].get(metric)
        return self._compare_value(model, metric, OLD_TEST)

    def new_metric(self, model: str, metric: str, source: str = 'baseline') -> Optional[float]:
        if source == 'baseline' and model in self.vs_baseline:
            return self.vs_baseline[model]['new'].get(metric)
        value = self._compare_value(model, metric, NEW_TEST)
        if value is None:
            value = self.test_metrics.get(model, {}).get(metric)
        return value

    def _compare_value(self, model, metric, model_label) -> Optional[float]:
        for row in self.compare_metrics:
            if row.get('Имя модели') == model and row.get('Model') == model_label \
                    and row.get('Metric group', 'full') == 'full' and metric in row:
                return row[metric]
        return None


@dataclass
class _Attempt:
    ok: bool
    errors: List[str]
    warnings: List[str]
    automl: Any = None


class ExperimentRunner:
    def __init__(self, task: TaskSpec, llm: LLMClient, run_dir: Path, external_config):
        self.task = task
        self.llm = llm
        self.run_dir = run_dir
        self.external_config = external_config
        self.automl_class = import_object(task.automl.get('class', 'mldataworker.automl_manager:AutoMLManager'))
        self.models_config = task.resolve(task.automl['models_config'])
        self.auto_ml_config = task.resolve(task.automl['auto_ml_config'])
        self.target_columns = set()
        self.results: List[ExperimentResult] = []  # все эксперименты запуска (в т.ч. при остановке)

    def run(self, name: str, data: pd.DataFrame, description: str = '',
            extra_kwargs: dict = None) -> ExperimentResult:
        exp_dir = self.run_dir / name
        exp_dir.mkdir(parents=True, exist_ok=True)
        result = ExperimentResult(name=name, description=description, n_rows=len(data))
        self.results.append(result)
        logger.info(f'AGENT||Эксперимент {name}: {description}, строк {len(data)}')

        editor = self._prepare_config(data, exp_dir, result)
        if result.status == 'failed':
            return result

        for attempt in range(1, self.task.max_fix_attempts + 2):
            result.attempts = attempt
            config_path = editor.save(exp_dir / f'models_config_attempt{attempt}.json')
            outcome = self._run_once(config_path, exp_dir, attempt, extra_kwargs or {})
            result.warnings += outcome.warnings
            if outcome.ok:
                result.status = 'success'
                self._collect(result, outcome.automl)
                break
            result.errors = outcome.errors
            if attempt > self.task.max_fix_attempts:
                result.status = 'failed'
                break

            action = self._fix(editor, name, attempt, outcome.errors, data)
            result.config_changes = list(editor.changes)
            if action == 'skip':
                result.status = 'skipped'
                break
            if action == 'abort':
                result.status = 'failed'
                self._finish(result, editor, exp_dir)
                raise AgentAbort(f'Эксперимент {name}: агент остановлен по решению LLM')

        self._finish(result, editor, exp_dir)
        return result

    # ---------- подготовка ----------
    def _prepare_config(self, data: pd.DataFrame, exp_dir: Path, result: ExperimentResult) -> ModelsConfigEditor:
        dataset_path = exp_dir / 'dataset.parquet'
        data.to_parquet(dataset_path)
        temp_dataset = self.task.data.get('temp_dataset')
        if temp_dataset:  # совместимость со скриптами, которые читают фиксированный temp-файл
            data.to_parquet(self.task.resolve(temp_dataset))

        editor = ModelsConfigEditor.from_file(self.models_config)
        editor.data.setdefault('data_config', {})
        editor.data['data_config']['source'] = 'parquet'
        editor.data['data_config']['local_name_source'] = str(dataset_path)
        if SPLIT_COLUMN in data.columns:
            data_config = editor.data['data_config']
            data_config['separation'] = {**(data_config.get('separation') or {}), 'kind': 'date',
                                         'period_column': [SPLIT_COLUMN],
                                         'train_period': [0, 0], 'test_period': [1, 1]}
            data_config['extra_columns'] = list(dict.fromkeys((data_config.get('extra_columns') or [])
                                                              + [SPLIT_COLUMN]))

        # Проверка до запуска: колонки из конфига, которых нет в данных эксперимента.
        protected = set()
        for model in editor.models:
            protected |= {model.get('column_target'), model.get('column_exposure')} - {None}
        self.target_columns = protected
        missing = sorted(editor.referenced_columns() - set(data.columns))
        missing_targets = [c for c in missing if c in protected]
        if missing_targets:
            result.status = 'failed'
            result.errors.append(f'В данных нет целевых колонок: {missing_targets}')
            return editor
        for column in missing:
            if self.task.agent.get('drop_missing_features', True):
                editor.remove_feature(column)
                result.warnings.append(f'Колонка {column} отсутствует в данных — удалена из конфига до запуска')
        result.config_changes = list(editor.changes)
        return editor

    # ---------- запуск ----------
    def _run_once(self, config_path: Path, exp_dir: Path, attempt: int, extra_kwargs: dict) -> _Attempt:
        captured: List[str] = []
        sink_errors = logger.add(lambda m: captured.append(m.record['message']), level='ERROR')
        sink_file = logger.add(exp_dir / f'attempt{attempt}.log', level='DEBUG', encoding='utf-8')
        automl, exc_text = None, None
        try:
            kwargs = dict(self.task.automl.get('extra_kwargs') or {})
            kwargs.update(extra_kwargs)
            automl = self.automl_class(auto_ml_config=str(self.auto_ml_config),
                                       models_config=str(config_path),
                                       external_config=self.external_config,
                                       retro=bool(self.task.automl.get('retro', False)),
                                       hp_tune=bool(self.task.automl.get('hp_tune', False)),
                                       **kwargs)
            automl.update_models(send_mail=False)
            hook = self.task.automl.get('post_hook')
            if hook:
                try:
                    getattr(automl, hook)()
                except Exception as hook_exc:
                    logger.error(f'post_hook {hook}: {hook_exc}')
        except Exception as exc:
            exc_text = ''.join(traceback.format_exception(type(exc), exc, exc.__traceback__)[-6:])
            logger.error(f'AGENT||Исключение при запуске: {exc}')
        finally:
            logger.remove(sink_errors)
            logger.remove(sink_file)

        status = getattr(automl, 'status', {}) if automl is not None else {}
        stage_errors = [f'{stage}: {err}' for stage, err in (getattr(automl, 'errors', {}) or {}).items() if err]
        ok = exc_text is None and bool(status.get('Fitting'))
        errors = ([exc_text] if exc_text else []) + stage_errors + captured
        if ok:
            # Ошибки нефатальных этапов (MLflow, Grafana, письмо) — только предупреждения.
            return _Attempt(ok=True, errors=[], warnings=self._dedup(errors), automl=automl)
        if not errors:
            errors = [f'Обучение не завершено, статусы этапов: {status}']
        return _Attempt(ok=False, errors=self._dedup(errors)[-30:], warnings=[], automl=automl)

    @staticmethod
    def _dedup(items: List[str]) -> List[str]:
        return list(dict.fromkeys(str(i)[:2000] for i in items))

    # ---------- исправление ----------
    MISSING_COLUMN_PATTERNS = [
        r"\[([^\]]+)\] not in index",
        r"KeyError: ['\"]([^'\"]+)['\"]",
        r"^['\"]([A-Za-z0-9_]+)['\"]$",
        r"[Cc]olumn ['\"]?([A-Za-z0-9_]+)['\"]? (?:not found|does not exist|is missing)",
        r"ColumnNotFoundError: (?:unable to find column )?['\"]?([A-Za-z0-9_]+)",
    ]

    def _rule_patches(self, editor: ModelsConfigEditor, errors: List[str], columns: set) -> List[dict]:
        known = editor.referenced_columns() | set(editor.feature_names())
        found = set()
        for text in errors:
            for pattern in self.MISSING_COLUMN_PATTERNS:
                for match in re.finditer(pattern, text, re.MULTILINE):
                    found |= {n.strip(" '\"") for n in match.group(1).split(',')}
        candidates = sorted(c for c in found if c in known and c not in self.target_columns)
        # Колонка либо отсутствует в данных, либо стала константой/пустой в сегменте.
        return [{'op': 'remove_feature', 'model': '*', 'feature': c} for c in candidates]

    def _fix(self, editor: ModelsConfigEditor, name: str, attempt: int, errors: List[str],
             data: pd.DataFrame) -> str:
        patches = self._rule_patches(editor, errors, set(data.columns))
        if patches:
            try:
                applied = editor.apply(patches)
                logger.info(f'AGENT||Правила: {applied}')
                return 'patch'
            except PatchError as exc:
                logger.warning(f'AGENT||Правило не применилось: {exc}')

        prompt = prompts.fix_config_prompt(task=self.task.to_prompt_dict(), experiment=name, attempt=attempt,
                                           errors=errors, config_summary=editor.summary(),
                                           previous_changes=editor.changes,
                                           dataset_columns=list(map(str, data.columns)))
        try:
            answer = self.llm.ask_json('fix_config', prompts.SYSTEM_PROMPT, prompt,
                                       context={'errors': errors})
        except Exception as exc:
            logger.error(f'AGENT||LLM недоступна: {exc}')
            return 'abort' if self.task.agent.get('abort_on_llm_error', False) else 'skip'

        action = answer.get('action', 'abort')
        logger.info(f"AGENT||LLM: {action} — {answer.get('diagnosis')}")
        if action == 'patch':
            try:
                editor.apply(answer.get('patches') or [])
            except (PatchError, KeyError) as exc:
                logger.error(f'AGENT||Правка LLM отклонена: {exc}')
                return 'skip'
        if action == 'abort' and not self.task.agent.get('allow_llm_abort', False):
            return 'skip'  # по умолчанию ошибка одного эксперимента не останавливает серию
        return action if action in ('patch', 'retry', 'skip', 'abort') else 'skip'

    # ---------- результаты ----------
    def _collect(self, result: ExperimentResult, automl):
        result.automl = automl
        result.automl_status = dict(getattr(automl, 'status', {}))
        res = getattr(automl, 'automl_results', None)
        if res is None:
            return
        result.result_pickle = res.result_pickle_name
        result.new_features = res.new_features
        for model_name, model_result in (res.ds_manager_result or {}).items():
            try:
                result.test_metrics[model_name] = round_floats(model_result.metrics['test']['full'])
            except Exception as exc:
                result.warnings.append(f'Метрики {model_name} недоступны: {exc}')
        df = res.compare_metrics_df
        if isinstance(df, pd.DataFrame) and not df.empty:
            result.compare_metrics = [round_floats(r) for r in json.loads(df.to_json(orient='records', force_ascii=False))]

    def evaluate_vs_baseline(self, result: ExperimentResult, baseline: ExperimentResult):
        """Оценивает сегментную и исходную модели на одних и тех же тестовых строках сегмента.

        Обе оценки идут через ``model_predict`` сегментного automl: разбиение train/test строится
        с теми же separation/random_state по данным сегмента, поэтому метрики сопоставимы.
        """
        segment, reference = result.automl, baseline.automl
        if result.status != 'success' or segment is None or reference is None:
            return
        reference_results = getattr(reference, '_results', None) or {}
        for model_name in getattr(segment, '_results', None) or {}:
            if model_name not in reference_results:
                result.warnings.append(f'В baseline нет модели {model_name}')
                continue
            try:
                new = segment.model_predict(data=segment.dataset, model_name=model_name)
                ref = segment.model_predict(data=segment.dataset, model_name=model_name,
                                            model_result=reference_results[model_name])
                result.vs_baseline[model_name] = {'new': round_floats(new.metrics['test']['full']),
                                                  'reference': round_floats(ref.metrics['test']['full'])}
            except Exception as exc:
                result.warnings.append(f'Сравнение {model_name} с baseline не удалось: {exc}')
                logger.error(f'AGENT||Сравнение {result.name}/{model_name} с baseline: {exc}')
        self._save_result(result)

    def release(self, result: ExperimentResult):
        """Освобождает память от automl-объекта (датасеты КАСКО большие)."""
        result.automl = None

    @staticmethod
    def _save_result(result: ExperimentResult):
        if result.final_config:
            with open(Path(result.final_config).parent / 'result.json', 'w', encoding='utf-8') as f:
                json.dump(result.to_dict(), f, ensure_ascii=False, indent=2, default=str)

    @staticmethod
    def _finish(result: ExperimentResult, editor: ModelsConfigEditor, exp_dir: Path):
        result.config_changes = list(editor.changes)
        result.final_config = str(editor.save(exp_dir / 'models_config_final.json'))
        ExperimentRunner._save_result(result)


def metric_columns(result: ExperimentResult) -> List[str]:
    cols = []
    for row in result.compare_metrics:
        cols += [k for k, v in row.items() if k not in NON_METRIC_COLUMNS and isinstance(v, (int, float))]
    for metrics in result.test_metrics.values():
        cols += list(metrics)
    return list(dict.fromkeys(cols))
