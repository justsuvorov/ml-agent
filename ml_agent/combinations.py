"""Комбинации моделей блоков и их оценка бизнес-метрикой (фин. эффект).

Для каждой группы фактора перебираются все варианты «общая модель / модель группы» по каждому
блоку (2^число блоков). Из пиклов выбранных моделей собирается единый пикл
(например частота + тяжесть + тоталь — как ``CascoRelease.pickle_models``) и сравнивается
с эталоном «все модели общие» на тестовых строках группы — как ``compare_models``:
``compare_pickle_models(data, first_model, second_model, business_metric, ...)``.

Настройки ``[evaluation]``::

    business_metric = business_metric:FinEffectMetric   ; класс метрики (параметры по умолчанию)
    business_metric_kwargs = {}                          ; переопределение параметров при необходимости
    compare_function = mldataworker.automl:compare_pickle_models
    extra_columns = GLM_INTERVAL_PREMIUM, SUM_INSURED    ; нужны метрике (премия, страховая сумма)
    models_order = frequency, severity, total            ; порядок моделей в собранном пикле
    no_exposure_models = severity                        ; как в CompareCascoModels: тяжесть без экспозиции
    query = POLICY_PRODUCT == 'КЛАССИКА'                 ; доп. фильтр строк оценки (по умолчанию нет)
"""
import copy
import itertools
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from loguru import logger

from ml_agent.runner import COMMON, SPLIT_COLUMN, ExperimentResult, ExperimentRunner
from ml_agent.task import TaskSpec, _as_list
from ml_agent.utils import import_object, round_floats

REFERENCE = 'все общие'


def combination_label(choice: Dict[str, str]) -> str:
    """{'frequency': 'group', 'severity_total': 'common'} -> 'frequency: группа, severity_total: общая'."""
    return ', '.join(f"{block}: {'группа' if kind == 'group' else 'общая'}" for block, kind in choice.items())


class CombinationEvaluator:
    def __init__(self, task: TaskSpec, runner: ExperimentRunner):
        self.task = task
        self.runner = runner
        settings = task.evaluation
        self.metric_class = settings.get('business_metric')
        self.metric_kwargs = settings.get('business_metric_kwargs') or {}
        self.compare_function = settings.get('compare_function', 'mldataworker.automl:compare_pickle_models')
        self.extra_columns = _as_list(settings.get('extra_columns')) or None
        self.models_order = _as_list(settings.get('models_order'))
        self.no_exposure_models = set(_as_list(settings.get('no_exposure_models')))
        self.query = settings.get('query')
        self.metric_keys: set = set()  # имена значений, которые вернула бизнес-метрика (например Margin)

    @property
    def enabled(self) -> bool:
        return bool(self.metric_class)

    def evaluate(self, data: pd.DataFrame, groups: Dict[str, pd.Series],
                 experiments: List[ExperimentResult]) -> List[dict]:
        """:param data: область оценки (после base_query задания); :param groups: группа -> маска по data."""
        blocks = self.task.block_names
        by_key = {(e.block, e.group): e for e in experiments}
        reference_parts = {b: by_key.get((b, COMMON)) for b in blocks}
        if any(self.runner.pickle_path(e) is None for e in reference_parts.values()):
            logger.error('AGENT||Фин. эффект: не все общие модели обучены — комбинации не оцениваются')
            return []
        reference = self._assemble(reference_parts)

        rows = []
        for group, mask in groups.items():
            group_data = data.loc[mask]
            if SPLIT_COLUMN in group_data.columns:
                group_data = group_data.loc[group_data[SPLIT_COLUMN] == 1]
            if self.query:
                group_data = group_data.query(self.query)
            for kinds in itertools.product(('common', 'group'), repeat=len(blocks)):
                if 'group' not in kinds:
                    continue  # эталон
                choice = dict(zip(blocks, kinds))
                row = {'group': group, 'combination': combination_label(choice), 'choice': choice,
                       'reference': REFERENCE, 'n_rows_test': len(group_data)}
                if 'GLM_INTERVAL_PREMIUM' in group_data.columns:
                    row['premium_test'] = round(float(group_data['GLM_INTERVAL_PREMIUM'].sum()))
                parts = {b: by_key.get((b, group if k == 'group' else COMMON)) for b, k in choice.items()}
                missing = [b for b, e in parts.items() if self.runner.pickle_path(e) is None]
                if missing:
                    row['error'] = f'нет обученной модели блоков {missing} для группы {group}'
                    rows.append(row)
                    continue
                try:
                    row.update(round_floats(self._compare(group_data, self._assemble(parts), reference)))
                except Exception as exc:
                    row['error'] = str(exc)[:500]
                    logger.error(f"AGENT||Фин. эффект {group} / {row['combination']}: {exc}")
                rows.append(row)
                logger.info(f"AGENT||Фин. эффект {group} / {row['combination']}: "
                            f"{ {k: v for k, v in row.items() if k in self.metric_keys} or row.get('error')}")
        return rows

    def _assemble(self, parts: Dict[str, ExperimentResult]) -> list:
        """Пикл комбинации: модели блоков в порядке models_order (как CascoRelease)."""
        models = []
        for experiment in parts.values():
            models += copy.deepcopy(pd.read_pickle(self.runner.pickle_path(experiment)))
        if self.models_order:
            position = {name: i for i, name in enumerate(self.models_order)}
            models.sort(key=lambda m: position.get(m['model_config']['name'], len(position)))
        for model in models:
            if model['model_config']['name'] in self.no_exposure_models:
                model['model_config']['column_exposure'] = None
        return models

    def _compare(self, data: pd.DataFrame, first_model: list, second_model: list) -> dict:
        compare = import_object(self.compare_function)
        metric = import_object(self.metric_class)(**self.metric_kwargs)
        result = compare(data=data, first_model=first_model, second_model=second_model, business_metric=metric,
                         extra_columns_names=self.extra_columns, use_exposure=True,
                         external_config=self.runner.external_config)
        values = {k: float(v) for k, v in (result.get('business_metric_value') or {}).items()}
        self.metric_keys |= set(values)
        return values


def summarize(rows: List[dict]) -> pd.DataFrame:
    """Сводная: группа × комбинация -> значения метрики (+ сумма по группам)."""
    ok = [r for r in rows if 'error' not in r]
    if not ok:
        return pd.DataFrame()
    df = pd.DataFrame(ok).drop(columns=['choice', 'reference'], errors='ignore')
    value_columns = [c for c in df.columns if c not in ('group', 'combination', 'n_rows_test', 'premium_test')]
    total = df.groupby('combination', sort=False)[value_columns + ['n_rows_test']].sum(min_count=1).reset_index()
    total['group'] = 'ИТОГО (сумма по группам)'
    return pd.concat([df, total], ignore_index=True)
