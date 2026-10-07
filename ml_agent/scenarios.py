"""Сценарии экспериментов агента.

``factor_split`` — гипотеза «модели, обученные отдельно по группам фактора, лучше общих»:

1. датасет -> ``preprocess_hook`` -> единое разбиение train/test (``AGENT_IS_TEST``) на всех строках;
2. для каждого блока моделей (``[block.*]``: частота, тяжесть+тоталь, ...):
   общая модель на данных блока (``base_query`` блока) + модель на каждой группе
   ``segment_column`` (``segment_values``); каждая — ``update_models()`` с автоисправлением конфига;
   модель группы сравнивается с общей моделью блока на тех же тестовых строках (стат. метрики);
3. комбинации «общая / групповая» по блокам оцениваются бизнес-метрикой (фин. эффект) на тестовых
   строках группы в области ``[data] base_query`` (см. :mod:`ml_agent.combinations`).

``segment_split`` — прежнее имя сценария (один блок из ``[automl]``).
"""
import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from loguru import logger

from ml_agent.combinations import CombinationEvaluator
from ml_agent.config_editor import ModelsConfigEditor
from ml_agent.runner import COMMON, SPLIT_COLUMN, AgentAbort, ExperimentResult, ExperimentRunner
from ml_agent.task import TaskSpec
from ml_agent.utils import import_object

OTHER = '__OTHER__'


def load_dataset(task: TaskSpec) -> pd.DataFrame:
    """Датасет задания после ``preprocess_hook`` (без фильтров — они применяются по блокам)."""
    path = task.resolve(task.data['source_file'])
    df = pd.read_csv(path) if path.suffix.lower() == '.csv' else pd.read_parquet(path)
    logger.info(f'AGENT||Датасет {path}: {df.shape}')
    if task.data.get('preprocess_hook'):
        df = import_object(task.data['preprocess_hook'])(df)
    return df.reset_index(drop=True)


def apply_query(df: pd.DataFrame, query: str = None, label: str = '') -> pd.DataFrame:
    if not query:
        return df
    result = df.query(query)
    logger.info(f"AGENT||{label} '{query}': {result.shape}")
    return result


def assign_fixed_split(task: TaskSpec, df: pd.DataFrame) -> pd.DataFrame:
    """Назначает train/test один раз на всём датасете (колонка ``AGENT_IS_TEST``).

    Иначе random-разбиение строится заново в каждом эксперименте, тестовые строки группы попадают
    в train общей модели, а модели разных блоков (частота, тяжесть) видят разный тест.
    Для разбиения по периоду (kind=date) ничего не делаем: оно и так одинаково везде.
    """
    first_block = task.block_settings(task.block_names[0])
    separation = ModelsConfigEditor.from_file(task.resolve(first_block['models_config'])) \
        .data.get('data_config', {}).get('separation') or {}
    if not task.evaluation.get('fixed_split', True) or separation.get('kind', 'random') != 'random':
        return df
    rng = np.random.RandomState(separation.get('random_state', 42))
    df[SPLIT_COLUMN] = (rng.rand(len(df)) < float(separation.get('test_train_proportion', 0.2))).astype(int)
    logger.info(f'AGENT||Единое разбиение train/test: доля теста {df[SPLIT_COLUMN].mean():.3f}')
    return df


def _slug(value) -> str:
    return re.sub(r'[^0-9A-Za-zА-Яа-яЁё_-]+', '_', str(value)).strip('_')[:40] or 'empty'


def plan_segments(task: TaskSpec, df: pd.DataFrame) -> List[str]:
    column = task.data['segment_column']
    counts = df[column].astype(str).value_counts()
    logger.info(f'AGENT||Распределение {column}:\n{counts.to_string()}')
    values = task.segment_values
    if not values or values == ['auto']:
        values = list(counts.index)
    return [v for v in values if v in counts.index] + \
        ([OTHER] if task.data.get('group_small_segments') else [])


def group_masks(task: TaskSpec, df: pd.DataFrame) -> Dict[str, pd.Series]:
    """Группа -> маска строк. Группы меньше min_rows (в области base_query) уходят в __OTHER__ или пропускаются."""
    column = task.data['segment_column']
    if column not in df.columns:
        raise AgentAbort(f'В датасете нет поля {column}')
    keys = df[column].astype(str)
    scope = apply_query(df, task.data.get('base_query'), 'Область гипотезы')
    scope_keys = keys.loc[scope.index]
    min_rows = int(task.data.get('min_rows', 0))
    values = plan_segments(task, scope)
    small = [v for v in values if v != OTHER and (scope_keys == v).sum() < min_rows]
    masks = {}
    for value in values:
        mask = keys.isin(small) if value == OTHER else keys == value
        if value == OTHER:
            skip = not small or mask.loc[scope.index].sum() < max(min_rows, 1)
        else:
            skip = value in small
        if skip:
            logger.info(f'AGENT||Группа {value} пропущена: мало строк')
            continue
        masks[value] = mask
    return masks


def run_factor_split(task: TaskSpec, runner: ExperimentRunner) -> List[ExperimentResult]:
    df = assign_fixed_split(task, load_dataset(task))
    groups = group_masks(task, df)
    column = task.data['segment_column']
    compare_kwarg = task.evaluation.get('baseline_compare_kwarg')

    results: List[ExperimentResult] = []
    for block in task.block_names:
        settings = task.block_settings(block)
        block_df = apply_query(df, settings.get('base_query'), f'Блок {block}')
        common = runner.run(f'{block}__common', block_df.reset_index(drop=True), block=block, group=COMMON,
                            description=f'{block}: общая модель')
        results.append(common)
        extra_kwargs = {}
        if compare_kwarg and runner.pickle_path(common):
            extra_kwargs[compare_kwarg] = str(runner.pickle_path(common))

        for group, mask in groups.items():
            group_df = block_df.loc[mask.loc[block_df.index]].reset_index(drop=True)
            result = runner.run(f'{block}__{_slug(group)}', group_df, block=block, group=group,
                                description=f'{block}: {column} == {group}' if group != OTHER
                                else f'{block}: малые группы {column} вместе',
                                extra_kwargs=extra_kwargs)
            runner.evaluate_vs_baseline(result, common)
            runner.release(result)
            results.append(result)
        runner.release(common)

    evaluator = CombinationEvaluator(task, runner)
    if evaluator.enabled:
        scope = apply_query(df, task.data.get('base_query'), 'Область оценки')
        runner.combinations = evaluator.evaluate(scope, {g: m.loc[scope.index] for g, m in groups.items()},
                                                 results)
        runner.metric_keys = sorted(evaluator.metric_keys)
    return results


SCENARIOS = {'factor_split': run_factor_split,
             'segment_split': run_factor_split}
