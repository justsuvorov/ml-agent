"""Сценарии экспериментов агента.

``segment_split`` — проверка гипотезы «отдельные модели по значениям поля лучше единой»:

1. ``baseline`` — обучение на всём датасете (исходная постановка);
2. для каждого значения ``segment_column`` — датасет фильтруется и запускается та же
   процедура ``update_models()``; модель сравнивается с эталоном на тесте того же сегмента.

Эталон (``[evaluation] compare_with``):

* ``baseline`` (по умолчанию) — исходная модель из шага 1; агент сам оценивает её и сегментную
  модель на одних и тех же тестовых строках сегмента (``model_predict``);
* ``automl`` — то, с чем сравнивает сам automl-класс (prod-пикл, ``model_to_compare`` и т.п.).

Дополнительно ``baseline_compare_kwarg`` передаёт путь к пиклу baseline в automl-класс
(например ``model_to_compare`` у CascoFLAutoML), чтобы его штатное сравнение тоже шло с baseline.
"""
import re
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from loguru import logger

from ml_agent.config_editor import ModelsConfigEditor
from ml_agent.runner import SPLIT_COLUMN, AgentAbort, ExperimentResult, ExperimentRunner
from ml_agent.task import TaskSpec
from ml_agent.utils import import_object

OTHER = '__OTHER__'


def load_dataset(task: TaskSpec) -> pd.DataFrame:
    path = task.resolve(task.data['source_file'])
    if path.suffix.lower() == '.csv':
        df = pd.read_csv(path)
    else:
        df = pd.read_parquet(path)
    logger.info(f'AGENT||Датасет {path}: {df.shape}')
    if task.data.get('preprocess_hook'):
        df = import_object(task.data['preprocess_hook'])(df)
    if task.data.get('base_query'):
        df = df.query(task.data['base_query'])
        logger.info(f"AGENT||После base_query '{task.data['base_query']}': {df.shape}")
    return df.reset_index(drop=True)


def assign_fixed_split(task: TaskSpec, df: pd.DataFrame) -> pd.DataFrame:
    """Назначает train/test один раз на всём датасете (колонка ``AGENT_IS_TEST``).

    Иначе random-разбиение строится заново внутри каждого эксперимента, и тестовые строки
    сегмента попадают в train исходной модели — сравнение с ней было бы завышено в её пользу.
    Для разбиения по периоду (kind=date) ничего не делаем: оно и так одинаково во всех экспериментах.
    """
    separation = ModelsConfigEditor.from_file(task.resolve(task.automl['models_config']))         .data.get('data_config', {}).get('separation') or {}
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


def run_segment_split(task: TaskSpec, runner: ExperimentRunner) -> List[ExperimentResult]:
    df = assign_fixed_split(task, load_dataset(task))
    column = task.data['segment_column']
    if column not in df.columns:
        raise AgentAbort(f'В датасете нет поля {column}')
    min_rows = int(task.data.get('min_rows', 0))
    keys = df[column].astype(str)

    results: List[ExperimentResult] = []
    baseline = None
    if task.data.get('include_baseline', True):
        baseline = runner.run('baseline', df, description='Все данные, исходная постановка')
        results.append(baseline)

    compare_with = task.evaluation.get('compare_with', 'baseline')
    if compare_with == 'baseline' and (baseline is None or baseline.status != 'success'):
        raise AgentAbort('compare_with=baseline, но исходная модель не обучена — сравнивать не с чем')
    extra_kwargs = {}
    compare_kwarg = task.evaluation.get('baseline_compare_kwarg')
    if compare_kwarg and baseline is not None and baseline.result_pickle:
        extra_kwargs[compare_kwarg] = str(Path(runner.external_config.results_path) / baseline.result_pickle)

    segments = plan_segments(task, df)
    small = [v for v in segments if v != OTHER and (keys == v).sum() < min_rows]
    for value in segments:
        if value == OTHER:
            mask = keys.isin(small)
            description = f'{column} in {small} (малые сегменты вместе)'
        else:
            mask = keys == value
            description = f'{column} == {value}'
        name = f'segment_{_slug(value)}'
        n_rows = int(mask.sum())
        if value in small or (value == OTHER and n_rows < max(min_rows, 1)):
            skipped = ExperimentResult(name=name, description=description, n_rows=n_rows, status='skipped',
                                       warnings=[f'Строк {n_rows} < min_rows={min_rows}'])
            runner.results.append(skipped)
            results.append(skipped)
            logger.info(f'AGENT||{name} пропущен: {n_rows} строк')
            continue
        result = runner.run(name, df.loc[mask].reset_index(drop=True),
                            description=description, extra_kwargs=extra_kwargs)
        if compare_with == 'baseline':
            runner.evaluate_vs_baseline(result, baseline)
        runner.release(result)
        results.append(result)
    if baseline is not None:
        runner.release(baseline)
    return results


SCENARIOS = {'segment_split': run_segment_split}
