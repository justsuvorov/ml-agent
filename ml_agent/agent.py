"""Оркестратор агента: задание -> эксперименты -> сравнение -> выводы LLM -> письмо.

Пример (Jupyter, рядом с config.py проекта — как в main_fit)::

    import config
    from ml_agent import MLAgent

    agent = MLAgent('tasks/business_type_split.txt', config=config)
    agent.dry_run()
    report = agent.run()
    report  # HTML-отчёт в ячейке
"""
import shutil
import sys
from datetime import datetime
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Optional, Union

import pandas as pd
from loguru import logger

from ml_agent.config_editor import ModelsConfigEditor
from ml_agent.llm import build_llm_client
from ml_agent.report import AgentReport, build_report, send_email
from ml_agent.runner import AgentAbort, ExperimentRunner
from ml_agent.scenarios import OTHER, SCENARIOS, load_dataset, plan_segments
from ml_agent.task import TaskSpec, load_task

AGENT_LOG_PREFIXES = ('AGENT', 'LLM', 'VSK', 'Qwen')


class MLAgent:
    """ML-агент поверх mldataworker.

    :param task: путь к txt-заданию или уже загруженный :class:`TaskSpec`.
    :param config: модуль ``config`` проекта (как ``external_config`` в main_fit): пути, почта, настройки LLM.
    :param project_dir: корень проекта — от него считаются пути задания и импортируются
        ``casco_automl`` и т.п. По умолчанию — текущий каталог.
    :param llm_model_type: переопределить ``config.llm_model_type`` (например ``'offline'`` для отладки).
    """

    def __init__(self, task: Union[str, Path, TaskSpec], config: ModuleType,
                 project_dir: Union[str, Path] = None, llm_model_type: str = None):
        self.task = task if isinstance(task, TaskSpec) else load_task(task)
        self.config = config
        self.task.agent['base_dir'] = str(Path(project_dir or Path.cwd()).resolve())
        if llm_model_type:
            self.task.llm['model_type'] = llm_model_type
        if self.task.agent['base_dir'] not in sys.path:
            sys.path.insert(0, self.task.agent['base_dir'])
        self.run_dir: Optional[Path] = None
        self.report: Optional[AgentReport] = None

    # ---------- проверка ----------
    def dry_run(self) -> dict:
        """Проверка без обучения: размер данных, план сегментов, колонки конфига, которых нет в данных."""
        df = load_dataset(self.task)
        editor = ModelsConfigEditor.from_file(self.task.resolve(self.task.automl['models_config']))
        column = self.task.data['segment_column']
        min_rows = int(self.task.data.get('min_rows', 0))
        counts = df[column].astype(str).value_counts()
        segments = pd.DataFrame([{'segment': v, 'n_rows': int(counts.get(v, 0))}
                                 for v in plan_segments(self.task, df) if v != OTHER])
        if not segments.empty:
            small = segments['n_rows'] < min_rows
            segments['action'] = 'обучить'
            segments.loc[small, 'action'] = 'в __OTHER__' if self.task.data.get('group_small_segments') \
                else 'пропустить'
        missing = sorted(editor.referenced_columns() - set(df.columns))
        result = {'rows': len(df), 'models': editor.model_names(), 'n_features': len(editor.feature_names()),
                  'missing_columns': missing, 'segments': segments}
        logger.info(f"AGENT||dry-run: строк {len(df)}, модели {result['models']}, "
                    f"нет в данных: {missing or 'нет'}")
        return result

    # ---------- запуск ----------
    def run(self, send_mail: bool = None) -> AgentReport:
        """Полный цикл. ``send_mail`` переопределяет ``[email] send`` задания."""
        self.run_dir = self.task.resolve(self.task.agent.get('workdir', 'agent_runs')) / \
            f'{datetime.now():%Y%m%d_%H%M%S}_{self.task.name}'
        self.run_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(self.task.path, self.run_dir / self.task.path.name)
        log_id = logger.add(self.run_dir / 'agent.log', level='INFO', encoding='utf-8',
                            filter=lambda r: r['message'].startswith(AGENT_LOG_PREFIXES))
        try:
            logger.info(f'AGENT||Задание {self.task.name}, сценарий {self.task.scenario}, директория {self.run_dir}')
            llm = build_llm_client(self.config, overrides=self.task.llm, log_path=self.run_dir / 'llm_log.jsonl')
            run_config = self._run_config()
            runner = ExperimentRunner(self.task, llm, self.run_dir, run_config)
            try:
                SCENARIOS[self.task.scenario](self.task, runner)
            except AgentAbort as exc:
                logger.error(f'AGENT||Остановлен: {exc}')
            if not runner.results:
                raise AgentAbort('Ни одного эксперимента не выполнено — см. agent.log')

            self.report = build_report(self.task, llm, runner.results)
            if self.task.email.get('send', False) if send_mail is None else send_mail:
                try:
                    send_email(self.report, run_config, receivers=self.task.email_receivers,
                               email_class=self.task.email.get('class', 'mldataworker.core.email:EMail'))
                except Exception as exc:
                    logger.error(f'AGENT||Письмо не отправлено: {exc}')
            self.report.save(self.run_dir)
            logger.info(f'AGENT||Рекомендация: {self.report.recommendation}')
            return self.report
        finally:
            logger.remove(log_id)

    def send(self, receivers: list = None) -> None:
        """Отправить последний отчёт письмом (например, после просмотра в ноутбуке)."""
        if self.report is None:
            raise RuntimeError('Сначала выполните run()')
        send_email(self.report, self._run_config(isolate=False), receivers=receivers or self.task.email_receivers,
                   email_class=self.task.email.get('class', 'mldataworker.core.email:EMail'))
        self.report.save(self.run_dir)

    def _run_config(self, isolate: bool = True) -> SimpleNamespace:
        """Копия config.py с переопределениями из ``[external_config]``; результаты — в папку запуска.

        Исходный модуль config не изменяется (он может использоваться другими скриптами в том же ядре).
        """
        values = {k: getattr(self.config, k) for k in dir(self.config) if not k.startswith('__')}
        values = {k: v for k, v in values.items() if not isinstance(v, ModuleType)}
        values.update(self.task.external_config)
        if isolate and self.task.agent.get('isolate_results', True) and self.run_dir is not None:
            values['results_path'] = self.run_dir / 'results'
        if values.get('results_path') is not None:
            Path(values['results_path']).mkdir(parents=True, exist_ok=True)
        return SimpleNamespace(**values)
