"""Сравнение экспериментов, выводы LLM и итоговое письмо."""
import html
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union

import pandas as pd
from loguru import logger

from ml_agent import prompts
from ml_agent.llm import LLMClient
from ml_agent.combinations import summarize
from ml_agent.runner import COMMON, ExperimentResult, metric_columns
from ml_agent.task import TaskSpec
from ml_agent.utils import import_object

MAXIMIZE_HINTS = ('gini', 'auc', 'r2', 'accuracy', 'f1', 'precision', 'recall', 'lift', 'effect', 'эффект')
VERDICTS = {'better': 'лучше', 'worse': 'хуже', 'neutral': 'без изменений', 'unknown': 'нет данных'}

Section = Tuple[str, Union[str, pd.DataFrame]]


def metric_direction(task: TaskSpec, metric: str) -> str:
    direction = task.evaluation.get('direction')
    if direction in ('maximize', 'minimize'):
        return direction
    return 'maximize' if any(h in metric.lower() for h in MAXIMIZE_HINTS) else 'minimize'


def build_comparison(task: TaskSpec, experiments: List[ExperimentResult]) -> Tuple[str, List[dict]]:
    success = [e for e in experiments if e.status == 'success']
    metric = task.evaluation.get('main_metric')
    if not metric:
        candidates = [m for e in success for m in metric_columns(e)]
        metric = candidates[0] if candidates else None
        logger.warning(f'AGENT||main_metric не задан, используется {metric}')
    if metric is None:
        return '', []
    direction = metric_direction(task, metric)
    source = task.evaluation.get('compare_with', 'baseline')

    rows = []
    for exp in success:
        models = list(dict.fromkeys(list(exp.test_metrics) + [r.get('Имя модели') for r in exp.compare_metrics]))
        for model in filter(None, models):
            new, ref = exp.new_metric(model, metric, source), exp.reference_metric(model, metric, source)
            reference = 'исходная модель на тесте сегмента' if model in exp.vs_baseline else                 ('эталон automl' if ref is not None else '-')
            improvement = rel = None
            if new is not None and ref is not None:
                improvement = round((new - ref) if direction == 'maximize' else (ref - new), 6)
                rel = round(100 * improvement / abs(ref), 3) if ref else None
            rows.append({'experiment': exp.name, 'block': exp.block, 'group': exp.group,
                         'segment': exp.description, 'n_rows': exp.n_rows,
                         'model': model, 'metric': metric, 'direction': direction,
                         'reference': reference, 'new_value': new, 'reference_value': ref,
                         'improvement': improvement, 'improvement_pct': rel})
    return metric, rows


def run_compare_hook(task: TaskSpec, experiments: List[ExperimentResult]) -> Optional[pd.DataFrame]:
    """Пользовательское сравнение (например, compare_models по пиклам в casco)."""
    hook = task.evaluation.get('compare_hook')
    if not hook:
        return None
    try:
        df = import_object(hook)(experiments=experiments, task=task)
        return df if isinstance(df, pd.DataFrame) else pd.DataFrame(df)
    except Exception as exc:
        logger.error(f'AGENT||compare_hook {hook}: {exc}')
        return None


def business_key(task: TaskSpec, metric_keys: List[str]) -> Optional[str]:
    """Основное значение бизнес-метрики (``[evaluation] business_metric_key``, по умолчанию первое — Margin)."""
    return task.evaluation.get('business_metric_key') or (metric_keys[0] if metric_keys else None)


def ask_conclusions(task: TaskSpec, llm: LLMClient, metric: str, comparison: List[dict],
                    experiments: List[ExperimentResult], extra: Optional[pd.DataFrame],
                    business: List[dict] = None, key: str = None) -> dict:
    details = [{k: v for k, v in e.to_dict().items() if k not in ('final_config',)} for e in experiments]
    if extra is not None:
        details.append({'custom_comparison': json.loads(extra.to_json(orient='records', force_ascii=False))})
    business = [{k: v for k, v in r.items() if k != 'choice'} for r in business or []]
    prompt = prompts.report_prompt(task=task.to_prompt_dict(), instructions=task.instructions,
                                   business=business, business_key=key, comparison=comparison,
                                   experiments=details)
    try:
        return llm.ask_json('report', prompts.SYSTEM_PROMPT, prompt,
                            context={'comparison': comparison, 'business': business, 'business_key': key,
                                     'min_improvement': task.evaluation.get('min_improvement', 0)})
    except Exception as exc:
        logger.error(f'AGENT||LLM не сформировала выводы: {exc}')
        return {'summary': f'Выводы LLM недоступны: {exc}', 'segments': [], 'recommendation': '-',
                'risks': [], 'next_steps': []}


def fin_effect_table(business: List[dict], key: str) -> pd.DataFrame:
    """Группа × комбинация -> значение бизнес-метрики относительно эталона «все общие»."""
    summary = summarize(business)
    if summary.empty or key not in summary.columns:
        return pd.DataFrame()
    table = summary.pivot_table(index='group', columns='combination', values=key, aggfunc='first', sort=False)
    table.columns.name = None
    return table.reset_index().rename(columns={'group': 'Группа'})


def build_sections(task: TaskSpec, metric: str, comparison: List[dict], conclusions: dict,
                   experiments: List[ExperimentResult], extra: Optional[pd.DataFrame],
                   business: List[dict] = None, key: str = None) -> List[Section]:
    sections: List[Section] = [
        ('Задача', task.description or task.name),
        ('Вывод', conclusions.get('summary', '')),
        ('Рекомендация', conclusions.get('recommendation', '')),
    ]
    fin_effect = fin_effect_table(business or [], key)
    if not fin_effect.empty:
        sections.append((f'{key}: комбинации моделей против «все общие» (тест группы; > 0 — лучше эталона)',
                         fin_effect))
    errors = [r for r in business or [] if 'error' in r]
    if errors:
        sections.append(('Комбинации, которые не удалось оценить', pd.DataFrame(errors)[
            ['group', 'combination', 'error']]))
    verdicts = conclusions.get('groups') or conclusions.get('segments')
    if verdicts:
        df = pd.DataFrame(verdicts)
        if 'verdict' in df:
            df['verdict'] = df['verdict'].map(lambda v: VERDICTS.get(v, v))
        sections.append(('Оценка по группам', df))
    if comparison:
        df = pd.DataFrame(comparison)[['block', 'group', 'n_rows', 'model', 'reference', 'new_value',
                                       'reference_value', 'improvement', 'improvement_pct']]
        df['group'] = df['group'].replace({COMMON: 'все данные'})
        df.columns = ['Блок', 'Группа', 'Строк', 'Модель', 'Эталон', f'{metric}: новая',
                      f'{metric}: эталон', 'Прирост', 'Прирост, %']
        sections.append((f'Статистические метрики по моделям: {metric} (тест, {comparison[0]["direction"]})', df))
    if extra is not None and not extra.empty:
        sections.append(('Дополнительное сравнение', extra))
    sections.append(('Статус экспериментов', pd.DataFrame([{
        'Эксперимент': e.name, 'Блок': e.block, 'Описание': e.description, 'Строк': e.n_rows, 'Статус': e.status,
        'Попыток': e.attempts, 'Правки конфига': '; '.join(e.config_changes) or '-',
        'Ошибки': ' | '.join(x.splitlines()[-1] for x in e.errors if x.strip())[:500] or '-',
        'Пикл': e.result_pickle or '-'} for e in experiments])))
    for title, key in (('Риски и ограничения', 'risks'), ('Следующие шаги', 'next_steps')):
        if conclusions.get(key):
            sections.append((title, '\n'.join(f'• {x}' for x in conclusions[key])))
    return sections


def render_html(title: str, sections: List[Section]) -> str:
    parts = [f'<html><head><meta charset="utf-8"><title>{html.escape(title)}</title>'
             '<style>body{font-family:sans-serif;max-width:1200px;margin:24px auto;padding:0 16px}'
             'table{border-collapse:collapse;font-size:13px}td,th{border:1px solid #ccc;padding:4px 8px}'
             'th{background:#1f3864;color:#fff}</style></head><body>', f'<h1>{html.escape(title)}</h1>']
    for name, body in sections:
        parts.append(f'<h2>{html.escape(name)}</h2>')
        if isinstance(body, pd.DataFrame):
            parts.append(body.to_html(index=False, na_rep='-'))
        else:
            parts.append('<p>' + html.escape(str(body)).replace('\n', '<br>') + '</p>')
    parts.append('</body></html>')
    return '\n'.join(parts)


@dataclass
class AgentReport:
    """Итог работы агента: таблицы для ноутбука + пути к сохранённым отчётам."""
    task: str
    subject: str
    metric: str
    comparison: pd.DataFrame
    conclusions: dict
    experiments: List[ExperimentResult]
    fin_effect: pd.DataFrame = field(default_factory=pd.DataFrame)  # группа × комбинация -> бизнес-метрика
    combinations: List[dict] = field(default_factory=list)  # все оценки комбинаций (в т.ч. с ошибками)
    sections: List[Section] = field(repr=False, default_factory=list)
    run_dir: Optional[Path] = None
    email_sent: bool = False

    @property
    def recommendation(self) -> str:
        return self.conclusions.get('recommendation', '')

    @property
    def status(self) -> pd.DataFrame:
        return pd.DataFrame([{'experiment': e.name, 'block': e.block, 'group': e.group, 'n_rows': e.n_rows,
                              'status': e.status, 'attempts': e.attempts,
                              'config_changes': len(e.config_changes)} for e in self.experiments])

    @property
    def html(self) -> str:
        return render_html(self.subject, self.sections)

    def _repr_html_(self) -> str:  # отображение в Jupyter
        return self.html

    def to_dict(self) -> dict:
        return {'task': self.task, 'metric': self.metric,
                'comparison': self.comparison.to_dict(orient='records'),
                'fin_effect': self.fin_effect.to_dict(orient='records'),
                'combinations': [{k: v for k, v in r.items() if k != 'choice'} for r in self.combinations],
                'conclusions': self.conclusions, 'experiments': [e.to_dict() for e in self.experiments],
                'run_dir': str(self.run_dir), 'email_sent': self.email_sent}

    def save(self, run_dir: Path):
        self.run_dir = run_dir
        with open(run_dir / 'report.json', 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2, default=str)
        (run_dir / 'report.html').write_text(self.html, encoding='utf-8')
        logger.info(f'AGENT||Отчёт: {run_dir / "report.html"}')


def build_report(task: TaskSpec, llm: LLMClient, experiments: List[ExperimentResult],
                 combinations: List[dict] = None, metric_keys: List[str] = None) -> AgentReport:
    metric, comparison = build_comparison(task, experiments)
    extra = run_compare_hook(task, experiments)
    key = business_key(task, metric_keys or [])
    conclusions = ask_conclusions(task, llm, metric, comparison, experiments, extra, combinations, key)
    subject = task.email.get('subject') or f'ML-агент: {task.name}'
    return AgentReport(task=task.name, subject=subject, metric=metric, comparison=pd.DataFrame(comparison),
                       conclusions=conclusions, experiments=experiments,
                       fin_effect=fin_effect_table(combinations or [], key), combinations=combinations or [],
                       sections=build_sections(task, metric, comparison, conclusions, experiments, extra,
                                               combinations, key))


def send_email(report: AgentReport, config, receivers: List[str] = None,
               email_class: str = 'mldataworker.core.email:EMail'):
    """Отправляет отчёт письмом через EMail из mldataworker (SMTP-настройки — из config.py)."""
    receivers = receivers or list(getattr(config, 'email_receivers', []) or [])
    if not receivers:
        raise ValueError('Не заданы получатели: [email] receivers в задании или email_receivers в config.py')
    config.email_receivers = receivers
    email = import_object(email_class)(config=config)
    email.header(report.subject)
    for name, body in report.sections:
        email.mail.add_text(name, properties=['bold'], n_line_breaks=1)
        if isinstance(body, pd.DataFrame):
            email.mail.add_pandas_table(body.fillna('-'), params=dict(text_align='left', font_family='sans-serif'))
            email.mail.add_line_breaks(1)
        else:
            email.mail.add_text(str(body).replace('\n', '<br>'), n_line_breaks=2)
    email.send()
    report.email_sent = True
    logger.info(f'AGENT||Письмо отправлено: {receivers}')
