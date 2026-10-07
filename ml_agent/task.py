"""Разбор txt-файла с заданием для агента.

Блоки моделей — секции ``[block.<имя>]`` (например ``[block.frequency]``, ``[block.severity_total]``):
каждый блок — отдельный конфиг моделей и отдельный запуск ``update_models()``. Ключи блока
переопределяют общие ключи ``[automl]``; известные ключи (models_config, base_query, class,
auto_ml_config, retro, hp_tune, post_hook, extra_kwargs) — настройки запуска, остальные
(например ``model_to_compare``) передаются в конструктор automl-класса.

Формат — INI-секции (key = value) + свободный текст. Всё, что идёт после заголовка
``[instructions]`` до конца файла, считается свободной инструкцией и передаётся в LLM
как есть (без требований к отступам). Комментарии — строки, начинающиеся с ``#`` или ``;``.
"""
import configparser
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_BLOCK = 'main'  # задание без [block.*] — один блок из [automl]
BLOCK_KEYS = {'models_config', 'base_query', 'class', 'auto_ml_config', 'retro', 'hp_tune', 'post_hook',
              'extra_kwargs'}
INSTRUCTIONS_HEADER = re.compile(r'^\[instructions\]\s*$', re.IGNORECASE | re.MULTILINE)


def _to_value(raw: str) -> Any:
    """Преобразует строковое значение из txt в python-тип (bool/None/число/JSON/строка)."""
    value = raw.strip()
    low = value.lower()
    if low in ('true', 'yes', 'on'):
        return True
    if low in ('false', 'no', 'off'):
        return False
    if low in ('none', 'null', ''):
        return None
    if value[:1] in ('[', '{'):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            pass
    return value


def _as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    return [v.strip() for v in str(value).split(',') if v.strip()]


@dataclass
class TaskSpec:
    """Задание для агента, прочитанное из txt."""
    path: Path
    name: str
    scenario: str
    description: str = ''
    instructions: str = ''
    data: Dict[str, Any] = field(default_factory=dict)
    automl: Dict[str, Any] = field(default_factory=dict)
    evaluation: Dict[str, Any] = field(default_factory=dict)
    agent: Dict[str, Any] = field(default_factory=dict)
    llm: Dict[str, Any] = field(default_factory=dict)
    external_config: Dict[str, Any] = field(default_factory=dict)
    email: Dict[str, Any] = field(default_factory=dict)
    blocks: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    @property
    def block_names(self) -> List[str]:
        return list(self.blocks) or [DEFAULT_BLOCK]

    def block_settings(self, name: str) -> Dict[str, Any]:
        """Настройки запуска блока: ``[automl]`` + ``[block.<name>]``; extra_kwargs объединяются."""
        block = self.blocks.get(name, {})
        settings = {k: v for k, v in self.automl.items() if k != 'extra_kwargs'}
        settings.update({k: v for k, v in block.items() if k in BLOCK_KEYS and k != 'extra_kwargs'})
        settings['extra_kwargs'] = {**(self.automl.get('extra_kwargs') or {}),
                                    **(block.get('extra_kwargs') or {}),
                                    **{k: v for k, v in block.items() if k not in BLOCK_KEYS}}
        if 'base_query' not in block:  # ключ блока есть, но пустой — без фильтра
            settings['base_query'] = self.data.get('base_query')
        if not settings.get('models_config'):
            raise ValueError(f'Блок {name}: не задан models_config')
        return settings

    @property
    def base_dir(self) -> Path:
        """Корень проекта: от него разрешаются пути задания (задаётся MLAgent(project_dir=...))."""
        return Path(self.agent.get('base_dir') or Path.cwd()).resolve()

    def resolve(self, path: Optional[str]) -> Optional[Path]:
        if path is None:
            return None
        p = Path(path)
        return p if p.is_absolute() else (self.base_dir / p).resolve()

    @property
    def max_fix_attempts(self) -> int:
        return int(self.agent.get('max_fix_attempts', 3))

    @property
    def segment_values(self) -> List[str]:
        return _as_list(self.data.get('segment_values'))

    @property
    def email_receivers(self) -> List[str]:
        return _as_list(self.email.get('receivers'))

    def to_prompt_dict(self) -> dict:
        """Краткое описание задания для промпта LLM (без секретов)."""
        return {'name': self.name,
                'scenario': self.scenario,
                'description': self.description,
                'data': self.data,
                'evaluation': self.evaluation,
                'automl': {k: v for k, v in self.automl.items() if k != 'extra_kwargs'},
                'blocks': {name: {k: v for k, v in block.items() if k != 'extra_kwargs'}
                           for name, block in self.blocks.items()}}


def load_task(path: str) -> TaskSpec:
    path = Path(path).resolve()
    text = path.read_text(encoding='utf-8')

    instructions = ''
    match = INSTRUCTIONS_HEADER.search(text)
    if match:
        instructions = text[match.end():].strip()
        text = text[:match.start()]

    parser = configparser.ConfigParser(interpolation=None, inline_comment_prefixes=(' #', ' ;'))
    parser.optionxform = str  # сохраняем регистр ключей (важно для external_config)
    parser.read_string(text)

    sections = {s: {k: _to_value(v) for k, v in parser.items(s)} for s in parser.sections()}
    task_section = sections.get('task', {})
    if 'name' not in task_section:
        raise ValueError(f'{path}: в секции [task] не задан name')

    return TaskSpec(path=path,
                    name=str(task_section['name']),
                    scenario=str(task_section.get('scenario', 'factor_split')),
                    description=str(task_section.get('description') or ''),
                    instructions=instructions,
                    data=sections.get('data', {}),
                    automl=sections.get('automl', {}),
                    evaluation=sections.get('evaluation', {}),
                    agent=sections.get('agent', {}),
                    llm=sections.get('llm', {}),
                    external_config=sections.get('external_config', {}),
                    email=sections.get('email', {}),
                    blocks={s.split('.', 1)[1]: v for s, v in sections.items() if s.startswith('block.')},
                    )
