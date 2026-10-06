"""ML-агент поверх mldataworker.

Читает задание из txt, готовит данные и копии конфигов, запускает стандартные инструменты
обучения (``CascoFLAutoML`` / ``AutoMLManager``), чинит конфиги при ошибках с помощью LLM,
сравнивает модели и отправляет итоговый отчёт письмом. Все настройки — в ``config.py`` проекта.
"""
import sys

from loguru import logger

from ml_agent.agent import MLAgent
from ml_agent.report import AgentReport
from ml_agent.task import TaskSpec, load_task

__all__ = ['MLAgent', 'AgentReport', 'TaskSpec', 'load_task', 'setup_logging']


def setup_logging(level: str = 'INFO', agent_only: bool = True):
    """Настраивает вывод loguru в ноутбуке.

    :param agent_only: показывать только сообщения агента и LLM (mldataworker пишет очень подробно);
        полный лог каждого запуска всё равно сохраняется в папке эксперимента.
    """
    from ml_agent.agent import AGENT_LOG_PREFIXES
    logger.remove()
    logger.add(sys.stdout, level=level, colorize=False,
               format='{time:HH:mm:ss} | {level: <7} | {message}',
               filter=(lambda r: r['message'].startswith(AGENT_LOG_PREFIXES)) if agent_only else None)
