"""CLI (основной сценарий — запуск из ноутбука, см. example_agent.ipynb).

    python -m ml_agent tasks/business_type_split.txt
    python -m ml_agent tasks/business_type_split.txt --dry-run
    python -m ml_agent tasks/business_type_split.txt --llm offline
"""
import argparse
import importlib
import sys
from pathlib import Path

from ml_agent import MLAgent


def main(argv=None):
    parser = argparse.ArgumentParser(description='ML-агент поверх mldataworker')
    parser.add_argument('task', help='txt-файл с заданием')
    parser.add_argument('--project-dir', default='.', help='корень проекта (где config.py, casco_automl.py)')
    parser.add_argument('--config', default='config', help='имя модуля config проекта')
    parser.add_argument('--llm', help='переопределить llm_model_type (например offline)')
    parser.add_argument('--dry-run', action='store_true', help='только проверить данные и план, без обучения')
    args = parser.parse_args(argv)

    sys.path.insert(0, str(Path(args.project_dir).resolve()))
    config = importlib.import_module(args.config)
    agent = MLAgent(args.task, config=config, project_dir=args.project_dir, llm_model_type=args.llm)
    if args.dry_run:
        print(agent.dry_run())
        return 0
    report = agent.run()
    print(report.recommendation)
    print(f'Отчёт: {agent.run_dir / "report.html"}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
