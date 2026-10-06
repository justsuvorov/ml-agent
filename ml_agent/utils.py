"""Общие утилиты агента."""
import importlib


def import_object(path: str):
    """``package.module:attr`` или ``package.module.attr`` -> объект."""
    module_name, _, attr = path.partition(':') if ':' in path else path.rpartition('.')
    module = importlib.import_module(module_name)
    return getattr(module, attr) if attr else module


def round_floats(obj, digits: int = 6):
    """Рекурсивно округляет float в словарях (для компактных отчётов и промптов)."""
    if isinstance(obj, dict):
        return {k: round_floats(v, digits) for k, v in obj.items()}
    if isinstance(obj, float):
        return round(obj, digits)
    return obj
